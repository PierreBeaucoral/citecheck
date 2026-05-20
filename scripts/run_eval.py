#!/usr/bin/env python3
"""Run citecheck against the labeled eval set; report precision/recall/FPR.

Loads data/eval/labels.csv, rebuilds a RawReference per row from the parsed
fields written by build_eval_set.py, then runs the full Phase 1 + Phase 3
pipeline (resolve + check_hallucination) and compares the predicted verdict
against the expected verdict.

We treat hallucination detection as a binary classification problem:
    positive class = predicted likely_hallucinated OR suspicious
    negative class = predicted real_high OR real_low

Writes:
    data/eval/results.md     -- human-readable report
    data/eval/results.json   -- per-row predictions for downstream analysis
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from citecheck.checks.cache import CacheStore  # noqa: E402
from citecheck.checks.hallucination import check_hallucination  # noqa: E402
from citecheck.models import (  # noqa: E402
    Author,
    HallucinationVerdict,
    RawReference,
)
from citecheck.resolution import resolve  # noqa: E402

LABELS_CSV = REPO_ROOT / "data" / "eval" / "labels.csv"
RESULTS_MD = REPO_ROOT / "data" / "eval" / "results.md"
RESULTS_JSON = REPO_ROOT / "data" / "eval" / "results.json"
PROGRESS_LOG = REPO_ROOT / "data" / "eval" / "progress.log"

# Binary mapping for precision/recall scoring.
POSITIVE = {HallucinationVerdict.LIKELY_HALLUCINATED, HallucinationVerdict.SUSPICIOUS}
NEGATIVE = {HallucinationVerdict.REAL_HIGH_CONFIDENCE, HallucinationVerdict.REAL_LOW_CONFIDENCE}


def _parse_authors(s: str) -> list[Author]:
    out: list[Author] = []
    for chunk in (s or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "," in chunk:
            family, given = chunk.split(",", 1)
            out.append(Author(family=family.strip(), given=given.strip() or None))
        else:
            out.append(Author(family=chunk))
    return out


def _build_raw(row: dict) -> RawReference:
    year = None
    if row["parsed_year"]:
        try:
            year = int(float(row["parsed_year"]))
        except (TypeError, ValueError):
            year = None
    return RawReference(
        raw_text=row["raw_text"],
        title=row["parsed_title"] or None,
        authors=_parse_authors(row["parsed_authors"]),
        year=year,
        journal=row["parsed_journal"] or None,
        doi=row["parsed_doi"] or None,
    )


def _is_positive_pred(verdict: HallucinationVerdict) -> bool:
    return verdict in POSITIVE


def _expected_is_hallucinated(expected: str) -> bool:
    return expected == "likely_hallucinated"


def main() -> int:
    if not LABELS_CSV.is_file():
        print(f"missing: {LABELS_CSV}", file=sys.stderr)
        return 2

    with LABELS_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"loaded {len(rows)} rows", file=sys.stderr)

    cache = CacheStore()
    predictions: list[dict] = []
    confusion = {("pos", "pos"): 0, ("pos", "neg"): 0, ("neg", "pos"): 0, ("neg", "neg"): 0}

    # Progress log is flushed every row so we can monitor without waiting for buffered output.
    progress_fh = PROGRESS_LOG.open("w", encoding="utf-8", buffering=1)
    progress_fh.write(f"eval started; {len(rows)} rows\n")

    start = time.time()
    try:
        for i, row in enumerate(rows, 1):
            raw = _build_raw(row)
            ref = resolve(raw)
            check = check_hallucination(ref, cache=cache)

            expected_hall = _expected_is_hallucinated(row["expected_verdict"])
            predicted_pos = _is_positive_pred(check.verdict)
            ekey = "pos" if expected_hall else "neg"
            pkey = "pos" if predicted_pos else "neg"
            confusion[(ekey, pkey)] += 1

            # Per-layer flag info enables the firing-rate analysis the
            # writer-critic flagged as missing. We store {layer_name: flagged}
            # so the per-layer table can be computed from this output without
            # re-running the eval.
            layer_flags = {s.layer: bool(s.flagged) for s in check.signals}
            predictions.append(
                {
                    "id": row["id"],
                    "class": row["class"],
                    "expected_verdict": row["expected_verdict"],
                    "predicted_verdict": check.verdict.value,
                    "red_flag_count": check.red_flag_count,
                    "layer_flags": layer_flags,
                    "resolution_status": ref.status.value,
                    "resolved_doi": ref.resolved_doi,
                    "title_truncated": (raw.title or raw.raw_text)[:100],
                    "match": (expected_hall == predicted_pos),
                }
            )

            # Progress: one short line per row to the log; periodic summaries to stderr.
            progress_fh.write(
                f"{i}/{len(rows)} expected={row['expected_verdict']} "
                f"predicted={check.verdict.value} red_flags={check.red_flag_count}\n"
            )
            if i % 25 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta = (len(rows) - i) / rate if rate > 0 else 0
                msg = (
                    f"  {i:4d}/{len(rows)} ({i / len(rows):.0%})  "
                    f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s"
                )
                print(msg, file=sys.stderr, flush=True)
                progress_fh.write(msg + "\n")
    finally:
        progress_fh.close()
        cache.close()

    # Metrics — positive class = predicted hallucinated.
    tp = confusion[("pos", "pos")]  # truly hallucinated, predicted positive (correct catch)
    fn = confusion[("pos", "neg")]  # truly hallucinated, predicted real      (missed)
    fp = confusion[("neg", "pos")]  # truly real, predicted positive          (false alarm)
    tn = confusion[("neg", "neg")]  # truly real, predicted real              (correct pass)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision and recall and precision + recall > 0)
        else float("nan")
    )

    # By predicted verdict (full 5-class breakdown).
    by_verdict: dict[str, dict[str, int]] = {}
    for p in predictions:
        v = p["predicted_verdict"]
        bucket = by_verdict.setdefault(v, {"real": 0, "fabricated": 0})
        if _expected_is_hallucinated(p["expected_verdict"]):
            bucket["fabricated"] += 1
        else:
            bucket["real"] += 1

    # Write JSON.
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "confusion": {f"{k[0]}_{k[1]}": v for k, v in confusion.items()},
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "fpr": fpr,
                    "f1": f1,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                },
                "predictions": predictions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Write Markdown report.
    md = []
    md.append("# Citecheck eval results — Walters & Wilder (2023)\n")
    md.append(
        "Citecheck's 5-layer hallucination detector evaluated against the 636-citation "
        "ChatGPT corpus from Walters & Wilder, *Scientific Reports* 13:14045 (CC-BY).\n"
    )
    md.append(f"Run on {len(rows)} citations; binary classification (hallucinated vs real).\n\n")
    md.append("## Headline metrics (positive class = hallucinated)\n\n")
    md.append("| Metric | Value |\n|---|---|\n")
    md.append(f"| Precision | {precision:.3f} |\n")
    md.append(f"| Recall | {recall:.3f} |\n")
    md.append(f"| False-positive rate | {fpr:.3f} |\n")
    md.append(f"| F1 | {f1:.3f} |\n\n")
    md.append("## Confusion matrix\n\n")
    md.append("|           | Predicted hallucinated | Predicted real |\n|---|---|---|\n")
    md.append(f"| Truly hallucinated | {tp} | {fn} |\n")
    md.append(f"| Truly real         | {fp} | {tn} |\n\n")
    md.append("## Breakdown by predicted verdict\n\n")
    md.append("| Predicted verdict | Truly real | Truly hallucinated |\n|---|---|---|\n")
    for v in [
        "real_high_confidence",
        "real_low_confidence",
        "suspicious",
        "likely_hallucinated",
        "unchecked",
    ]:
        b = by_verdict.get(v, {"real": 0, "fabricated": 0})
        md.append(f"| {v} | {b['real']} | {b['fabricated']} |\n")
    md.append("\n")
    md.append(
        "**Caveat.** The Walters & Wilder corpus is biased toward general humanities / "
        "scholarly topics; ~half are books, which Crossref and OpenAlex index poorly. "
        "The asymmetric low-coverage caveat in citecheck deliberately keeps book-only "
        "refs at real_low_confidence rather than flagging them suspicious — a design "
        "choice that trades a little recall for substantially fewer false positives.\n"
    )
    RESULTS_MD.write_text("".join(md), encoding="utf-8")

    print(
        f"\nDONE. precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f} f1={f1:.3f}",
        file=sys.stderr,
    )
    print(f"  results: {RESULTS_MD.relative_to(REPO_ROOT)} and results.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
