#!/usr/bin/env python3
"""Evaluate Phase 4 (journal quality) against data/eval/journal_quality_labels.csv.

Loads the hand-curated label set, runs `check_journal_quality` on each entry,
and writes a markdown + JSON report and a per-row predictions file.

The label set carries three classes:
- high   : known-predatory publisher (Beall's list 2017 + multi-source)
- low    : DOAJ-indexed open-access journal
- medium : legitimate non-OA journal (not in DOAJ; mainstream subscription)

We score the binary classification "predicted high" vs "ground truth high"
(the most operator-relevant question: did the tool flag a predatory venue?).
We also report the per-class confusion to expose the LOW vs MEDIUM split,
because that division is where DOAJ coverage limits show up most clearly.

Usage:
    uv run python scripts/run_journal_eval.py
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
from citecheck.checks.journal_quality import check_journal_quality  # noqa: E402
from citecheck.models import (  # noqa: E402
    RawReference,
    Reference,
    ResolutionStatus,
)

DEFAULT_LABELS_CSV = REPO_ROOT / "data" / "eval" / "journal_quality_labels.csv"
DEFAULT_RESULTS_MD = REPO_ROOT / "data" / "eval" / "journal_results.md"
DEFAULT_RESULTS_JSON = REPO_ROOT / "data" / "eval" / "journal_results.json"


def _ref(journal: str) -> Reference:
    return Reference(
        raw=RawReference(raw_text="x", title="t", journal=journal),
        status=ResolutionStatus.RESOLVED,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Path to the labels CSV (default: data/eval/journal_quality_labels.csv).",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output markdown report path. Derived from --labels if omitted.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path. Derived from --labels if omitted.",
    )
    args = parser.parse_args()

    labels_csv = args.labels.resolve()
    if not labels_csv.is_file():
        print(f"missing: {labels_csv}", file=sys.stderr)
        return 2

    # Derive default output paths from the labels filename so the calibration
    # and held-out runs produce distinct artifacts side by side.
    stem = labels_csv.stem
    if stem == "journal_quality_labels":
        results_md = args.out_md or DEFAULT_RESULTS_MD
        results_json = args.out_json or DEFAULT_RESULTS_JSON
    else:
        # e.g. journal_quality_holdout -> journal_results_holdout.md
        suffix = stem.replace("journal_quality_", "").replace("labels", "")
        suffix = suffix.strip("_") or "alt"
        results_md = args.out_md or (labels_csv.parent / f"journal_results_{suffix}.md")
        results_json = args.out_json or (labels_csv.parent / f"journal_results_{suffix}.json")

    with labels_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"loaded {len(rows)} journals", file=sys.stderr)

    cache = CacheStore()
    predictions: list[dict] = []
    # 3x3 confusion: rows = expected, cols = predicted
    confusion: dict[tuple[str, str], int] = {}

    start = time.time()
    try:
        for i, row in enumerate(rows, 1):
            check = check_journal_quality(_ref(row["journal_name"]), cache=cache)
            expected = row["expected_risk_level"]
            predicted = check.risk_level.value
            confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1
            predictions.append(
                {
                    "id": row["id"],
                    "journal_name": row["journal_name"],
                    "expected": expected,
                    "predicted": predicted,
                    "doaj_listed": check.doaj_listed,
                    "on_concern_list": check.on_concern_list,
                    "matched_publisher": check.matched_publisher,
                    "match": expected == predicted,
                }
            )
            if i % 5 == 0:
                print(
                    f"  {i:3d}/{len(rows)} ({i / len(rows):.0%}) elapsed {time.time() - start:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        cache.close()

    # Binary scoring: positive = predicted HIGH.
    tp = sum(1 for p in predictions if p["expected"] == "high" and p["predicted"] == "high")
    fn = sum(1 for p in predictions if p["expected"] == "high" and p["predicted"] != "high")
    fp = sum(1 for p in predictions if p["expected"] != "high" and p["predicted"] == "high")
    tn = sum(1 for p in predictions if p["expected"] != "high" and p["predicted"] != "high")

    def _safe_div(a: int, b: int) -> float:
        return a / b if b else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    summary = {
        "n": len(rows),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "confusion_3x3": {f"{e}->{p}": c for (e, p), c in sorted(confusion.items())},
    }

    results_json.write_text(
        json.dumps({"summary": summary, "predictions": predictions}, indent=2), encoding="utf-8"
    )
    md = [
        "# Citecheck Phase 4 (journal quality) eval results",
        "",
        f"Evaluated {len(rows)} journals from `{labels_csv.relative_to(REPO_ROOT)}`.",
        "",
        "## Binary classification: predicted HIGH-risk",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {precision:.3f} |",
        f"| Recall | {recall:.3f} |",
        f"| False-positive rate | {fpr:.3f} |",
        f"| F1 | {f1:.3f} |",
        "",
        "## 3x3 confusion matrix (expected -> predicted)",
        "",
        "| Expected \\ Predicted | high | medium | low | unchecked |",
        "|---|---|---|---|---|",
    ]
    for expected in ("high", "medium", "low"):
        cells = [
            str(confusion.get((expected, predicted), 0))
            for predicted in ("high", "medium", "low", "unchecked")
        ]
        md.append(f"| {expected} | {' | '.join(cells)} |")
    results_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        f"DONE. precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f} f1={f1:.3f}",
        file=sys.stderr,
    )
    print(f"  results: {results_md} and {results_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
