#!/usr/bin/env python3
"""Evaluate Phase 2 (retraction check) on labeled retracted + clean DOIs.

The eval set is sampled live from OpenAlex (which sources its `is_retracted`
boolean from the Retraction Watch dataset via the Crossref / RW partnership)
and persisted to `data/eval/retraction_labels.csv` so re-runs use a stable
ground-truth set.  Sampling design:

* 30 retracted DOIs sampled from `is_retracted:true` with `cited_by_count:>50`
  (so we exclude obscure retractions with weak provenance).
* 30 clean DOIs sampled from `is_retracted:false` with `cited_by_count:>500`
  (highly-cited papers are unlikely to be silently retracted in the future).

For each DOI the runner:

  1. Resolves the DOI through citecheck's Crossref resolver, which populates
     the `update_to` field on the reference (the data the retraction check
     reads on its "Path A — Crossref" branch).
  2. Calls `check_retraction(reference)` — the same function the production
     pipeline uses.
  3. Records the verdict, which database produced the signal (Crossref vs
     OpenAlex), and the elapsed time.

Scoring is binary: positive = predicted RETRACTED or EXPRESSION_OF_CONCERN.
We also report the Crossref-coverage breakdown: of the retracted items that
the tool correctly flagged, how many were caught by Crossref's `update-to`
versus only by OpenAlex's `is_retracted`?  This is the substantive question
about why citecheck combines two sources.

There is one self-consistency caveat the paper should report.  The retracted
DOIs are sourced from OpenAlex's flag, and the tool consults the same flag,
so on the OpenAlex branch a 100% match is guaranteed by construction.  The
non-trivial measurement is the Crossref-vs-OpenAlex coverage breakdown and
the false-positive rate on the clean DOIs.

Usage:
    uv run python scripts/run_retraction_eval.py
    uv run python scripts/run_retraction_eval.py --resample  # re-source DOIs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

import httpx  # noqa: E402

from citecheck import budget  # noqa: E402
from citecheck.checks.cache import CacheStore  # noqa: E402
from citecheck.checks.retractions import check_retraction  # noqa: E402
from citecheck.models import (  # noqa: E402
    RawReference,
    RetractionStatus,
)
from citecheck.resolution.crossref import resolve_by_doi  # noqa: E402

LABELS_CSV = REPO_ROOT / "data" / "eval" / "retraction_labels.csv"
RESULTS_JSON = REPO_ROOT / "data" / "eval" / "retraction_results.json"
RESULTS_MD = REPO_ROOT / "data" / "eval" / "retraction_results.md"

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_N_RETRACTED = 30
DEFAULT_N_CLEAN = 30
SAMPLE_TIMEOUT_S = 30.0

log = logging.getLogger("retraction_eval")


def _openalex_polite_params() -> dict[str, str]:
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    return {"mailto": email} if email and "@" in email else {}


def _sample_dois(
    *, retracted: bool, n: int, min_citations: int, timeout_s: float = SAMPLE_TIMEOUT_S
) -> list[str]:
    """Pull a sample of DOIs from OpenAlex matching the retraction filter."""
    flag = "true" if retracted else "false"
    params = {
        "filter": f"is_retracted:{flag},cited_by_count:>{min_citations},has_doi:true",
        "select": "id,doi,is_retracted,cited_by_count,publication_year",
        "per-page": str(n),
        # Stable but spread-out sampling: order by citation count then page
        # through the result.  No randomization seed because that's not
        # supported by OpenAlex's API; the same call returns the same set.
        "sort": "cited_by_count:desc",
        **_openalex_polite_params(),
    }
    url = f"{OPENALEX_BASE}/works"
    headers = {"Accept": "application/json"}
    out: list[str] = []
    with httpx.Client(timeout=timeout_s, headers=headers) as client:
        resp = budget.openalex_get(client, url, **params)
        if resp is None or resp.status_code != 200:
            log.error(
                "sampling failed (retracted=%s): %s",
                retracted,
                resp.status_code if resp is not None else "circuit-open",
            )
            return []
        body = resp.json() or {}
        for hit in body.get("results", []):
            doi = (hit.get("doi") or "").removeprefix("https://doi.org/").lower()
            if doi and doi.startswith("10."):
                out.append(doi)
    return out[:n]


def build_labels(
    *,
    n_retracted: int = DEFAULT_N_RETRACTED,
    n_clean: int = DEFAULT_N_CLEAN,
    out_csv: Path = LABELS_CSV,
) -> int:
    """Sample DOIs and persist a labels CSV.  Idempotent: rewrites the file."""
    retracted = _sample_dois(retracted=True, n=n_retracted, min_citations=50)
    clean = _sample_dois(retracted=False, n=n_clean, min_citations=500)
    if not retracted or not clean:
        print("FATAL: sampling returned no DOIs (budget? network?)", file=sys.stderr)
        return 2
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["doi", "expected_status", "source"])
        for d in retracted:
            writer.writerow([d, "retracted", "openalex_is_retracted_true"])
        for d in clean:
            writer.writerow([d, "clean", "openalex_is_retracted_false_high_cite"])
    print(
        f"wrote {len(retracted) + len(clean)} labels to {out_csv}",
        file=sys.stderr,
    )
    return 0


def _predicted_label(status: RetractionStatus) -> str | None:
    """Binary mapping for scoring: positive = predicted RETRACTED.

    EXPRESSION_OF_CONCERN counts on the same side as RETRACTED (both are
    "flag this citation for the user").  CORRECTION is on the CLEAN side
    (the underlying work was not invalidated).
    """
    if status in (RetractionStatus.RETRACTED, RetractionStatus.EXPRESSION_OF_CONCERN):
        return "retracted"
    if status in (RetractionStatus.CLEAN, RetractionStatus.CORRECTION):
        return "clean"
    return None


def _safe_div(a: int, b: int) -> float:
    return a / b if b else float("nan")


def _format(v: float) -> str:
    return f"{v:.3f}" if v == v else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_CSV,
        help="Path to the labels CSV.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Re-source the labels CSV from OpenAlex before running the eval.",
    )
    parser.add_argument(
        "--n-retracted",
        type=int,
        default=DEFAULT_N_RETRACTED,
        help="Sample size for the retracted half (used only with --resample).",
    )
    parser.add_argument(
        "--n-clean",
        type=int,
        default=DEFAULT_N_CLEAN,
        help="Sample size for the clean half (used only with --resample).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Echo per-item progress to stderr.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.resample or not args.labels.is_file():
        code = build_labels(
            n_retracted=args.n_retracted, n_clean=args.n_clean, out_csv=args.labels
        )
        if code != 0:
            return code

    with args.labels.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"loaded {len(rows)} DOIs from {args.labels.name}", file=sys.stderr)

    cache = CacheStore()
    predictions: list[dict[str, Any]] = []
    start = time.time()
    try:
        for i, row in enumerate(rows, 1):
            doi = row["doi"]
            expected = row["expected_status"]
            t0 = time.time()
            # Resolve through Crossref so the reference carries update_to.
            ref = resolve_by_doi(RawReference(raw_text=doi, doi=doi))
            check = check_retraction(ref, cache=cache)
            elapsed = time.time() - t0

            predicted_label = _predicted_label(check.status)
            match = (predicted_label == expected) if predicted_label else None

            # Identify which source produced the signal: Crossref's update-to
            # would have populated `notice_doi`; OpenAlex-only paths do not.
            crossref_signal = bool(check.notice_doi)
            openalex_signal = (
                check.status == RetractionStatus.RETRACTED and not crossref_signal
            )

            predictions.append(
                {
                    "doi": doi,
                    "expected": expected,
                    "predicted_status": check.status.value,
                    "predicted_label": predicted_label,
                    "match": match,
                    "crossref_signal": crossref_signal,
                    "openalex_signal": openalex_signal,
                    "notice_doi": check.notice_doi,
                    "notice_date": check.notice_date,
                    "reason": check.reason,
                    "elapsed_s": round(elapsed, 2),
                }
            )
            if i % 10 == 0 or i == len(rows):
                wall = time.time() - start
                print(
                    f"  {i:3d}/{len(rows)} elapsed {wall:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        cache.close()

    # ---- Scoring -----------------------------------------------------------
    scored = [p for p in predictions if p["predicted_label"] is not None]
    tp = sum(1 for p in scored if p["predicted_label"] == "retracted" and p["expected"] == "retracted")
    fp = sum(1 for p in scored if p["predicted_label"] == "retracted" and p["expected"] == "clean")
    fn = sum(1 for p in scored if p["predicted_label"] == "clean" and p["expected"] == "retracted")
    tn = sum(1 for p in scored if p["predicted_label"] == "clean" and p["expected"] == "clean")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    fpr = _safe_div(fp, fp + tn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision == precision and recall == recall and (precision + recall) > 0
        else float("nan")
    )

    # Coverage breakdown: of the correctly-flagged retractions, how many had
    # a Crossref update-to entry, and how many were OpenAlex-only?
    flagged_retractions = [
        p for p in scored if p["expected"] == "retracted" and p["predicted_label"] == "retracted"
    ]
    with_crossref = sum(1 for p in flagged_retractions if p["crossref_signal"])
    openalex_only = sum(1 for p in flagged_retractions if p["openalex_signal"])

    summary = {
        "n_total": len(rows),
        "n_scored": len(scored),
        "n_excluded": len(rows) - len(scored),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "f1": f1,
        "coverage": {
            "n_correctly_flagged": len(flagged_retractions),
            "with_crossref_update_to": with_crossref,
            "openalex_only": openalex_only,
            "crossref_fraction": _safe_div(with_crossref, len(flagged_retractions)),
        },
    }

    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "predictions": predictions}, indent=2),
        encoding="utf-8",
    )

    md: list[str] = [
        "# Citecheck Phase 2 (retraction check) eval results",
        "",
        f"Evaluated {len(rows)} DOIs from `{args.labels.relative_to(REPO_ROOT)}`.",
        "",
        "## Headline (binary: predicted RETRACTED)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Items scored | {len(scored)} / {len(rows)} |",
        f"| Precision | {_format(precision)} |",
        f"| Recall | {_format(recall)} |",
        f"| False-positive rate | {_format(fpr)} |",
        f"| F1 | {_format(f1)} |",
        "",
        "## Confusion matrix",
        "",
        "| Expected \\ Predicted | clean | retracted |",
        "|---|---|---|",
        f"| clean | {tn} | {fp} |",
        f"| retracted | {fn} | {tp} |",
        "",
        "## Source coverage (correctly flagged retractions only)",
        "",
        "| Source | n | Fraction |",
        "|---|---|---|",
        f"| Crossref update-to populated | {with_crossref} | {_format(_safe_div(with_crossref, len(flagged_retractions)))} |",
        f"| OpenAlex is_retracted only | {openalex_only} | {_format(_safe_div(openalex_only, len(flagged_retractions)))} |",
    ]
    RESULTS_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        f"DONE. precision={_format(precision)} recall={_format(recall)} "
        f"fpr={_format(fpr)} f1={_format(f1)} crossref_coverage="
        f"{_format(_safe_div(with_crossref, len(flagged_retractions)))}",
        file=sys.stderr,
    )
    print(f"  results: {RESULTS_MD} and {RESULTS_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
