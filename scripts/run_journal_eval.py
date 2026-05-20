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

LABELS_CSV = REPO_ROOT / "data" / "eval" / "journal_quality_labels.csv"
RESULTS_MD = REPO_ROOT / "data" / "eval" / "journal_results.md"
RESULTS_JSON = REPO_ROOT / "data" / "eval" / "journal_results.json"


def _ref(journal: str) -> Reference:
    return Reference(
        raw=RawReference(raw_text="x", title="t", journal=journal),
        status=ResolutionStatus.RESOLVED,
    )


def main() -> int:
    if not LABELS_CSV.is_file():
        print(f"missing: {LABELS_CSV}", file=sys.stderr)
        return 2

    with LABELS_CSV.open(encoding="utf-8") as fh:
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

    RESULTS_JSON.write_text(
        json.dumps({"summary": summary, "predictions": predictions}, indent=2), encoding="utf-8"
    )
    md = [
        "# Citecheck Phase 4 (journal quality) eval results",
        "",
        f"Evaluated {len(rows)} journals from the hand-curated set at "
        "`data/eval/journal_quality_labels.csv`.",
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
    RESULTS_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(
        f"DONE. precision={precision:.3f} recall={recall:.3f} fpr={fpr:.3f} f1={f1:.3f}",
        file=sys.stderr,
    )
    print(f"  results: {RESULTS_MD} and {RESULTS_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
