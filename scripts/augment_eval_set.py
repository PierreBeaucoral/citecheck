#!/usr/bin/env python3
"""Append real-reference rows to data/eval/labels.csv from local PDF fixtures.

Why: the primary eval source (Walters & Wilder 2023) is humanities-heavy and
book-heavy. Adding refs from the econ-fixture PDFs in data/fixtures/ gives the
eval a second domain slice for FPR measurement. The added rows are all real by
construction — they appear in real published arXiv papers — so their expected
verdict is `real_high_confidence` (or `real_low_confidence` if no DOI).

We tag every added row with `class=real_fixture` and `labeler=<arxiv-id>` so
the eval report can break out per-source metrics. These refs are NOT a
substitute for an independent labeled corpus (citecheck already resolved them
during Phase 1, so reporting "accuracy" on them is partly tautological), but
they ARE useful for measuring FPR on a different distribution from Walters.

Usage:
    docker compose up -d grobid
    uv run python scripts/augment_eval_set.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from citecheck.extraction.grobid_client import extract_references, is_alive  # noqa: E402
from citecheck.resolution import resolve  # noqa: E402

LABELS_CSV = REPO_ROOT / "data" / "eval" / "labels.csv"
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"

# How to interpret each fixture for the eval set. Keys are PDF stems; values
# are the human-readable labeler tag and source notes.
FIXTURE_TAGS = {
    "1803.09015": ("callaway_santanna_2021", "DiD with multiple time periods (econ)"),
    "2108.12419": ("borusyak_jaravel_spiess_2024", "Revisiting event study designs (econ)"),
}


def _existing_ids() -> set[str]:
    if not LABELS_CSV.exists():
        return set()
    with LABELS_CSV.open(encoding="utf-8") as fh:
        return {row["id"] for row in csv.DictReader(fh)}


def main() -> int:
    if not is_alive():
        print(
            "GROBID is not reachable at http://localhost:8070.\n"
            "Start it with: docker compose up -d grobid",
            file=sys.stderr,
        )
        return 3

    pdfs = sorted(FIXTURES_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"no fixtures in {FIXTURES_DIR}", file=sys.stderr)
        return 2

    existing = _existing_ids()
    new_rows: list[list[str]] = []

    for pdf in pdfs:
        stem = pdf.stem
        labeler, notes = FIXTURE_TAGS.get(stem, (stem, ""))
        print(f"== {pdf.name} ==", file=sys.stderr)
        raws = extract_references(pdf)
        print(f"  extracted {len(raws)} refs", file=sys.stderr)
        kept = 0
        for j, raw in enumerate(raws, 1):
            row_id = f"fixture_{stem}_{j:03d}"
            if row_id in existing:
                continue
            ref = resolve(raw)
            if not raw.title:
                # Skip refs with no parseable title — nothing meaningful to evaluate.
                continue
            # Expected verdict: high if DOI-resolved, low otherwise. Citecheck
            # should never flag a real reference as suspicious — that's the FPR
            # we care about.
            expected = "real_high_confidence" if ref.resolved_doi else "real_low_confidence"
            authors_str = "; ".join(f"{a.family},{a.given or ''}" for a in raw.authors)
            new_rows.append(
                [
                    row_id,
                    raw.raw_text,
                    "real_fixture",
                    "",  # ground_truth_doi
                    expected,
                    stem if stem.replace(".", "").isdigit() else "",  # arxiv_id heuristic
                    "",  # biorxiv_id
                    f"{labeler}; {notes}",
                    labeler,
                    labeler,
                    raw.title or "",
                    raw.year or "",
                    raw.journal or "",
                    authors_str,
                    raw.doi or "",
                ]
            )
            kept += 1
        print(f"  kept {kept} (with parseable titles)", file=sys.stderr)

    if not new_rows:
        print("no new rows to append", file=sys.stderr)
        return 0

    with LABELS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(new_rows)
    print(f"appended {len(new_rows)} rows to {LABELS_CSV.relative_to(REPO_ROOT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
