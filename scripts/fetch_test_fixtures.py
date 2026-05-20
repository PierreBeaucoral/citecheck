#!/usr/bin/env python3
"""Fetch openly-licensed PDFs to data/fixtures/ for integration testing.

Usage:
    uv run python scripts/fetch_test_fixtures.py 1803.09015 2108.12419
    uv run python scripts/fetch_test_fixtures.py https://arxiv.org/abs/1803.09015

The fixtures themselves are gitignored (data/fixtures/*.pdf). This script lives
in version control so anyone running the test suite can populate fixtures with
one command.

Why arXiv: stable URLs, openly licensed for local use, and the corpus we care
about (econ/CS preprints) is well-represented. If you want a PMC OA paper,
just hand-download it to data/fixtures/ — the integration test only cares
that data/fixtures/ contains a .pdf.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"
USER_AGENT = "citecheck-test-fixtures/0.0.1 (mailto:pbeauco@gmail.com)"

# arXiv accepts both the modern (YYMM.NNNNN) and legacy (subject-class/YYMMNNN) forms.
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?"  # optional URL prefix
    r"(?:arXiv:)?"  # optional bare prefix
    r"(?P<id>\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)


def parse_arxiv_id(token: str) -> str:
    """Extract a canonical arXiv ID from any of: bare ID, abs URL, pdf URL, arXiv:ID."""
    match = _ARXIV_ID_RE.search(token)
    if not match:
        raise ValueError(f"Could not parse arXiv ID from: {token!r}")
    return match.group("id")


def fetch_arxiv_pdf(arxiv_id: str, dest_dir: Path, *, timeout_s: float = 60.0) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = arxiv_id.replace("/", "_")
    target = dest_dir / f"{safe_name}.pdf"
    if target.exists() and target.stat().st_size > 0:
        print(f"  skip  {target.name} (already present)")
        return target

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=timeout_s, headers=headers, follow_redirects=True) as client:
        resp = client.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"arXiv returned {resp.status_code} for {arxiv_id}: {resp.text[:120]}")
    # arXiv occasionally returns an HTML "preparing" page if the PDF is being built.
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(
            f"arXiv response for {arxiv_id} is not a PDF (first bytes: "
            f"{resp.content[:16]!r}). Try again in a few seconds."
        )
    target.write_bytes(resp.content)
    print(f"  ok    {target.name} ({len(resp.content) / 1024:.0f} KB)")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "ids",
        nargs="*",
        help="One or more arXiv IDs or URLs. Examples: 1803.09015, "
        "https://arxiv.org/abs/2108.12419",
    )
    parser.add_argument(
        "--polite-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between requests (default 1.0).",
    )
    args = parser.parse_args()

    if not args.ids:
        parser.error(
            "Pass at least one arXiv ID. Example: "
            "uv run python scripts/fetch_test_fixtures.py 1803.09015"
        )

    print(f"Fetching {len(args.ids)} fixture(s) into {FIXTURES_DIR.relative_to(REPO_ROOT)}")
    failures: list[str] = []
    for i, token in enumerate(args.ids):
        try:
            arxiv_id = parse_arxiv_id(token)
        except ValueError as exc:
            print(f"  err   {exc}")
            failures.append(token)
            continue
        try:
            fetch_arxiv_pdf(arxiv_id, FIXTURES_DIR)
        except (httpx.HTTPError, RuntimeError) as exc:
            print(f"  err   {arxiv_id}: {exc}")
            failures.append(token)
        if i < len(args.ids) - 1:
            time.sleep(args.polite_delay)

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
