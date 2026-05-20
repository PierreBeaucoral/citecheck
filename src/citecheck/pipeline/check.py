"""End-to-end Phase 2 pipeline: PDF -> extracted, resolved, retraction-checked."""

from __future__ import annotations

import logging
from pathlib import Path

from citecheck.checks.cache import CacheStore
from citecheck.checks.hallucination import check_hallucination
from citecheck.checks.retractions import check_retraction
from citecheck.models import CheckedReference, HallucinationCheck, RetractionCheck
from citecheck.pipeline.extract import run as run_extract

log = logging.getLogger(__name__)


def run(
    pdf_path: str | Path,
    *,
    use_cache: bool = True,
    skip_retraction: bool = False,
    skip_hallucination: bool = False,
) -> list[CheckedReference]:
    """Extract references, resolve them, and run per-reference checks.

    Phase 2 added `retraction`; Phase 3 adds `hallucination`. The skip flags let
    CLI consumers turn off either check (e.g. --only-hallucination disables the
    retraction pass).
    """
    references = run_extract(pdf_path)
    cache = CacheStore() if use_cache else None
    try:
        out: list[CheckedReference] = []
        for ref in references:
            retr = RetractionCheck() if skip_retraction else check_retraction(ref, cache=cache)
            hall = (
                HallucinationCheck()
                if skip_hallucination
                else check_hallucination(ref, cache=cache)
            )
            out.append(CheckedReference(reference=ref, retraction=retr, hallucination=hall))
        return out
    finally:
        if cache is not None:
            cache.close()
