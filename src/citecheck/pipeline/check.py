"""End-to-end Phase 2 pipeline: PDF -> extracted, resolved, retraction-checked."""

from __future__ import annotations

import logging
from pathlib import Path

from citecheck.checks.cache import CacheStore
from citecheck.checks.retractions import check_retraction
from citecheck.models import CheckedReference
from citecheck.pipeline.extract import run as run_extract

log = logging.getLogger(__name__)


def run(pdf_path: str | Path, *, use_cache: bool = True) -> list[CheckedReference]:
    """Extract references, resolve them, and run per-reference checks.

    Currently only the retraction check runs; later phases append hallucination,
    journal-quality, and claim-verification checks here.
    """
    references = run_extract(pdf_path)
    cache = CacheStore() if use_cache else None
    try:
        return [
            CheckedReference(
                reference=ref,
                retraction=check_retraction(ref, cache=cache),
            )
            for ref in references
        ]
    finally:
        if cache is not None:
            cache.close()
