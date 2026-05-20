"""End-to-end Phase 1 pipeline: PDF -> extracted, resolved references."""

from __future__ import annotations

import logging
from pathlib import Path

from citecheck.extraction.grobid_client import extract_references
from citecheck.models import Reference
from citecheck.resolution import resolve

log = logging.getLogger(__name__)


def run(pdf_path: str | Path) -> list[Reference]:
    """Extract references from a PDF and resolve each against Crossref.

    Resolution is sequential by design: Crossref's polite-pool rate guidance is
    ~50 req/s and a typical paper has 30-80 references, so we are comfortably
    inside the budget without thread pools.
    """
    raws = extract_references(pdf_path)
    log.info("GROBID returned %d references", len(raws))
    return [resolve(r) for r in raws]
