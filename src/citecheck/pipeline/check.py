"""End-to-end Phase 2 pipeline: PDF -> extracted, resolved, retraction-checked."""

from __future__ import annotations

import logging
from pathlib import Path

from citecheck.checks.cache import CacheStore
from citecheck.checks.hallucination import check_hallucination
from citecheck.checks.journal_quality import check_journal_quality
from citecheck.checks.retractions import check_retraction
from citecheck.extraction.fulltext import extract_claims
from citecheck.models import (
    CheckedReference,
    HallucinationCheck,
    JournalQualityCheck,
    RetractionCheck,
)
from citecheck.pipeline.extract import run as run_extract

log = logging.getLogger(__name__)


def run(
    pdf_path: str | Path,
    *,
    use_cache: bool = True,
    skip_retraction: bool = False,
    skip_hallucination: bool = False,
    skip_journal_quality: bool = False,
    verify_claims: bool = False,
) -> list[CheckedReference]:
    """Extract references, resolve them, and run per-reference checks.

    Phase 2 added `retraction`; Phase 3 adds `hallucination`. Phase 5 adds
    `claims` (opt-in via `verify_claims=True`) — for each in-text citation
    location, asks a local LLM whether the cited paper supports the claim.
    Phase 5 is opt-in because it is slow (minutes per paper) and requires
    Ollama + sentence-transformers, which are heavy optional dependencies.
    """
    references = run_extract(pdf_path)
    cache = CacheStore() if use_cache else None

    # Phase 5: fulltext claim extraction is a single GROBID call regardless of
    # the number of references, so we do it once up-front and then build a
    # ref_id -> [claim_sentence, ...] map. ref_id matches `RawReference.ref_id`.
    claims_by_ref: dict[str, list[str]] = {}
    if verify_claims:
        try:
            for ref_id, claim in extract_claims(pdf_path):
                claims_by_ref.setdefault(ref_id, []).append(claim)
        except Exception:
            claims_by_ref = {}

    try:
        out: list[CheckedReference] = []
        for ref in references:
            retr = RetractionCheck() if skip_retraction else check_retraction(ref, cache=cache)
            hall = (
                HallucinationCheck()
                if skip_hallucination
                else check_hallucination(ref, cache=cache)
            )
            jq = (
                JournalQualityCheck()
                if skip_journal_quality
                else check_journal_quality(ref, cache=cache)
            )
            claim_results = []
            if verify_claims and ref.raw.ref_id in claims_by_ref:
                # Lazy import: keeps the regular pipeline free of heavy deps.
                from citecheck.checks.claims import check_claims

                claim_results = check_claims(ref, claims_by_ref[ref.raw.ref_id], cache=cache)
            out.append(
                CheckedReference(
                    reference=ref,
                    retraction=retr,
                    hallucination=hall,
                    journal_quality=jq,
                    claims=claim_results,
                )
            )
        return out
    finally:
        if cache is not None:
            cache.close()
