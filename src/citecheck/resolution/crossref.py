"""Crossref REST client.

Two entry points:
- resolve_by_doi(doi) -> Reference: direct lookup of /works/{doi}
- resolve_by_metadata(raw) -> Reference: bibliographic query when no DOI is available

We use the Crossref "polite pool" (User-Agent with a mailto) which gets us
better rate limits without registering. The contact email comes from
CITECHECK_CONTACT_EMAIL in .env — see .env.example.

Scoring policy for metadata-based resolution:
- score >= 0.85 -> RESOLVED
- score in [0.65, 0.85) -> AMBIGUOUS (top candidate is taken but flagged for review)
- score < 0.65 -> UNRESOLVED

Score = title similarity + small year-agreement bonus (max ~0.05). Scores are
uncapped during ranking so the gap between candidates is preserved; the Reference
field caps to [0, 1] at storage time.

We do NOT do explicit tie-breaking between close candidates. Crossref ranks by
its own relevance signal (citation count, indexing weight), and the top hit is
usually correct — even when a runner-up scores nearly as well, it is typically
a preprint/published variant of the same paper. Distinguishing "preprint variant"
from "true ambiguity" needs more signals (author overlap, DOI registrant, journal
ISSN) and belongs in Phase 3.
"""

from __future__ import annotations

import logging
import os
from difflib import SequenceMatcher
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from citecheck import __version__
from citecheck.models import RawReference, Reference, ResolutionStatus, normalize_doi

log = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"
DEFAULT_TIMEOUT_S = 30.0
MAX_CANDIDATES = 5

# Scoring thresholds — see module docstring.
RESOLVED_THRESHOLD = 0.85
AMBIGUOUS_THRESHOLD = 0.65


def _user_agent() -> str:
    """Build the polite-pool User-Agent. Falls back to an anonymous UA if no email is set."""
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    repo = "https://github.com/pierrebeaucoral/citecheck"
    if email and "@" in email:
        return f"citecheck/{__version__} (mailto:{email}; +{repo})"
    return f"citecheck/{__version__} (+{repo})"


def _client(timeout_s: float) -> httpx.Client:
    return httpx.Client(
        base_url=CROSSREF_BASE,
        timeout=timeout_s,
        headers={"User-Agent": _user_agent(), "Accept": "application/json"},
    )


def _retryable_get(client: httpx.Client, url: str, **params: Any) -> httpx.Response:
    @retry(
        retry=retry_if_exception_type(httpx.RequestError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _do() -> httpx.Response:
        return client.get(url, params=params)

    return _do()


def _slim_work(work: dict[str, Any]) -> dict[str, Any]:
    """Pick the fields we want to keep on Reference.resolved_metadata.

    Crossref work records are large (hundreds of fields). Storing the full record
    bloats reports and pollutes JSON output. We keep the subset useful for the
    later check layers (retraction, journal-quality, claim verification).
    """
    issued = (work.get("issued") or {}).get("date-parts") or [[]]
    year = issued[0][0] if issued and issued[0] else None
    container = work.get("container-title") or []
    return {
        "doi": work.get("DOI"),
        "type": work.get("type"),
        "title": (work.get("title") or [None])[0],
        "container_title": container[0] if container else None,
        "issued_year": year,
        "volume": work.get("volume"),
        "issue": work.get("issue"),
        "page": work.get("page"),
        "publisher": work.get("publisher"),
        "is_referenced_by_count": work.get("is-referenced-by-count"),
        # Crossref signals retractions and corrections here — Phase 2 will read this.
        "update_to": work.get("update-to"),
    }


def _title_similarity(claimed: str | None, candidate: str | None) -> float:
    """Cheap, dependency-free fuzzy title match.

    We lowercase and collapse whitespace before comparing, which fixes most of the
    casing/spacing noise. SequenceMatcher is O(N*M) but titles are short (<200 chars).
    rapidfuzz would be faster but is a Phase 3 dependency we do not need here.
    """
    if not claimed or not candidate:
        return 0.0
    a = " ".join(claimed.lower().split())
    b = " ".join(candidate.lower().split())
    return SequenceMatcher(None, a, b).ratio()


def score_slim(raw: RawReference, slim: dict[str, Any]) -> float:
    """Source-agnostic candidate scoring.

    Takes a RawReference and a slim dict (the common-schema record produced by
    `_slim_work` for Crossref or `_slim_openalex_work` for OpenAlex). Returns an
    uncapped score so the caller can compare top/runner-up gaps before storing
    a [0, 1]-clamped value on Reference.
    """
    title_score = _title_similarity(raw.title, slim.get("title"))
    # Year agreement is a small additive bonus, not the main signal: titles are
    # far more discriminative than years (many papers in any year share titles).
    year_bonus = 0.0
    if raw.year and slim.get("issued_year"):
        delta = abs(raw.year - slim["issued_year"])
        if delta == 0:
            year_bonus = 0.05
        elif delta == 1:
            year_bonus = 0.02
        elif delta > 3:
            year_bonus = -0.10  # punish clearly-wrong years
    return max(0.0, title_score + year_bonus)


def _score_candidate(raw: RawReference, work: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Crossref-shaped candidate scoring. Wraps `score_slim` with the Crossref adapter."""
    slim = _slim_work(work)
    return score_slim(raw, slim), slim


def resolve_by_doi(raw: RawReference, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Reference:
    """Look up a known DOI. RawReference.doi must already be normalized."""
    doi = normalize_doi(raw.doi)
    if not doi:
        return Reference(
            raw=raw, status=ResolutionStatus.UNRESOLVED, notes=["resolve_by_doi: no DOI on input"]
        )

    with _client(timeout_s) as client:
        try:
            resp = _retryable_get(client, f"/works/{doi}")
        except httpx.RequestError as exc:
            log.warning("Crossref unreachable for DOI %s: %s", doi, exc)
            return Reference(
                raw=raw,
                status=ResolutionStatus.ERROR,
                notes=[f"resolve_by_doi: network error: {exc}"],
            )

    if resp.status_code == 404:
        return Reference(
            raw=raw,
            status=ResolutionStatus.UNRESOLVED,
            notes=[f"resolve_by_doi: Crossref 404 for {doi}"],
        )
    if resp.status_code != 200:
        return Reference(
            raw=raw,
            status=ResolutionStatus.ERROR,
            notes=[f"resolve_by_doi: Crossref {resp.status_code}"],
        )

    work = (resp.json() or {}).get("message") or {}
    return Reference(
        raw=raw,
        status=ResolutionStatus.RESOLVED,
        resolved_doi=work.get("DOI"),
        resolved_metadata=_slim_work(work),
        resolution_score=None,  # DOI matches are exact
    )


def resolve_by_metadata(raw: RawReference, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Reference:
    """Query Crossref bibliographically. Used when a reference has no DOI."""
    if not raw.title:
        return Reference(
            raw=raw,
            status=ResolutionStatus.UNRESOLVED,
            notes=["resolve_by_metadata: no title on input"],
        )

    params: dict[str, Any] = {
        "query.bibliographic": raw.title,
        "rows": MAX_CANDIDATES,
    }
    if raw.authors:
        params["query.author"] = raw.authors[0].family

    with _client(timeout_s) as client:
        try:
            resp = _retryable_get(client, "/works", **params)
        except httpx.RequestError as exc:
            log.warning("Crossref metadata query failed for %r: %s", raw.title, exc)
            return Reference(
                raw=raw,
                status=ResolutionStatus.ERROR,
                notes=[f"resolve_by_metadata: network error: {exc}"],
            )

    if resp.status_code != 200:
        return Reference(
            raw=raw,
            status=ResolutionStatus.ERROR,
            notes=[f"resolve_by_metadata: Crossref {resp.status_code}"],
        )

    items = ((resp.json() or {}).get("message") or {}).get("items") or []
    if not items:
        return Reference(
            raw=raw,
            status=ResolutionStatus.UNRESOLVED,
            notes=["resolve_by_metadata: no candidates"],
        )

    scored = sorted(
        (_score_candidate(raw, w) for w in items),
        key=lambda t: t[0],
        reverse=True,
    )
    top_score, top_slim = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    stored_score = min(1.0, top_score)  # Reference.resolution_score is in [0, 1]

    if top_score >= RESOLVED_THRESHOLD:
        return Reference(
            raw=raw,
            status=ResolutionStatus.RESOLVED,
            resolved_doi=top_slim["doi"],
            resolved_metadata=top_slim,
            resolution_score=stored_score,
        )
    if top_score >= AMBIGUOUS_THRESHOLD:
        return Reference(
            raw=raw,
            status=ResolutionStatus.AMBIGUOUS,
            resolved_doi=top_slim["doi"],
            resolved_metadata=top_slim,
            resolution_score=stored_score,
            notes=[
                f"resolve_by_metadata: top score {top_score:.2f}, "
                f"runner-up {second_score:.2f} — review before relying"
            ],
        )
    return Reference(
        raw=raw,
        status=ResolutionStatus.UNRESOLVED,
        resolution_score=stored_score,
        notes=[f"resolve_by_metadata: best score {top_score:.2f} below threshold"],
    )


# NOTE: the top-level `resolve()` orchestrator moved to citecheck.resolution.__init__
# so it can fan out to both Crossref and OpenAlex without creating a circular
# import. crossref.py stays focused on the Crossref REST client.
