"""OpenAlex resolver — used as a fallback when Crossref metadata search returns UNRESOLVED.

OpenAlex (https://openalex.org) has independent indexing from Crossref and free
public access. We treat it as a second-pass: only call it when Crossref couldn't
find anything close enough. Two reasons to keep it as a fallback rather than a
primary source:

1. Crossref's record is canonical for DOI metadata; when it matches, trust it.
2. OpenAlex is single-source for many works Crossref does not index (preprint
   servers, gray literature, older items) — that is exactly the unresolved tail.

Polite pool: we send `mailto=<email>` as a query parameter, the documented way.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from citecheck import budget
from citecheck.models import RawReference, Reference, ResolutionStatus
from citecheck.resolution.crossref import (
    AMBIGUOUS_THRESHOLD,
    RESOLVED_THRESHOLD,
    score_slim,
)

log = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT_S = 30.0
MAX_CANDIDATES = 5


def _polite_params() -> dict[str, str]:
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    return {"mailto": email} if email and "@" in email else {}


def _client(timeout_s: float) -> httpx.Client:
    return httpx.Client(
        base_url=OPENALEX_BASE,
        timeout=timeout_s,
        headers={"Accept": "application/json"},
    )


def _retryable_get(client: httpx.Client, url: str, **params: Any) -> httpx.Response | None:
    """Retry on network errors; return None when OpenAlex budget is exhausted."""

    @retry(
        retry=retry_if_exception_type(httpx.RequestError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _do() -> httpx.Response | None:
        return budget.openalex_get(client, url, **params)

    return _do()


def _slim_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    """Adapt an OpenAlex work record to the same shape `score_slim` expects.

    OpenAlex's DOI is stored as a URL ("https://doi.org/10.1234/x"); we strip the
    prefix so the field matches Crossref's `DOI` bare form.
    """
    doi_url = work.get("doi") or ""
    doi = doi_url.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    primary = (work.get("primary_location") or {}).get("source") or {}
    return {
        "doi": doi or None,
        "type": work.get("type"),
        "title": work.get("title") or work.get("display_name"),
        "container_title": primary.get("display_name"),
        "issued_year": work.get("publication_year"),
        "volume": (work.get("biblio") or {}).get("volume"),
        "issue": (work.get("biblio") or {}).get("issue"),
        "page": _format_page(work.get("biblio") or {}),
        "publisher": primary.get("host_organization_name"),
        "is_referenced_by_count": work.get("cited_by_count"),
        # OpenAlex does not have Crossref's update-to retraction flag, so this is None.
        # Phase 2 still has the option to cross-check via Crossref using the resolved DOI.
        "update_to": None,
        "openalex_id": work.get("id"),  # extra field; harmless if downstream ignores
    }


def _format_page(biblio: dict[str, Any]) -> str | None:
    first, last = biblio.get("first_page"), biblio.get("last_page")
    if first and last:
        return f"{first}-{last}"
    return first or last


def resolve_by_openalex(raw: RawReference, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Reference:
    """Search OpenAlex by title; score with the same scoring logic Crossref uses."""
    if not raw.title:
        return Reference(
            raw=raw,
            status=ResolutionStatus.UNRESOLVED,
            notes=["resolve_by_openalex: no title on input"],
        )

    params: dict[str, Any] = {
        "search": raw.title,
        "per-page": MAX_CANDIDATES,
        **_polite_params(),
    }

    with _client(timeout_s) as client:
        try:
            resp = _retryable_get(client, "/works", **params)
        except httpx.RequestError as exc:
            log.warning("OpenAlex unreachable for %r: %s", raw.title, exc)
            return Reference(
                raw=raw,
                status=ResolutionStatus.ERROR,
                notes=[f"resolve_by_openalex: network error: {exc}"],
            )

    if resp is None:
        return Reference(
            raw=raw,
            status=ResolutionStatus.UNRESOLVED,
            notes=["resolve_by_openalex: OpenAlex daily budget exhausted; skipped"],
        )

    if resp.status_code != 200:
        return Reference(
            raw=raw,
            status=ResolutionStatus.ERROR,
            notes=[f"resolve_by_openalex: OpenAlex {resp.status_code}"],
        )

    items = (resp.json() or {}).get("results") or []
    if not items:
        return Reference(
            raw=raw,
            status=ResolutionStatus.UNRESOLVED,
            notes=["resolve_by_openalex: no candidates"],
        )

    scored = sorted(
        ((score_slim(raw, _slim_openalex_work(w)), _slim_openalex_work(w)) for w in items),
        key=lambda t: t[0],
        reverse=True,
    )
    top_score, top_slim = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    stored_score = min(1.0, top_score)

    if top_score >= RESOLVED_THRESHOLD and top_slim["doi"]:
        return Reference(
            raw=raw,
            status=ResolutionStatus.RESOLVED,
            resolved_doi=top_slim["doi"],
            resolved_metadata=top_slim,
            resolution_score=stored_score,
            notes=["resolve_by_openalex: matched"],
        )
    if top_score >= AMBIGUOUS_THRESHOLD and top_slim["doi"]:
        return Reference(
            raw=raw,
            status=ResolutionStatus.AMBIGUOUS,
            resolved_doi=top_slim["doi"],
            resolved_metadata=top_slim,
            resolution_score=stored_score,
            notes=[
                f"resolve_by_openalex: top score {top_score:.2f}, "
                f"runner-up {second_score:.2f} — review before relying"
            ],
        )
    # Either below the ambiguous threshold or the top match has no DOI (some
    # OpenAlex records lack one — typically grey literature). Treat as unresolved.
    return Reference(
        raw=raw,
        status=ResolutionStatus.UNRESOLVED,
        resolution_score=stored_score,
        notes=[
            f"resolve_by_openalex: best score {top_score:.2f} below threshold"
            if top_score < AMBIGUOUS_THRESHOLD
            else "resolve_by_openalex: top candidate has no DOI"
        ],
    )
