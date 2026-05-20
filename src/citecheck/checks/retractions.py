"""Retraction check.

For each resolved reference, decide whether the cited work has been retracted,
withdrawn, flagged with an expression of concern, or corrected.

Two complementary sources:

1. **Crossref `update-to`** — when the publisher deposits a retraction notice
   into Crossref, the retracted paper's record lists it under `update-to`. We
   already capture this in the slim metadata during Phase 1, so Path A is free.

2. **OpenAlex `is_retracted`** — OpenAlex integrates the Retraction Watch
   dataset (Crossref + RW partnership, public since 2023). When Crossref's
   `update-to` is empty (which happens for older retractions whose publishers
   never backfilled — Wakefield's 1998 MMR paper is a real example), OpenAlex
   still flags `is_retracted: true`. Path B catches that gap.

Both paths are single HTTP calls and idempotent. Results are memoized in the
SQLite cache with the project-wide 7-day TTL.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from citecheck.checks.cache import CacheStore
from citecheck.models import Reference, RetractionCheck, RetractionStatus

log = logging.getLogger(__name__)

CROSSREF_BASE = "https://api.crossref.org"
OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT_S = 30.0


# Crossref normalizes update types as lowercase tokens, but the exact spelling
# has drifted (e.g. "expression-of-concern" vs "expression_of_concern"). Match
# loosely on substring after normalizing separators.
_RETRACTED_TOKENS = {"retraction", "retracted", "withdrawal", "withdrawn", "removal"}
_EOC_TOKENS = {"expression-of-concern", "expression_of_concern", "expressionofconcern", "concern"}
_CORRECTION_TOKENS = {"correction", "corrigendum", "erratum", "addendum"}


def _user_agent() -> str:
    from citecheck import __version__

    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    if email and "@" in email:
        return f"citecheck/{__version__} (mailto:{email}; +https://github.com/pierrebeaucoral/citecheck)"
    return f"citecheck/{__version__} (+https://github.com/pierrebeaucoral/citecheck)"


def _normalize_token(value: str | None) -> str:
    if not value:
        return ""
    return value.lower().replace("_", "-").strip()


def _classify_token(token: str) -> RetractionStatus | None:
    """Map an update-type token to a status; None for tokens we don't recognize."""
    norm = _normalize_token(token).replace("-", "")
    if any(t.replace("-", "") in norm for t in _RETRACTED_TOKENS):
        return RetractionStatus.RETRACTED
    if any(t.replace("-", "") == norm or t.replace("-", "") in norm for t in _EOC_TOKENS):
        return RetractionStatus.EXPRESSION_OF_CONCERN
    if any(t in norm for t in _CORRECTION_TOKENS):
        return RetractionStatus.CORRECTION
    return None


_STATUS_RANK = {
    RetractionStatus.RETRACTED: 3,
    RetractionStatus.EXPRESSION_OF_CONCERN: 2,
    RetractionStatus.CORRECTION: 1,
    RetractionStatus.CLEAN: 0,
}


def _worst(a: RetractionStatus, b: RetractionStatus) -> RetractionStatus:
    """Return the more severe of two statuses (retracted > EOC > correction > clean)."""
    return a if _STATUS_RANK.get(a, 0) >= _STATUS_RANK.get(b, 0) else b


def _interpret_update_to(update_to: list[dict[str, Any]] | None) -> RetractionCheck:
    """Convert a Crossref `update-to` list into a RetractionCheck.

    update-to entries look like: {"DOI": "10.x/y", "type": "retraction", "label": "Retraction", "updated": {"date-parts": [[2010, 2, 6]]}}
    """
    if not update_to:
        return RetractionCheck(status=RetractionStatus.CLEAN)

    worst = RetractionStatus.CLEAN
    notice_doi: str | None = None
    notice_date: str | None = None
    label: str | None = None

    for entry in update_to:
        if not isinstance(entry, dict):
            continue
        status = _classify_token(entry.get("type"))
        if status is None:
            continue
        if _STATUS_RANK[status] > _STATUS_RANK[worst]:
            worst = status
            notice_doi = entry.get("DOI") or notice_doi
            label = entry.get("label") or label
            updated = entry.get("updated") or {}
            parts = updated.get("date-parts") or []
            if parts and parts[0]:
                # date-parts is e.g. [[2010, 2, 6]] -> "2010-02-06"
                notice_date = "-".join(
                    f"{int(p):02d}" if i > 0 else f"{int(p)}" for i, p in enumerate(parts[0])
                )

    if worst == RetractionStatus.CLEAN:
        return RetractionCheck(status=RetractionStatus.CLEAN)

    return RetractionCheck(
        status=worst,
        notice_doi=notice_doi,
        notice_date=notice_date,
        reason=label,
        source_url=f"https://doi.org/{notice_doi}" if notice_doi else None,
        notes=["check_retraction: matched on Crossref update-to"],
    )


def _openalex_polite_params() -> dict[str, str]:
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    return {"mailto": email} if email and "@" in email else {}


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _query_openalex_retraction(
    doi: str, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> RetractionCheck:
    """Ask OpenAlex whether the DOI is flagged as retracted.

    OpenAlex's `is_retracted` boolean is fed by the Retraction Watch dataset
    (Crossref + RW partnership). When Crossref's own update-to is empty, this
    is usually where retraction signals live.

    OpenAlex does not expose the retraction notice DOI or date in the public
    /works endpoint, so we return RETRACTED with minimal provenance — the user
    can follow the OpenAlex work URL if they want full notice metadata.
    """
    params = {"select": "id,doi,is_retracted,title,display_name", **_openalex_polite_params()}
    headers = {"Accept": "application/json"}
    url = f"{OPENALEX_BASE}/works/doi:{doi}"
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=timeout_s)
    except httpx.RequestError as exc:
        log.warning("retraction: OpenAlex query failed for %s: %s", doi, exc)
        return RetractionCheck(
            status=RetractionStatus.ERROR,
            notes=[f"openalex query: network error: {exc}"],
        )

    if resp.status_code == 404:
        # OpenAlex does not know this DOI — cannot conclude retracted from absence.
        return RetractionCheck(
            status=RetractionStatus.CLEAN,
            notes=["openalex: DOI not indexed; no retraction signal available"],
        )
    if resp.status_code != 200:
        return RetractionCheck(
            status=RetractionStatus.ERROR,
            notes=[f"openalex query: HTTP {resp.status_code}"],
        )

    work = resp.json() or {}
    if not work.get("is_retracted"):
        return RetractionCheck(status=RetractionStatus.CLEAN)

    openalex_id = (work.get("id") or "").removeprefix("https://openalex.org/")
    return RetractionCheck(
        status=RetractionStatus.RETRACTED,
        reason=work.get("display_name") or work.get("title"),
        source_url=work.get("id") or None,
        notes=[f"check_retraction: OpenAlex is_retracted=true (work id {openalex_id})"],
    )


def _merge_checks(a: RetractionCheck, b: RetractionCheck) -> RetractionCheck:
    """Combine two RetractionChecks into one richer result.

    Policy:
    - If exactly one source errored, keep the other's verdict but PRESERVE the
      error in notes so the user sees that one source failed.
    - If both errored, the merged status is ERROR.
    - Otherwise, take the more severe status (retracted > EOC > correction > clean).
    - For provenance fields (notice DOI, date, reason, source URL), prefer
      Crossref (`a`) when present because its update-to is structured;
      fall back to OpenAlex (`b`).
    """
    if a.status == RetractionStatus.ERROR and b.status == RetractionStatus.ERROR:
        return RetractionCheck(
            status=RetractionStatus.ERROR,
            notes=[*a.notes, *b.notes],
        )

    # Pick the verdict-bearing side; merge notes from BOTH so partial failures
    # remain visible.
    if a.status == RetractionStatus.ERROR:
        verdict = b
    elif b.status == RetractionStatus.ERROR:
        verdict = a
    else:
        verdict = a if _STATUS_RANK.get(a.status, 0) >= _STATUS_RANK.get(b.status, 0) else b

    return RetractionCheck(
        status=verdict.status,
        notice_doi=a.notice_doi or b.notice_doi,
        notice_date=a.notice_date or b.notice_date,
        reason=a.reason or b.reason,
        source_url=a.source_url or b.source_url,
        notes=[*a.notes, *b.notes],
    )


def check_retraction(
    reference: Reference,
    *,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> RetractionCheck:
    """Phase 2 check. Safe to call on any Reference; UNCHECKED when no DOI.

    Runs BOTH the Crossref `update-to` inspection and the OpenAlex `is_retracted`
    query, then merges them. Running both is intentional: Crossref carries the
    structured notice (DOI + date) when populated, and OpenAlex catches the
    older retractions where Crossref's update-to is empty. The two together
    give the most complete answer.
    """
    doi = (reference.resolved_doi or "").lower().strip()
    if not doi:
        return RetractionCheck(
            status=RetractionStatus.UNCHECKED,
            notes=["check_retraction: no resolved DOI"],
        )

    key = f"retraction:{doi}"
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return RetractionCheck(**cached)

    # Path A — read what the resolver already cached on the Reference (free).
    slim = reference.resolved_metadata or {}
    crossref_check = _interpret_update_to(slim.get("update_to"))

    # Path B — ask OpenAlex; integrates Retraction Watch.
    openalex_check = _query_openalex_retraction(doi, timeout_s=timeout_s)

    merged = _merge_checks(crossref_check, openalex_check)
    if cache is not None:
        cache.set(key, merged.model_dump(mode="json"))
    return merged
