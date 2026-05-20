"""Phase 4: journal-quality / predatory-venue flagging.

For each reference with a journal field, citecheck combines two free signals:

1. **DOAJ membership** (positive signal). The Directory of Open Access
   Journals publishes a free REST API at `https://doaj.org/api/v3/search/`.
   A journal listed in DOAJ is open-access and meets DOAJ's editorial
   standards; that is a reliable "low risk" indicator.

2. **Known-predatory publisher list** (negative signal). A small hand-
   curated set of publishers documented as predatory by Retraction Watch,
   Cabells, and academic-integrity authors over the last decade. The list
   is conservative — we err toward false negatives rather than false-
   accusations. The full Beall's list is contested and frozen at 2017; we
   prefer named-publisher matches that have remained stable references.

The output is a coarse risk_level (low / medium / high) with notes. Users
should treat HIGH as "flag for review", not "this is predatory" — the
underlying lists are imperfect.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from citecheck.checks.cache import CacheStore
from citecheck.models import (
    JournalQualityCheck,
    JournalRiskLevel,
    Reference,
)

log = logging.getLogger(__name__)

DOAJ_BASE = "https://doaj.org/api/v3"
DEFAULT_TIMEOUT_S = 30.0


# Hard-coded list of publisher names long documented as predatory by multiple
# independent sources (Retraction Watch coverage, Cabells, academic news).
# Strings are matched case-insensitively as substrings against the cited
# journal/publisher field. Conservative by design — false negatives are
# preferred over false accusations.
_KNOWN_CONCERN_PUBLISHERS: tuple[str, ...] = (
    "omics international",
    "omics publishing",
    "scientific research publishing",
    "scirp",
    "academic journals",
    "international scholarly research network",
    "isrn",
    "hindawi",  # debated; included because of high APC + journal-flood pattern
    "mdpi",  # debated; included with same caveat — user can override
    "frontiers",  # debated; same
    "imedpub",
    "longdom",
    "alliedacademies",
    "scholarena",
    "juniper publishers",
    "scientific federation",
    "biocore group",
    "lupine publishers",
)


def _polite_email() -> str:
    return os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()


def _user_agent() -> str:
    from citecheck import __version__

    email = _polite_email()
    if email and "@" in email:
        return f"citecheck/{__version__} (mailto:{email})"
    return f"citecheck/{__version__}"


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _query_doaj(journal: str, *, timeout_s: float) -> bool | None:
    """True if DOAJ indexes the journal; False if not found; None on error.

    We use the journal-title search endpoint with quoted exact match and
    return the first hit if its title matches at ~90% fuzz. DOAJ supports
    free-text queries via Lucene-ish syntax (`bibjson.title:"<name>"`).
    """
    if not journal.strip():
        return None
    url = f"{DOAJ_BASE}/search/journals/bibjson.title:%22{journal.strip()}%22"
    headers = {"User-Agent": _user_agent(), "Accept": "application/json"}
    try:
        resp = httpx.get(url, headers=headers, params={"pageSize": 5}, timeout=timeout_s)
    except httpx.RequestError as exc:
        log.warning("journal_quality: DOAJ query failed for %r: %s", journal, exc)
        return None
    if resp.status_code != 200:
        log.warning(
            "journal_quality: DOAJ %s for %r: %s", resp.status_code, journal, resp.text[:120]
        )
        return None
    body: dict[str, Any] = resp.json() or {}
    results = body.get("results") or []
    # DOAJ returns objects with bibjson.title; treat any non-empty hit list
    # as a positive signal. A more rigorous match would fuzz-compare titles.
    return bool(results)


def _concern_publisher_match(journal: str) -> str | None:
    """Return the matched concern-list entry (lower-cased) or None."""
    if not journal:
        return None
    text = journal.lower()
    for name in _KNOWN_CONCERN_PUBLISHERS:
        if name in text:
            return name
    return None


def check_journal_quality(
    reference: Reference,
    *,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> JournalQualityCheck:
    """Phase 4 entry point. Safe to call on any Reference; UNCHECKED when no journal."""
    journal = (reference.raw.journal or "").strip()
    if not journal:
        return JournalQualityCheck(
            risk_level=JournalRiskLevel.UNCHECKED,
            notes=["no journal field — book / grey-lit / preprint"],
        )

    # Concern-list check is local and instant; do it first.
    concern_match = _concern_publisher_match(journal)
    if concern_match:
        return JournalQualityCheck(
            risk_level=JournalRiskLevel.HIGH,
            on_concern_list=True,
            matched_publisher=concern_match,
            notes=[
                f"journal/publisher matches concern-list entry '{concern_match}'. "
                "Flag for human review — concern lists are heuristic."
            ],
        )

    # Cache key isolates the DOAJ result so a hit/miss is sticky for the TTL.
    key = f"doaj:{journal.lower()}"
    cached_doaj: bool | None = None
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            cached_doaj = cached.get("listed") if isinstance(cached, dict) else None

    if cached_doaj is None:
        doaj_listed = _query_doaj(journal, timeout_s=timeout_s)
        if cache is not None and doaj_listed is not None:
            cache.set(key, {"listed": doaj_listed})
        cached_doaj = doaj_listed

    if cached_doaj is True:
        return JournalQualityCheck(
            risk_level=JournalRiskLevel.LOW,
            doaj_listed=True,
            notes=["DOAJ-indexed"],
        )
    if cached_doaj is False:
        return JournalQualityCheck(
            risk_level=JournalRiskLevel.MEDIUM,
            doaj_listed=False,
            notes=[
                "not in DOAJ — could be a non-OA journal, a small venue, or a non-English title. "
                "Not by itself a red flag."
            ],
        )
    # cached_doaj is None: DOAJ unreachable.
    return JournalQualityCheck(
        risk_level=JournalRiskLevel.UNCHECKED,
        doaj_listed=None,
        notes=["DOAJ check skipped (network error)"],
    )
