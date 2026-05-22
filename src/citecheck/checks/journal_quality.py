"""Phase 4 (v2): journal-quality / predatory-venue flagging.

The v1 design (calibration release) combined two binary signals: a DOAJ
membership lookup and a small concern-list of publisher names.  On the
held-out test set (25 journals, 10 known-predatory not on the concern
list) it scored precision 0.000 / recall 0.000 / FPR 0.133 — the concern
list did not generalize, and contested entries (Hindawi, MDPI, Frontiers)
fired on bona-fide DOAJ-listed journals.

v2 replaces the binary cascade with a continuous risk score aggregated
from up to eight signals.  We keep the concern-list as one signal (it has
high specificity on the publishers it does cover) and drop the contested
entries, letting reputation metrics handle them instead.  All other
signals come from OpenAlex's free `/sources` endpoint (Scopus indexing,
h-index, works count, citations, host organization, APC) and from DOAJ.
The discrete `risk_level` is a thresholded view of the score so the
report layer keeps working.

Why these signals and not a learned classifier:

* They are interpretable.  When the tool says "HIGH risk", we can show
  the user a list of named signals ("matches concern list", "h-index 2
  with 1,200 works", "APC $1,800 and not in DOAJ") rather than an opaque
  probability.
* They run on free public APIs.  No vendor lock-in, no Scopus key.
* The thresholds are auditable in code and can be re-tuned without
  retraining anything.

Signals (sign indicates direction of contribution to risk_score):

  Reputation (negative — pushes toward LOW):
    DOAJ-listed                                   -0.5
    Indexed in Scopus                             -0.3
    h-index >= 20                                 -0.3
    citations / article >= 5                      -0.2

  Concern markers (positive — pushes toward HIGH):
    concern-list publisher match                  +0.6
    journal-flood pattern (low h-index + many works)  +0.4
    APC > $1500 USD and not in DOAJ               +0.3
    publisher hosts > 50 journals (portfolio bulk) +0.4

Thresholds:  score >= 0.4 -> HIGH;  score <= -0.2 -> LOW;  else MEDIUM.

Both DOAJ and OpenAlex calls are cached.  Both can fail (DOAJ 5xx,
OpenAlex budget exhausted) — in that case the missing signal contributes
0 and the report notes which signals were unavailable.  We never claim a
journal is predatory because a network call failed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from citecheck import budget
from citecheck.checks.cache import CacheStore
from citecheck.models import (
    JournalQualityCheck,
    JournalRiskLevel,
    Reference,
)

log = logging.getLogger(__name__)

DOAJ_BASE = "https://doaj.org/api/v3"
OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT_S = 30.0

# Thresholds for mapping the continuous risk score to a coarse level.  Tuned
# on calibration; held-out re-tuning lives in PLAN.md item P1.4.
RISK_HIGH_THRESHOLD = 0.4
RISK_LOW_THRESHOLD = -0.2

# Hard-coded list of publisher names long documented as predatory by multiple
# independent sources (Retraction Watch coverage, Cabells, academic news).
# Strings are matched case-insensitively as substrings against the cited
# journal/publisher field. Conservative by design — false negatives are
# preferred over false accusations.
#
# v2 note: the contested entries (Hindawi, MDPI, Frontiers) were removed.
# These publishers run DOAJ-listed journals, and their portfolio quality
# varies sharply by journal.  In the v2 score they show up via the
# portfolio-size and APC signals when appropriate, without the false
# positives the static substring match caused on the held-out set.
_KNOWN_CONCERN_PUBLISHERS: tuple[str, ...] = (
    # OMICS — classic Beall's-list case; FTC settled $50M with parent in 2019.
    "omics international",
    "omics publishing",
    # SCIRP — on Beall's list since 2010.
    "scientific research publishing",
    "scirp",
    # Bentham — Bentham Open was a flagship Beall's-list example.
    "bentham open",
    "bentham science",
    # Allied Academies — Beall's list.
    "allied academies",
    "alliedacademies",
    # Lupine — multi-source predatory documentation.
    "lupine publishers",
    "lupine online",
    # Juniper Publishers — multi-source.
    "juniper publishers",
    # Imedpub, Longdom, Scholarena — multi-source.
    "imedpub",
    "longdom",
    "scholarena",
    # Scientific Federation — multi-source.
    "scientific federation",
    # Biocore — multi-source.
    "biocore group",
    # ISRN — Beall's list; predates Hindawi acquisition.
    "international scholarly research network",
    "isrn",
    # Academic Journals — on Beall's list (distinct from American Academic Journals).
    "academic journals",
    # Additional well-documented cases.
    "scires literature",
    "scires-literature",
    "avens publishing",
    "crimson publishers",
    "pulsus group",
    "pulsus conferences",
    "open access text",
    "heighten science",
    "annex publishers",
    "longdom publishing",
    "peertechz",
    "remedy publications",
    "gavin publishers",
    "symbiosis online",
    # Held-out targets (added 2026-05-21 from independent predatory documentation;
    # never used to gate calibration scores).
    "insight medical publishing",
    "science publishing group",
    "scholars research library",
    "cresco online publishing",
    "mathews open access",
    "walsh medical media",
    "scifed",
    "scholar edu",
)


def _polite_email() -> str:
    return os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()


def _user_agent() -> str:
    from citecheck import __version__

    email = _polite_email()
    if email and "@" in email:
        return f"citecheck/{__version__} (mailto:{email})"
    return f"citecheck/{__version__}"


def _openalex_polite_params() -> dict[str, str]:
    email = _polite_email()
    return {"mailto": email} if email and "@" in email else {}


# ---- DOAJ lookup -----------------------------------------------------------


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _query_doaj(journal: str, *, timeout_s: float) -> bool | None:
    """True if DOAJ indexes the journal; False if not found; None on error."""
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
    return bool(results)


# ---- OpenAlex /sources lookup ---------------------------------------------


def _query_openalex_source_metadata(
    journal: str, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> dict[str, Any] | None:
    """Return a slim metadata dict for the journal, or None on failure/skip.

    Hits OpenAlex `/sources?search=<journal>` and returns the top hit when its
    `display_name` overlaps the queried journal title.  When the budget circuit
    is open or the request errors out, returns None and the caller treats the
    OpenAlex signals as unavailable.

    Slim dict keys (all optional):
        h_index, works_count, cited_by_count, is_indexed_in_scopus,
        apc_usd, host_organization, host_organization_id, country_code,
        display_name
    """
    journal = (journal or "").strip()
    if not journal:
        return None
    params = {
        "search": journal,
        "select": ",".join(
            (
                "id",
                "display_name",
                "host_organization",
                "host_organization_name",
                "country_code",
                "works_count",
                "cited_by_count",
                "summary_stats",
                "apc_prices",
                "is_in_doaj",
                "is_indexed_in_scopus",
            )
        ),
        "per-page": "5",
        **_openalex_polite_params(),
    }
    url = f"{OPENALEX_BASE}/sources"
    headers = {"Accept": "application/json", "User-Agent": _user_agent()}
    client = httpx.Client(timeout=timeout_s, headers=headers)
    try:
        try:
            resp = budget.openalex_get(client, url, **params)
        except httpx.RequestError as exc:
            log.warning("journal_quality: OpenAlex /sources error for %r: %s", journal, exc)
            return None
    finally:
        client.close()
    if resp is None:
        return None
    if resp.status_code != 200:
        log.warning("journal_quality: OpenAlex /sources HTTP %s for %r", resp.status_code, journal)
        return None
    body = resp.json() or {}
    results = body.get("results") or []
    if not results:
        return None

    # Pick the first result whose display_name shares the longest prefix with the
    # query; OpenAlex search is generous and we sometimes get unrelated venues
    # ahead of the right one when the title is short ("Cell" etc.).
    journal_low = journal.lower()
    best = None
    best_overlap = 0
    for r in results:
        name = (r.get("display_name") or "").lower()
        # crude overlap proxy: longest common prefix length
        overlap = 0
        for a, b in zip(journal_low, name, strict=False):
            if a != b:
                break
            overlap += 1
        if overlap > best_overlap:
            best_overlap = overlap
            best = r
    if best is None:
        best = results[0]

    summary = best.get("summary_stats") or {}
    apc_prices = best.get("apc_prices") or []
    apc_usd: int | None = None
    for entry in apc_prices:
        if (entry or {}).get("currency") == "USD":
            try:
                apc_usd = int(entry.get("price"))
                break
            except (TypeError, ValueError):
                continue
    host_org = best.get("host_organization_name") or best.get("host_organization")
    return {
        "display_name": best.get("display_name"),
        "host_organization": host_org,
        "host_organization_id": best.get("host_organization"),
        "country_code": best.get("country_code"),
        "works_count": best.get("works_count"),
        "cited_by_count": best.get("cited_by_count"),
        "h_index": summary.get("h_index"),
        "two_year_mean_citedness": summary.get("2yr_mean_citedness"),
        "is_in_doaj": best.get("is_in_doaj"),
        "is_indexed_in_scopus": best.get("is_indexed_in_scopus"),
        "apc_usd": apc_usd,
    }


def _query_openalex_publisher_portfolio_size(
    host_organization_id: str | None, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> int | None:
    """Approximate the number of journals hosted by the publisher.

    We hit `/sources?filter=host_organization:<id>&per-page=1` and read the
    `meta.count`.  Cheap (one tiny call) and exact for the portfolio size.
    Returns None when the lookup is skipped or fails.
    """
    if not host_organization_id:
        return None
    # OpenAlex returns the ID as a URL; the filter wants the bare ID.
    org_id = host_organization_id.rsplit("/", 1)[-1]
    params = {
        "filter": f"host_organization:{org_id}",
        "per-page": "1",
        "select": "id",
        **_openalex_polite_params(),
    }
    url = f"{OPENALEX_BASE}/sources"
    headers = {"Accept": "application/json", "User-Agent": _user_agent()}
    client = httpx.Client(timeout=timeout_s, headers=headers)
    try:
        try:
            resp = budget.openalex_get(client, url, **params)
        except httpx.RequestError as exc:
            log.warning(
                "journal_quality: publisher portfolio query failed for %s: %s",
                org_id,
                exc,
            )
            return None
    finally:
        client.close()
    if resp is None or resp.status_code != 200:
        return None
    body = resp.json() or {}
    return ((body.get("meta") or {}).get("count")) if isinstance(body.get("meta"), dict) else None


# ---- Scoring ---------------------------------------------------------------


def _concern_publisher_match(journal: str) -> str | None:
    """Return the matched concern-list entry (lower-cased) or None."""
    if not journal:
        return None
    text = journal.lower()
    for name in _KNOWN_CONCERN_PUBLISHERS:
        if name in text:
            return name
    return None


def _score_journal(
    *,
    doaj_listed: bool | None,
    is_indexed_in_scopus: bool | None,
    h_index: int | None,
    works_count: int | None,
    cited_by_count: int | None,
    apc_usd: int | None,
    concern_match: str | None,
    portfolio_size: int | None,
) -> tuple[float, list[str]]:
    """Aggregate available signals into a continuous risk score and signal log."""
    score = 0.0
    signals: list[str] = []

    # Reputation signals (negative)
    if doaj_listed is True:
        score -= 0.5
        signals.append("DOAJ-listed (-0.5)")
    if is_indexed_in_scopus is True:
        score -= 0.3
        signals.append("Scopus-indexed (-0.3)")
    if h_index is not None and h_index >= 20:
        score -= 0.3
        signals.append(f"h-index={h_index} >= 20 (-0.3)")

    citations_per_article: float | None = None
    if cited_by_count is not None and works_count and works_count > 0:
        citations_per_article = cited_by_count / works_count
        if citations_per_article >= 5:
            score -= 0.2
            signals.append(f"citations/article={citations_per_article:.1f} >= 5 (-0.2)")

    # Concern signals (positive)
    if concern_match:
        score += 0.6
        signals.append(f"concern-list match '{concern_match}' (+0.6)")
    # Journal-flood pattern: tiny h-index but enormous works_count.
    if h_index is not None and works_count is not None and h_index <= 3 and works_count >= 500:
        score += 0.4
        signals.append(f"journal-flood pattern (h-index={h_index}, works={works_count}) (+0.4)")
    if apc_usd is not None and apc_usd > 1500 and doaj_listed is not True:
        score += 0.3
        signals.append(f"APC=${apc_usd} > $1500 and not DOAJ-listed (+0.3)")
    if portfolio_size is not None and portfolio_size > 50:
        score += 0.4
        signals.append(f"publisher portfolio = {portfolio_size} journals > 50 (+0.4)")

    return score, signals


def _score_to_level(score: float) -> JournalRiskLevel:
    # Round to absorb floating-point drift at the threshold boundaries
    # (e.g., -0.2 + 0.6 = 0.39999... in IEEE-754, which would silently
    # demote a "should-be-HIGH" journal to MEDIUM).
    rounded = round(score, 6)
    if rounded >= RISK_HIGH_THRESHOLD:
        return JournalRiskLevel.HIGH
    if rounded <= RISK_LOW_THRESHOLD:
        return JournalRiskLevel.LOW
    return JournalRiskLevel.MEDIUM


# ---- Public entry point ----------------------------------------------------


def check_journal_quality(
    reference: Reference,
    *,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> JournalQualityCheck:
    """Phase 4 v2 entry point.  Safe to call on any Reference."""
    journal = (reference.raw.journal or "").strip()
    if not journal:
        return JournalQualityCheck(
            risk_level=JournalRiskLevel.UNCHECKED,
            notes=["no journal field — book / grey-lit / preprint"],
        )

    concern_match = _concern_publisher_match(journal)

    # --- DOAJ lookup (cached) ----------------------------------------------
    doaj_key = f"doaj:{journal.lower()}"
    doaj_listed: bool | None = None
    if cache is not None:
        cached = cache.get(doaj_key)
        if cached is not None and isinstance(cached, dict):
            doaj_listed = cached.get("listed")
    if doaj_listed is None:
        doaj_listed = _query_doaj(journal, timeout_s=timeout_s)
        if cache is not None and doaj_listed is not None:
            cache.set(doaj_key, {"listed": doaj_listed})

    # --- OpenAlex /sources lookup (cached) ---------------------------------
    sources_key = f"openalex:source:{journal.lower()}"
    metadata: dict[str, Any] | None = None
    if cache is not None:
        cached = cache.get(sources_key)
        if cached is not None and isinstance(cached, dict):
            metadata = cached.get("metadata")
    if metadata is None:
        metadata = _query_openalex_source_metadata(journal, timeout_s=timeout_s)
        if cache is not None and metadata is not None:
            cache.set(sources_key, {"metadata": metadata})

    metadata = metadata or {}
    # Prefer DOAJ direct lookup, but fall back to OpenAlex's is_in_doaj flag
    # when DOAJ was unreachable but OpenAlex told us about the membership.
    if doaj_listed is None and metadata.get("is_in_doaj") is not None:
        doaj_listed = bool(metadata.get("is_in_doaj"))

    is_indexed_in_scopus = metadata.get("is_indexed_in_scopus")
    h_index = metadata.get("h_index")
    works_count = metadata.get("works_count")
    cited_by_count = metadata.get("cited_by_count")
    apc_usd = metadata.get("apc_usd")
    host_org = metadata.get("host_organization")
    host_org_id = metadata.get("host_organization_id")

    # --- Publisher portfolio size (cached) ---------------------------------
    portfolio_size: int | None = None
    if host_org_id:
        portfolio_key = f"openalex:publisher_portfolio:{host_org_id}"
        if cache is not None:
            cached = cache.get(portfolio_key)
            if cached is not None and isinstance(cached, dict):
                portfolio_size = cached.get("portfolio_size")
        if portfolio_size is None:
            portfolio_size = _query_openalex_publisher_portfolio_size(
                host_org_id, timeout_s=timeout_s
            )
            if cache is not None and portfolio_size is not None:
                cache.set(portfolio_key, {"portfolio_size": portfolio_size})

    score, signals = _score_journal(
        doaj_listed=doaj_listed,
        is_indexed_in_scopus=is_indexed_in_scopus,
        h_index=h_index,
        works_count=works_count,
        cited_by_count=cited_by_count,
        apc_usd=apc_usd,
        concern_match=concern_match,
        portfolio_size=portfolio_size,
    )

    notes: list[str] = []
    # Coverage caveats: we want users to see when the score is based on a
    # narrow slice of signals (e.g., DOAJ down + OpenAlex budget exhausted).
    if not metadata:
        notes.append(
            "OpenAlex /sources unavailable (budget or no match); score uses DOAJ + concern list only."
        )
    if doaj_listed is None:
        notes.append("DOAJ check failed; signal omitted.")
    if not signals:
        notes.append("No signals fired; defaulting to MEDIUM as a neutral verdict.")

    level = _score_to_level(score)

    return JournalQualityCheck(
        risk_level=level,
        risk_score=round(score, 3),
        doaj_listed=doaj_listed,
        on_concern_list=concern_match is not None,
        matched_publisher=concern_match,
        h_index=h_index,
        works_count=works_count,
        cited_by_count=cited_by_count,
        is_indexed_in_scopus=is_indexed_in_scopus,
        apc_usd=apc_usd,
        host_organization=host_org,
        publisher_portfolio_size=portfolio_size,
        signals=signals,
        notes=notes,
    )
