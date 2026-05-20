"""Five-layer hallucination detector.

For each Reference, runs five cheap-ish checks and aggregates the count of
flagged signals into a verdict. The rules are intentionally rule-based (no
LLM); a paragraph of reasoning is templated from which layers fired.

Layers
------
L1 — **DOI integrity**: if the citation supplies a DOI, compare the
     resolved metadata (title / first-author family / year) against what the
     citation claims. Mismatches imply the citation pasted a real DOI under
     a fabricated description.

L2 — **Cross-database existence**: read `Reference.status` from Phase 1's
     dispatch. UNRESOLVED across Crossref + OpenAlex = strong red flag.
     AMBIGUOUS is informational, not a flag.

L3 — **Author plausibility**: query OpenAlex `/authors` for the first two
     listed authors. If NEITHER appears, flag. Single-author misses on
     non-English / obscure researchers are common, so we require both to
     miss before flagging.

L4 — **Venue plausibility**: query OpenAlex `/sources` for the journal.
     If the journal is unknown to OpenAlex, flag. (Books and working papers
     legitimately lack a journal, so L4 is skipped in that case.)

L5 — **Aggregation**: rule-based per PLAN.md:
     - 4+ red flags                       -> likely_hallucinated
     - 2-3 red flags                      -> suspicious
     - 0-1 red flags, resolved DOI        -> real_high_confidence
     - 0-1 red flags, no DOI              -> real_low_confidence

Low-coverage caveat
-------------------
References to pre-2000 papers, books (no journal), or non-English venues
get caveat notes attached. The caveat downgrades a *real_high_confidence*
verdict to *real_low_confidence* (one rank), but DOES NOT push real_low
further into "suspicious". Without this asymmetric policy, every obscure
or pre-2000 paper would be flagged as suspicious — confirmed empirically
against the Callaway/Sant'Anna fixture, where the Hájek/Basu 1971 paper
and the Neumark/Wascher MIT Press book were both wrongly elevated to
suspicious by the original (symmetric) policy.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from rapidfuzz import fuzz
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from citecheck import budget
from citecheck.checks.cache import CacheStore
from citecheck.models import (
    Author,
    HallucinationCheck,
    HallucinationVerdict,
    LayerSignal,
    Reference,
    ResolutionStatus,
)

log = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org"
DEFAULT_TIMEOUT_S = 30.0

# Thresholds. Calibration is heuristic; the Phase 3 eval set (data/eval/) will
# tune these against the held-out 20-row split.
TITLE_MATCH_THRESHOLD = 75  # rapidfuzz 0-100 scale
AUTHOR_FAMILY_THRESHOLD = 85
YEAR_TOLERANCE = 1


def _polite_params() -> dict[str, str]:
    email = os.environ.get("CITECHECK_CONTACT_EMAIL", "").strip()
    return {"mailto": email} if email and "@" in email else {}


def _client(timeout_s: float) -> httpx.Client:
    return httpx.Client(
        base_url=OPENALEX_BASE,
        timeout=timeout_s,
        headers={"Accept": "application/json"},
    )


@retry(
    retry=retry_if_exception_type(httpx.RequestError),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _get(client: httpx.Client, url: str, **params: Any) -> httpx.Response | None:
    """Wrap client.get with budget circuit + metrics; None when budget exhausted."""
    return budget.openalex_get(client, url, **params)


# ----- Layer 1: DOI integrity -------------------------------------------------


def _layer1_doi_integrity(reference: Reference) -> LayerSignal:
    """Compare claimed vs resolved metadata when a DOI is present.

    A real DOI that resolves to a paper with a completely different title is
    the classic LLM-hallucination signature.
    """
    raw = reference.raw
    if not raw.doi or reference.status != ResolutionStatus.RESOLVED:
        return LayerSignal(
            layer="L1_doi_integrity",
            flagged=False,
            reasoning="No claimed DOI to check (or resolution did not produce metadata).",
        )

    resolved = reference.resolved_metadata or {}
    title_claimed = raw.title or ""
    title_resolved = resolved.get("title") or ""
    title_score = (
        fuzz.token_set_ratio(title_claimed.lower(), title_resolved.lower())
        if title_claimed and title_resolved
        else 0
    )

    title_mismatch = title_score < TITLE_MATCH_THRESHOLD

    year_mismatch = False
    if raw.year and resolved.get("issued_year"):
        year_mismatch = abs(raw.year - int(resolved["issued_year"])) > YEAR_TOLERANCE

    flagged = title_mismatch or year_mismatch
    if not flagged:
        return LayerSignal(
            layer="L1_doi_integrity",
            flagged=False,
            score=title_score / 100.0,
            reasoning=f"DOI resolves and claimed metadata matches (title sim {title_score}/100).",
        )
    return LayerSignal(
        layer="L1_doi_integrity",
        flagged=True,
        score=title_score / 100.0,
        reasoning=(
            f"DOI resolves but metadata diverges: "
            f"title sim={title_score}/100"
            + (
                f", year mismatch ({raw.year} vs {resolved.get('issued_year')})"
                if year_mismatch
                else ""
            )
        ),
    )


# ----- Layer 2: cross-database existence -------------------------------------


def _layer2_cross_db(reference: Reference) -> LayerSignal:
    """Use Phase 1's resolution outcome as a database-existence signal.

    By the time we reach Phase 3, the resolve() chain has already tried
    Crossref by DOI, Crossref by metadata, and OpenAlex by title. UNRESOLVED
    means we struck out in two independent indexes.
    """
    if reference.status == ResolutionStatus.RESOLVED:
        return LayerSignal(
            layer="L2_cross_db",
            flagged=False,
            reasoning="Resolved cleanly via Crossref or OpenAlex.",
        )
    if reference.status == ResolutionStatus.AMBIGUOUS:
        return LayerSignal(
            layer="L2_cross_db",
            flagged=False,
            reasoning="Resolution is ambiguous but at least one DB has a near-match.",
        )
    if reference.status == ResolutionStatus.UNRESOLVED:
        return LayerSignal(
            layer="L2_cross_db",
            flagged=True,
            reasoning="No close match in Crossref or OpenAlex.",
        )
    return LayerSignal(
        layer="L2_cross_db",
        flagged=False,
        reasoning=f"Resolution returned {reference.status.value}; treating as inconclusive.",
    )


# ----- Layer 3: author plausibility ------------------------------------------


def _openalex_author_lookup(
    family: str, given: str | None, *, client: httpx.Client, cache: CacheStore | None
) -> dict | None:
    """Return the best-matching OpenAlex author record, or None if not found.

    Replaces the old `_openalex_author_exists` so layer 5 can re-use the resolved
    OpenAlex ID to check whether that specific author has a work matching the
    citation's claimed title.
    """
    family = (family or "").strip()
    if not family:
        return None
    query = f"{given} {family}".strip() if given else family
    cache_key = f"openalex:author_lookup:{query.lower()}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            # We cache `False` for known-misses (JSON-serialisable, distinct from None
            # which would re-trigger the lookup).
            return cached if cached else None

    try:
        resp = _get(client, "/authors", search=query, **_polite_params())
    except httpx.RequestError as exc:
        log.warning("hallucination L3: author lookup failed for %r: %s", query, exc)
        return None  # network error: leave the layer to give benefit of doubt
    if resp is None or resp.status_code != 200:
        # Budget exhausted (None) or upstream error: give benefit of doubt.
        return None

    items = (resp.json() or {}).get("results") or []
    best: dict | None = None
    for item in items[:5]:
        candidate_family = (item.get("display_name") or "").split()[-1]
        if fuzz.ratio(family.lower(), candidate_family.lower()) >= AUTHOR_FAMILY_THRESHOLD:
            best = {"id": item.get("id"), "display_name": item.get("display_name")}
            break
    if cache is not None:
        cache.set(cache_key, best if best else False)
    return best


def _layer3_authors(
    reference: Reference, *, client: httpx.Client, cache: CacheStore | None
) -> tuple[LayerSignal, list[tuple[Author, dict]]]:
    """Flag references whose first two authors are both absent from OpenAlex.

    Returns (signal, found_authors). `found_authors` is consumed by L5 to query
    each found author's works without re-doing the /authors lookup.
    """
    authors = reference.raw.authors[:2]
    if not authors:
        return LayerSignal(
            layer="L3_authors",
            flagged=False,
            reasoning="No author names extracted; cannot check.",
        ), []

    misses: list[str] = []
    found: list[tuple[Author, dict]] = []
    for author in authors:
        record = _openalex_author_lookup(author.family, author.given, client=client, cache=cache)
        if record is None:
            misses.append(author.display())
        else:
            found.append((author, record))

    if not misses:
        signal = LayerSignal(
            layer="L3_authors",
            flagged=False,
            reasoning=f"All {len(authors)} checked authors found in OpenAlex.",
        )
    elif len(misses) == len(authors):
        # Require BOTH to miss before flagging. Sparse coverage on a single author
        # is too common to treat as a red flag on its own.
        signal = LayerSignal(
            layer="L3_authors",
            flagged=True,
            reasoning=(
                f"None of the first {len(authors)} authors found in OpenAlex: " + ", ".join(misses)
            ),
        )
    else:
        signal = LayerSignal(
            layer="L3_authors",
            flagged=False,
            reasoning=f"Partial match — {len(authors) - len(misses)}/{len(authors)} authors found.",
        )
    return signal, found


# ----- Layer 4: venue plausibility -------------------------------------------


def _openalex_source_exists(
    journal: str, *, client: httpx.Client, cache: CacheStore | None
) -> bool:
    journal = (journal or "").strip()
    if not journal:
        return False
    cache_key = f"openalex:sources:{journal.lower()}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return bool(cached)

    try:
        resp = _get(client, "/sources", search=journal, **_polite_params())
    except httpx.RequestError as exc:
        log.warning("hallucination L4: venue lookup failed for %r: %s", journal, exc)
        return True
    if resp is None or resp.status_code != 200:
        # Budget exhausted or upstream error: benefit of the doubt (don't flag).
        return True

    items = (resp.json() or {}).get("results") or []
    exists = False
    for item in items[:5]:
        name = item.get("display_name") or ""
        if fuzz.token_set_ratio(journal.lower(), name.lower()) >= 80:
            exists = True
            break
    if cache is not None:
        cache.set(cache_key, exists)
    return exists


def _layer4_venue(
    reference: Reference, *, client: httpx.Client, cache: CacheStore | None
) -> LayerSignal:
    """Flag references whose claimed journal is unknown to OpenAlex.

    Skip when journal is empty — many references (books, working papers) have
    no journal field; the absence is not itself suspicious.
    """
    journal = reference.raw.journal
    if not journal:
        return LayerSignal(
            layer="L4_venue",
            flagged=False,
            reasoning="No journal field on reference; layer not applicable.",
        )
    if _openalex_source_exists(journal, client=client, cache=cache):
        return LayerSignal(
            layer="L4_venue",
            flagged=False,
            reasoning=f"Venue '{journal}' found in OpenAlex.",
        )
    return LayerSignal(
        layer="L4_venue",
        flagged=True,
        reasoning=f"Venue '{journal}' not found in OpenAlex.",
    )


# ----- Layer 5: author-title coherence ---------------------------------------


def _author_published_title(
    author_id: str | None,
    claimed_title: str,
    *,
    client: httpx.Client,
    cache: CacheStore | None,
) -> bool:
    """True if OpenAlex shows the author with `author_id` published a work whose
    title fuzzy-matches `claimed_title`.

    ChatGPT's modal failure on articles is pairing a real author with an
    invented title. This check directly attacks that pattern. We query the
    /works endpoint filtered to the author's OpenAlex ID, with a free-text
    search on the claimed title, then fuzzy-match the top results.

    Returns True (no flag) on network/HTTP failure: give benefit of the doubt.
    """
    if not author_id or not claimed_title:
        return False
    short_id = (author_id or "").removeprefix("https://openalex.org/")
    cache_key = f"openalex:author_title:{short_id}:{claimed_title.lower()[:80]}"
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return bool(cached)

    try:
        resp = _get(
            client,
            "/works",
            filter=f"authorships.author.id:{short_id}",
            search=claimed_title,
            **{"per-page": "5", **_polite_params()},
        )
    except httpx.RequestError as exc:
        log.warning("hallucination L5: author-title lookup failed: %s", exc)
        return True  # network error → don't flag
    if resp is None or resp.status_code != 200:
        # Budget exhausted or upstream error: don't flag.
        return True

    items = (resp.json() or {}).get("results") or []
    match = False
    for item in items[:5]:
        item_title = item.get("title") or item.get("display_name") or ""
        if fuzz.token_set_ratio(claimed_title.lower(), item_title.lower()) >= TITLE_MATCH_THRESHOLD:
            match = True
            break
    if cache is not None:
        cache.set(cache_key, match)
    return match


def _layer5_author_title_coherence(
    reference: Reference,
    found_authors: list[tuple[Author, dict]],
    *,
    client: httpx.Client,
    cache: CacheStore | None,
) -> LayerSignal:
    """Verify that at least one listed author published a work matching the title.

    Skip when:
    - The citation has no claimed title (nothing to match against).
    - L3 found no authors at all (L3 already flagged; this would be redundant).
    """
    if not reference.raw.title:
        return LayerSignal(
            layer="L5_author_title",
            flagged=False,
            reasoning="No claimed title; layer not applicable.",
        )
    if not found_authors:
        return LayerSignal(
            layer="L5_author_title",
            flagged=False,
            reasoning="No authors found in L3; layer not applicable.",
        )
    if not reference.raw.journal:
        # L5 is article-only. Books and grey literature are indexed too sparsely
        # in OpenAlex for the "author has this work" signal to be reliable, and
        # empirically the layer drives book false-positive rate above 0.6. We
        # accept the loss of recall on book fabrications in exchange.
        return LayerSignal(
            layer="L5_author_title",
            flagged=False,
            reasoning="No journal field (book / grey lit) — L5 not applicable.",
        )

    matched_authors: list[str] = []
    for input_author, record in found_authors:
        if _author_published_title(
            record.get("id"), reference.raw.title, client=client, cache=cache
        ):
            matched_authors.append(input_author.display())

    if matched_authors:
        return LayerSignal(
            layer="L5_author_title",
            flagged=False,
            reasoning=(
                "Listed author(s) have a published work matching the claimed title: "
                + ", ".join(matched_authors)
            ),
        )
    return LayerSignal(
        layer="L5_author_title",
        flagged=True,
        reasoning=(
            "Listed authors exist but none of their indexed works match the "
            "claimed title — real-author / invented-title pattern."
        ),
    )


# ----- Layer 6: aggregate verdict --------------------------------------------


_VERDICT_RANK = {
    HallucinationVerdict.LIKELY_HALLUCINATED: 3,
    HallucinationVerdict.SUSPICIOUS: 2,
    HallucinationVerdict.REAL_LOW_CONFIDENCE: 1,
    HallucinationVerdict.REAL_HIGH_CONFIDENCE: 0,
}
_VERDICT_BY_RANK = {v: k for k, v in _VERDICT_RANK.items()}


def _downgrade(verdict: HallucinationVerdict) -> HallucinationVerdict:
    """Move one step toward 'suspicious' for low-coverage refs."""
    rank = _VERDICT_RANK.get(verdict)
    if rank is None or rank >= 3:
        return verdict
    return _VERDICT_BY_RANK[rank + 1]


def _low_coverage_caveats(reference: Reference) -> list[str]:
    """Return caveats that apply to this reference; empty list if none."""
    caveats: list[str] = []
    if reference.raw.year and reference.raw.year < 2000:
        caveats.append(f"pre-2000 paper (year {reference.raw.year}) — limited DB coverage")
    if not reference.raw.journal:
        caveats.append("no journal field — likely book / working paper / preprint")
    # Non-English detection is fragile from metadata alone; for MVP we rely on
    # OpenAlex to flag it implicitly via L3/L4 sparseness. Future work could
    # query the resolved work's `language` field.
    return caveats


# L1 (DOI integrity) and L5 (author-title coherence) are *coherence* checks:
# they compare fields against each other. They are stronger signals than the
# *existence* checks (L2, L3, L4), which only ask whether a field appears in
# any database. A single coherence flag alone is enough to elevate to suspicious.
_STRONG_LAYERS = {"L1_doi_integrity", "L5_author_title"}


def _aggregate(reference: Reference, signals: list[LayerSignal]) -> HallucinationCheck:
    red_flags = sum(1 for s in signals if s.flagged)
    strong_flags = sum(1 for s in signals if s.flagged and s.layer in _STRONG_LAYERS)
    l2_fired = any(s.layer == "L2_cross_db" and s.flagged for s in signals)
    has_doi_resolution = (
        reference.resolved_doi is not None and reference.status == ResolutionStatus.RESOLVED
    )

    # Verdict rules:
    # - 4+ total flags, OR 3+ total with at least one strong → likely_hallucinated
    # - 2+ total flags, OR any single strong flag → suspicious
    # - else: real_*_confidence by whether the DOI resolved
    if red_flags >= 4 or (red_flags >= 3 and strong_flags >= 1):
        verdict = HallucinationVerdict.LIKELY_HALLUCINATED
    elif red_flags >= 2 or strong_flags >= 1:
        verdict = HallucinationVerdict.SUSPICIOUS
    elif has_doi_resolution:
        verdict = HallucinationVerdict.REAL_HIGH_CONFIDENCE
    else:
        verdict = HallucinationVerdict.REAL_LOW_CONFIDENCE

    caveats = _low_coverage_caveats(reference)
    no_journal_caveat = any("no journal" in c for c in caveats)

    # Asymmetric caveat handling, two rules:
    # 1) real_high → real_low when any caveat applies.
    # 2) suspicious → real_low when the no-journal caveat applies AND L2 did not
    #    fire. This protects legitimate books / grey-literature whose L5 misses
    #    are driven by sparse OpenAlex coverage rather than by fabrication.
    if caveats and verdict == HallucinationVerdict.REAL_HIGH_CONFIDENCE:
        verdict = _downgrade(verdict)
    elif no_journal_caveat and verdict == HallucinationVerdict.SUSPICIOUS and not l2_fired:
        verdict = HallucinationVerdict.REAL_LOW_CONFIDENCE
        caveats.append("downgraded: book / grey-lit; coherence-only flag insufficient evidence")

    flagged_names = [s.layer for s in signals if s.flagged]
    n_layers = len(signals) or 1
    reasoning_lines = [
        f"{red_flags} red flag(s) across {n_layers} active layers: "
        + (", ".join(flagged_names) or "none"),
        f"Verdict: {verdict.value}.",
    ]
    if caveats:
        reasoning_lines.append("Caveats: " + "; ".join(caveats))

    return HallucinationCheck(
        verdict=verdict,
        red_flag_count=red_flags,
        signals=signals,
        reasoning=" ".join(reasoning_lines),
        caveats=caveats,
    )


# ----- Public entry point -----------------------------------------------------


def check_hallucination(
    reference: Reference,
    *,
    cache: CacheStore | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    skip_network_layers: bool = False,
) -> HallucinationCheck:
    """Run the full five-layer hallucination check on a single reference.

    Set `skip_network_layers=True` to run only L1 + L2 (no OpenAlex calls).
    Useful for fast smoke checks and unit tests.
    """
    signals = [_layer1_doi_integrity(reference), _layer2_cross_db(reference)]
    if skip_network_layers:
        signals.extend(
            [
                LayerSignal(
                    layer="L3_authors",
                    flagged=False,
                    reasoning="Skipped (network layers disabled).",
                ),
                LayerSignal(
                    layer="L4_venue",
                    flagged=False,
                    reasoning="Skipped (network layers disabled).",
                ),
                LayerSignal(
                    layer="L5_author_title",
                    flagged=False,
                    reasoning="Skipped (network layers disabled).",
                ),
            ]
        )
    else:
        with _client(timeout_s) as client:
            l3_signal, found_authors = _layer3_authors(reference, client=client, cache=cache)
            signals.append(l3_signal)
            signals.append(_layer4_venue(reference, client=client, cache=cache))
            signals.append(
                _layer5_author_title_coherence(reference, found_authors, client=client, cache=cache)
            )
    return _aggregate(reference, signals)
