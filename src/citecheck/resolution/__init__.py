"""Top-level resolution dispatch.

Orchestrates the three resolution paths:
1. Crossref DOI lookup (fast, exact, used whenever a DOI is on the input)
2. Crossref metadata search (used when no DOI; first metadata pass)
3. OpenAlex search (fallback when Crossref metadata returns UNRESOLVED or ERROR)

Keeping the dispatch here (rather than inside crossref.py) avoids the circular
import that would otherwise happen — openalex.py imports scoring helpers from
crossref.py, and dispatch must import from both.
"""

from __future__ import annotations

from citecheck.models import RawReference, Reference, ResolutionStatus
from citecheck.resolution import crossref, openalex

DEFAULT_TIMEOUT_S = 30.0


def resolve(raw: RawReference, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Reference:
    """Resolve a raw reference to a canonical record. Tries DOI, metadata, OpenAlex in order."""
    fell_back_from_doi = False

    # 1) DOI path — fast and exact when GROBID or the raw_text regex gave us a DOI.
    if raw.doi:
        doi_result = crossref.resolve_by_doi(raw, timeout_s=timeout_s)
        if doi_result.status == ResolutionStatus.RESOLVED:
            return doi_result
        # DOI lookup misfired (404, error, or normalized to None). Keep going.
        fell_back_from_doi = True

    # 2) Crossref metadata — primary path when no DOI is available.
    crossref_result: Reference | None = None
    if raw.title:
        crossref_result = crossref.resolve_by_metadata(raw, timeout_s=timeout_s)
        if fell_back_from_doi:
            crossref_result.notes.append("resolve: DOI lookup missed; fell back to metadata")
        if crossref_result.status in (
            ResolutionStatus.RESOLVED,
            ResolutionStatus.AMBIGUOUS,
        ):
            return crossref_result

    # 3) OpenAlex fallback — independent index, often catches preprints / older
    # works Crossref does not have.
    if raw.title:
        oa_result = openalex.resolve_by_openalex(raw, timeout_s=timeout_s)
        if oa_result.status == ResolutionStatus.RESOLVED:
            oa_result.notes.insert(0, "resolve: Crossref missed; OpenAlex matched")
            return oa_result
        if oa_result.status == ResolutionStatus.AMBIGUOUS and (
            crossref_result is None or crossref_result.status != ResolutionStatus.AMBIGUOUS
        ):
            oa_result.notes.insert(0, "resolve: Crossref missed; OpenAlex ambiguous")
            return oa_result

    # 4) Nothing worked. Prefer to return whichever of the previous attempts
    # carries the most diagnostic information for the user.
    if crossref_result is not None:
        return crossref_result
    return Reference(
        raw=raw,
        status=ResolutionStatus.UNRESOLVED,
        notes=["resolve: no DOI, no title — nothing to look up"],
    )


__all__ = ["resolve"]
