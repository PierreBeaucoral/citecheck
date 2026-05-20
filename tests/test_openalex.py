"""Unit tests for citecheck.resolution.openalex."""

from __future__ import annotations

import httpx

from citecheck.models import RawReference, ResolutionStatus
from citecheck.resolution.openalex import _slim_openalex_work, resolve_by_openalex


def _oa_work(
    doi: str | None,
    title: str,
    year: int,
    *,
    journal: str = "OA Journal",
    cited_by: int = 7,
) -> dict:
    return {
        "id": "https://openalex.org/W123",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": title,
        "display_name": title,
        "publication_year": year,
        "type": "article",
        "cited_by_count": cited_by,
        "primary_location": {
            "source": {
                "display_name": journal,
                "host_organization_name": "Some Publisher",
            }
        },
        "biblio": {"volume": "1", "issue": "2", "first_page": "10", "last_page": "20"},
    }


class TestSlimAdapter:
    def test_strips_doi_url_prefix(self) -> None:
        slim = _slim_openalex_work(_oa_work("10.1234/abc", "T", 2020))
        assert slim["doi"] == "10.1234/abc"

    def test_handles_missing_doi(self) -> None:
        slim = _slim_openalex_work(_oa_work(None, "T", 2020))
        assert slim["doi"] is None

    def test_pages_formatted(self) -> None:
        slim = _slim_openalex_work(_oa_work("10.1234/x", "T", 2020))
        assert slim["page"] == "10-20"


class TestResolveByOpenAlex:
    def test_resolved(self, respx_mock) -> None:
        respx_mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        _oa_work("10.1234/match", "Role of institutions in growth", 2001),
                    ]
                },
            )
        )
        ref = resolve_by_openalex(
            RawReference(raw_text="x", title="Role of institutions in growth", year=2001)
        )
        assert ref.status == ResolutionStatus.RESOLVED
        assert ref.resolved_doi == "10.1234/match"
        assert ref.resolved_metadata["title"] == "Role of institutions in growth"

    def test_top_candidate_without_doi_is_unresolved(self, respx_mock) -> None:
        # OpenAlex sometimes indexes works that have no DOI (grey literature).
        # We refuse to call those RESOLVED because downstream phases need a DOI.
        respx_mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                200, json={"results": [_oa_work(None, "Exact title match", 2020)]}
            )
        )
        ref = resolve_by_openalex(RawReference(raw_text="x", title="Exact title match"))
        assert ref.status == ResolutionStatus.UNRESOLVED
        assert "no DOI" in " ".join(ref.notes)

    def test_no_results(self, respx_mock) -> None:
        respx_mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        ref = resolve_by_openalex(RawReference(raw_text="x", title="Anything"))
        assert ref.status == ResolutionStatus.UNRESOLVED

    def test_no_title_input(self) -> None:
        ref = resolve_by_openalex(RawReference(raw_text="x"))
        assert ref.status == ResolutionStatus.UNRESOLVED

    def test_500_error(self, respx_mock) -> None:
        respx_mock.get("https://api.openalex.org/works").mock(return_value=httpx.Response(500))
        ref = resolve_by_openalex(RawReference(raw_text="x", title="Any title"))
        assert ref.status == ResolutionStatus.ERROR


class TestResolveDispatchWithOpenAlexFallback:
    """End-to-end dispatch tests for the top-level resolve()."""

    def test_openalex_picks_up_crossref_miss(self, respx_mock) -> None:
        from citecheck.resolution import resolve

        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json={"status": "ok", "message": {"items": []}})
        )
        respx_mock.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(
                200,
                json={"results": [_oa_work("10.1234/found-via-oa", "Niche title", 2018)]},
            )
        )
        ref = resolve(RawReference(raw_text="x", title="Niche title", year=2018))
        assert ref.status == ResolutionStatus.RESOLVED
        assert ref.resolved_doi == "10.1234/found-via-oa"
        assert any("OpenAlex matched" in n for n in ref.notes)

    def test_crossref_resolved_short_circuits(self, respx_mock) -> None:
        from citecheck.resolution import resolve

        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1234/crossref-hit",
                                "title": ["Exact title match"],
                                "issued": {"date-parts": [[2020]]},
                                "container-title": ["J"],
                                "type": "journal-article",
                            }
                        ]
                    },
                },
            )
        )
        # No OpenAlex mock set up — if dispatch called OpenAlex, the test would error.
        ref = resolve(RawReference(raw_text="x", title="Exact title match", year=2020))
        assert ref.status == ResolutionStatus.RESOLVED
        assert ref.resolved_doi == "10.1234/crossref-hit"
        assert not any("OpenAlex" in n for n in ref.notes)
