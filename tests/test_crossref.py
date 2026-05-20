"""Unit tests for citecheck.resolution.crossref.

Uses respx's pytest fixture (auto-discovered from `respx_mock` parameter).
The class-level `@respx.mock` decorator silently breaks pytest collection,
so methods take the fixture explicitly instead.
"""

from __future__ import annotations

import httpx

from citecheck.models import Author, RawReference, ResolutionStatus
from citecheck.resolution import resolve
from citecheck.resolution.crossref import (
    AMBIGUOUS_THRESHOLD,
    RESOLVED_THRESHOLD,
    _user_agent,
    resolve_by_doi,
    resolve_by_metadata,
)


def _work(doi: str, title: str, year: int, **extra) -> dict:
    return {
        "DOI": doi,
        "title": [title],
        "issued": {"date-parts": [[year]]},
        "container-title": [extra.get("journal", "Journal X")],
        "volume": extra.get("volume", "10"),
        "issue": extra.get("issue", "1"),
        "page": extra.get("page", "1-20"),
        "type": "journal-article",
        "publisher": "Some Publisher",
        "is-referenced-by-count": 42,
    }


class TestUserAgent:
    def test_includes_mailto_when_env_set(self, monkeypatch) -> None:
        monkeypatch.setenv("CITECHECK_CONTACT_EMAIL", "foo@bar.com")
        ua = _user_agent()
        assert "mailto:foo@bar.com" in ua
        assert "citecheck/" in ua

    def test_omits_mailto_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("CITECHECK_CONTACT_EMAIL", raising=False)
        ua = _user_agent()
        assert "mailto:" not in ua


class TestResolveByDoi:
    def test_resolved(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works/10.1257/aer.91.5.1369").mock(
            return_value=httpx.Response(
                200, json={"status": "ok", "message": _work("10.1257/aer.91.5.1369", "T", 2001)}
            )
        )
        raw = RawReference(raw_text="x", doi="10.1257/aer.91.5.1369")
        ref = resolve_by_doi(raw)
        assert ref.status == ResolutionStatus.RESOLVED
        assert ref.resolved_doi == "10.1257/aer.91.5.1369"
        assert ref.resolved_metadata["title"] == "T"
        assert ref.resolution_score is None  # DOI matches are exact

    def test_404_marks_unresolved(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works/10.9999/missing").mock(
            return_value=httpx.Response(404, json={"status": "error"})
        )
        raw = RawReference(raw_text="x", doi="10.9999/missing")
        ref = resolve_by_doi(raw)
        assert ref.status == ResolutionStatus.UNRESOLVED

    def test_no_doi_input(self) -> None:
        ref = resolve_by_doi(RawReference(raw_text="x"))
        assert ref.status == ResolutionStatus.UNRESOLVED

    def test_500_marks_error(self, respx_mock) -> None:
        # 5xx is a real HTTP response, not a transport error — tenacity does NOT
        # retry under our config, so one mock is enough.
        respx_mock.get("https://api.crossref.org/works/10.1234/abcxyz").mock(
            return_value=httpx.Response(500)
        )
        raw = RawReference(raw_text="x", doi="10.1234/abcxyz")
        ref = resolve_by_doi(raw)
        assert ref.status == ResolutionStatus.ERROR


class TestResolveByMetadata:
    def test_high_score_resolves(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "message": {
                        "items": [
                            _work("10.1234/a", "Role of institutions in growth", 2001),
                            _work("10.1234/b", "Wildly different paper", 2010),
                        ]
                    },
                },
            )
        )
        raw = RawReference(
            raw_text="x",
            title="Role of institutions in growth",
            authors=[Author(family="Acemoglu")],
            year=2001,
        )
        ref = resolve_by_metadata(raw)
        assert ref.status == ResolutionStatus.RESOLVED
        assert ref.resolved_doi == "10.1234/a"
        assert ref.resolution_score is not None
        assert ref.resolution_score > 0.85

    def test_ambiguous_when_mid_range_score(self, respx_mock) -> None:
        # Crossref's top candidate is only a partial title match — score lands in
        # [AMBIGUOUS_THRESHOLD, RESOLVED_THRESHOLD). We still return a DOI but
        # flag it so the caller can review before relying on it.
        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "message": {
                        "items": [
                            _work("10.1234/a", "Role of institutions in growth", 2001),
                        ]
                    },
                },
            )
        )
        raw = RawReference(
            raw_text="x",
            title="role institutions",  # truncated/noisy — partial match only
            authors=[Author(family="Smith")],
            year=2001,
        )
        ref = resolve_by_metadata(raw)
        assert ref.status == ResolutionStatus.AMBIGUOUS
        assert ref.resolved_doi == "10.1234/a"
        assert ref.resolution_score is not None
        assert AMBIGUOUS_THRESHOLD <= ref.resolution_score < RESOLVED_THRESHOLD

    def test_low_score_unresolved(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "message": {"items": [_work("10.1234/a", "Completely unrelated paper", 2001)]},
                },
            )
        )
        raw = RawReference(raw_text="x", title="Role of institutions in growth")
        ref = resolve_by_metadata(raw)
        assert ref.status == ResolutionStatus.UNRESOLVED

    def test_no_results_unresolved(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json={"status": "ok", "message": {"items": []}})
        )
        raw = RawReference(raw_text="x", title="Some title")
        ref = resolve_by_metadata(raw)
        assert ref.status == ResolutionStatus.UNRESOLVED

    def test_no_title_input(self) -> None:
        ref = resolve_by_metadata(RawReference(raw_text="x"))
        assert ref.status == ResolutionStatus.UNRESOLVED


class TestResolveDispatch:
    def test_with_doi_uses_doi_endpoint(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works/10.1234/a").mock(
            return_value=httpx.Response(
                200, json={"status": "ok", "message": _work("10.1234/a", "T", 2020)}
            )
        )
        ref = resolve(RawReference(raw_text="x", doi="10.1234/a"))
        assert ref.status == ResolutionStatus.RESOLVED

    def test_without_doi_uses_metadata_endpoint(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "message": {"items": [_work("10.1234/a", "Exact Title Match", 2020)]},
                },
            )
        )
        ref = resolve(RawReference(raw_text="x", title="Exact Title Match", year=2020))
        assert ref.status == ResolutionStatus.RESOLVED

    def test_doi_404_falls_back_to_metadata(self, respx_mock) -> None:
        respx_mock.get("https://api.crossref.org/works/10.9999/missing").mock(
            return_value=httpx.Response(404)
        )
        respx_mock.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(
                200,
                json={
                    "status": "ok",
                    "message": {"items": [_work("10.1234/found", "Recovered title", 2020)]},
                },
            )
        )
        raw = RawReference(raw_text="x", doi="10.9999/missing", title="Recovered title", year=2020)
        ref = resolve(raw)
        assert ref.status == ResolutionStatus.RESOLVED
        assert ref.resolved_doi == "10.1234/found"
        assert any("fell back to metadata" in n for n in ref.notes)
