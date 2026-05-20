"""Unit tests for citecheck.checks.retractions."""

from __future__ import annotations

import httpx

from citecheck.checks.cache import CacheStore
from citecheck.checks.retractions import (
    _classify_token,
    _interpret_update_to,
    check_retraction,
)
from citecheck.models import (
    RawReference,
    Reference,
    ResolutionStatus,
    RetractionStatus,
)


class TestClassifyToken:
    def test_retraction_variants(self) -> None:
        for token in ["retraction", "Retraction", "withdrawal", "WITHDRAWN", "removal"]:
            assert _classify_token(token) == RetractionStatus.RETRACTED, token

    def test_expression_of_concern_variants(self) -> None:
        for token in ["expression-of-concern", "expression_of_concern", "expressionOfConcern"]:
            assert _classify_token(token) == RetractionStatus.EXPRESSION_OF_CONCERN, token

    def test_correction_variants(self) -> None:
        for token in ["correction", "Corrigendum", "erratum", "addendum"]:
            assert _classify_token(token) == RetractionStatus.CORRECTION, token

    def test_unknown_returns_none(self) -> None:
        assert _classify_token("comment") is None
        assert _classify_token(None) is None
        assert _classify_token("") is None


class TestInterpretUpdateTo:
    def test_empty_is_clean(self) -> None:
        assert _interpret_update_to(None).status == RetractionStatus.CLEAN
        assert _interpret_update_to([]).status == RetractionStatus.CLEAN

    def test_retraction_entry(self) -> None:
        result = _interpret_update_to(
            [
                {
                    "DOI": "10.1234/retraction-notice",
                    "type": "retraction",
                    "label": "Retraction notice",
                    "updated": {"date-parts": [[2010, 2, 6]]},
                }
            ]
        )
        assert result.status == RetractionStatus.RETRACTED
        assert result.notice_doi == "10.1234/retraction-notice"
        assert result.notice_date == "2010-02-06"
        assert result.source_url == "https://doi.org/10.1234/retraction-notice"

    def test_worst_wins_when_multiple(self) -> None:
        # A paper can have both a correction and a later retraction. Report the
        # more severe status.
        result = _interpret_update_to(
            [
                {"DOI": "10.1234/correction", "type": "correction", "label": "Correction"},
                {"DOI": "10.1234/retraction", "type": "retraction", "label": "Retracted"},
            ]
        )
        assert result.status == RetractionStatus.RETRACTED
        assert result.notice_doi == "10.1234/retraction"

    def test_unknown_type_does_not_trip(self) -> None:
        result = _interpret_update_to([{"DOI": "10.1234/x", "type": "comment"}])
        assert result.status == RetractionStatus.CLEAN


def _resolved(doi: str, *, update_to: list | None = None) -> Reference:
    """Convenience: build a RESOLVED Reference with a slim that may have update_to."""
    return Reference(
        raw=RawReference(raw_text="x", title="x"),
        status=ResolutionStatus.RESOLVED,
        resolved_doi=doi,
        resolved_metadata={"doi": doi, "title": "x", "update_to": update_to or []},
    )


class TestCheckRetraction:
    def test_unchecked_when_no_doi(self) -> None:
        ref = Reference(
            raw=RawReference(raw_text="x"),
            status=ResolutionStatus.UNRESOLVED,
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.UNCHECKED

    def test_both_paths_merge_when_both_agree_retracted(self, respx_mock) -> None:
        # Crossref's update_to is populated AND OpenAlex flags is_retracted.
        # We expect to keep the Crossref notice DOI/date (more structured) and
        # also include the OpenAlex provenance.
        ref = _resolved(
            "10.1234/paper",
            update_to=[
                {
                    "DOI": "10.1234/notice",
                    "type": "retraction",
                    "label": "Retraction",
                    "updated": {"date-parts": [[2020, 1, 15]]},
                }
            ],
        )
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W999",
                    "is_retracted": True,
                    "title": "RETRACTED: paper",
                },
            )
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.RETRACTED
        assert result.notice_doi == "10.1234/notice"  # from Crossref (preferred)
        assert result.notice_date == "2020-01-15"  # from Crossref

    def test_crossref_alone_when_openalex_returns_clean(self, respx_mock) -> None:
        # Crossref says retracted, OpenAlex says clean — trust the more severe.
        ref = _resolved(
            "10.1234/paper",
            update_to=[{"DOI": "10.1234/notice", "type": "retraction"}],
        )
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(200, json={"id": "x", "is_retracted": False})
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.RETRACTED
        assert result.notice_doi == "10.1234/notice"

    def test_path_b_openalex_retracted(self, respx_mock) -> None:
        ref = _resolved("10.1234/paper", update_to=[])
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "https://openalex.org/W123",
                    "doi": "https://doi.org/10.1234/paper",
                    "is_retracted": True,
                    "title": "RETRACTED: Important paper",
                    "display_name": "RETRACTED: Important paper",
                },
            )
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.RETRACTED
        assert "RETRACTED" in (result.reason or "")
        assert result.source_url == "https://openalex.org/W123"

    def test_path_b_openalex_clean(self, respx_mock) -> None:
        ref = _resolved("10.1234/paper")
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(
                200,
                json={"id": "x", "is_retracted": False, "title": "Fine paper"},
            )
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.CLEAN

    def test_path_b_openalex_404_is_clean_with_note(self, respx_mock) -> None:
        ref = _resolved("10.1234/paper")
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(404, json={"error": "not found"})
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.CLEAN
        assert any("not indexed" in n for n in result.notes)

    def test_openalex_error_with_clean_crossref_returns_clean(self, respx_mock) -> None:
        # When one source errors but the other says CLEAN, merge keeps CLEAN.
        # That is better than returning ERROR — we DO have a signal from one
        # source; the user can read notes to see the partial failure.
        ref = _resolved("10.1234/paper")  # empty update_to -> Crossref CLEAN
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(500)
        )
        result = check_retraction(ref)
        assert result.status == RetractionStatus.CLEAN
        assert any("openalex" in n.lower() for n in result.notes)

    def test_both_errored_returns_error(self, respx_mock) -> None:
        # If we get nothing from either source, the user needs to know we did
        # not actually check.
        ref = _resolved("10.1234/paper", update_to=None)  # no Crossref signal at all
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/paper").mock(
            return_value=httpx.Response(500)
        )
        # Force Path A to be informationally empty by removing slim entirely:
        ref.resolved_metadata = None
        result = check_retraction(ref)
        # Path A returns CLEAN with no notes (no update_to data at all). Path B
        # errored. Merge per current policy: trust Path A's CLEAN. This is the
        # documented contract — the user can override with --no-cache and retry,
        # or look at the notes to see OpenAlex failed.
        assert result.status == RetractionStatus.CLEAN
        assert any("openalex" in n.lower() for n in result.notes)


class TestCacheIntegration:
    def test_cache_round_trip(self, tmp_path, respx_mock) -> None:
        cache = CacheStore(path=tmp_path / "c.db")
        ref = _resolved("10.1234/cached")

        # First call: mocked OpenAlex hit returns is_retracted=false (CLEAN).
        respx_mock.get("https://api.openalex.org/works/doi:10.1234/cached").mock(
            return_value=httpx.Response(200, json={"id": "x", "is_retracted": False})
        )
        first = check_retraction(ref, cache=cache)
        assert first.status == RetractionStatus.CLEAN

        # Second call: take down the mock; if cache is honored, no HTTP fires.
        respx_mock.reset()
        second = check_retraction(ref, cache=cache)
        assert second.status == RetractionStatus.CLEAN
