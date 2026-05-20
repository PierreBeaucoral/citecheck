"""Unit tests for citecheck.models."""

from __future__ import annotations

import pytest

from citecheck.models import (
    Author,
    RawReference,
    Reference,
    ResolutionStatus,
    extract_doi_from_text,
    normalize_doi,
)


class TestNormalizeDoi:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10.1086/261876", "10.1086/261876"),
            ("https://doi.org/10.1086/261876", "10.1086/261876"),
            ("https://dx.doi.org/10.1086/261876", "10.1086/261876"),
            ("doi:10.1086/261876", "10.1086/261876"),
            ("DOI:10.1086/261876", "10.1086/261876"),
            ("  10.1086/261876  ", "10.1086/261876"),
            ("10.1086/ABC123", "10.1086/abc123"),
        ],
    )
    def test_normalizes_valid(self, raw: str, expected: str) -> None:
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", "not-a-doi", "10.foo", "10/foo", "http://x"])
    def test_rejects_invalid(self, bad: str) -> None:
        assert normalize_doi(bad) is None

    def test_none_passes_through(self) -> None:
        assert normalize_doi(None) is None


class TestExtractDoiFromText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "Smith, J. (2020). A paper. Journal, 12, 1-10. doi:10.1086/261876",
                "10.1086/261876",
            ),
            (
                "Trailing period in citation https://doi.org/10.1086/261876.",
                "10.1086/261876",
            ),
            (
                "DOI in parentheses (10.1086/261876).",
                "10.1086/261876",
            ),
            (
                "Multiple DOIs: 10.1086/261876 and 10.9999/other — take the first.",
                "10.1086/261876",
            ),
            (
                "No DOI here, just a plain reference.",
                None,
            ),
            (
                "Looks like a DOI: 10.1/x but the registrant is too short.",
                None,
            ),
            (None, None),
            ("", None),
        ],
    )
    def test_extraction(self, text, expected) -> None:
        assert extract_doi_from_text(text) == expected


class TestRawReference:
    def test_doi_validator_normalizes(self) -> None:
        r = RawReference(raw_text="x", doi="https://doi.org/10.1086/261876")
        assert r.doi == "10.1086/261876"

    def test_bad_doi_becomes_none(self) -> None:
        r = RawReference(raw_text="x", doi="not-a-doi")
        assert r.doi is None

    def test_year_bounds(self) -> None:
        with pytest.raises(ValueError):
            RawReference(raw_text="x", year=1200)
        with pytest.raises(ValueError):
            RawReference(raw_text="x", year=2200)

    def test_minimal_construction(self) -> None:
        r = RawReference(raw_text="Doe, J. (2020). A paper.")
        assert r.raw_text.startswith("Doe")
        assert r.authors == []
        assert r.doi is None


class TestAuthor:
    def test_display_with_given(self) -> None:
        assert Author(family="Doe", given="John").display() == "John Doe"

    def test_display_surname_only(self) -> None:
        assert Author(family="Doe").display() == "Doe"


class TestReference:
    def test_default_status_is_unresolved(self) -> None:
        ref = Reference(raw=RawReference(raw_text="x"))
        assert ref.status == ResolutionStatus.UNRESOLVED
        assert ref.resolved_doi is None
        assert ref.notes == []

    def test_resolved_doi_is_normalized(self) -> None:
        ref = Reference(
            raw=RawReference(raw_text="x"),
            status=ResolutionStatus.RESOLVED,
            resolved_doi="https://doi.org/10.1086/261876",
        )
        assert ref.resolved_doi == "10.1086/261876"

    def test_resolution_score_bounds(self) -> None:
        with pytest.raises(ValueError):
            Reference(raw=RawReference(raw_text="x"), resolution_score=1.5)
