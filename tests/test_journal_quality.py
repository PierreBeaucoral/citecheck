"""Unit tests for citecheck.checks.journal_quality."""

from __future__ import annotations

import httpx

from citecheck.checks.cache import CacheStore
from citecheck.checks.journal_quality import (
    _concern_publisher_match,
    check_journal_quality,
)
from citecheck.models import (
    JournalRiskLevel,
    RawReference,
    Reference,
    ResolutionStatus,
)


def _ref(journal: str | None) -> Reference:
    return Reference(
        raw=RawReference(raw_text="x", title="t", journal=journal),
        status=ResolutionStatus.RESOLVED,
    )


# ---- concern-list matching (no network) ------------------------------------


class TestConcernMatch:
    def test_known_predatory(self) -> None:
        assert _concern_publisher_match("OMICS International J.") == "omics international"

    def test_real_publisher_not_matched(self) -> None:
        assert _concern_publisher_match("Nature") is None
        assert _concern_publisher_match("American Economic Review") is None

    def test_case_insensitive(self) -> None:
        assert (
            _concern_publisher_match("scientific research publishing house")
            == "scientific research publishing"
        )

    def test_empty(self) -> None:
        assert _concern_publisher_match("") is None
        assert _concern_publisher_match(None) is None  # type: ignore[arg-type]


# ---- full check (mocks DOAJ) ------------------------------------------------


class TestCheckJournalQuality:
    def test_no_journal_unchecked(self) -> None:
        result = check_journal_quality(_ref(None))
        assert result.risk_level == JournalRiskLevel.UNCHECKED

    def test_concern_list_high_short_circuit(self, respx_mock) -> None:
        # No DOAJ mock set; if dispatch called DOAJ this would error.
        result = check_journal_quality(_ref("OMICS International"))
        assert result.risk_level == JournalRiskLevel.HIGH
        assert result.on_concern_list is True
        assert result.matched_publisher == "omics international"

    def test_doaj_listed_low(self, respx_mock) -> None:
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(
                200,
                json={"results": [{"bibjson": {"title": "Open Journal of Foo"}}]},
            )
        )
        result = check_journal_quality(_ref("Open Journal of Foo"))
        assert result.risk_level == JournalRiskLevel.LOW
        assert result.doaj_listed is True

    def test_doaj_unlisted_medium(self, respx_mock) -> None:
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        result = check_journal_quality(_ref("Some Niche Journal"))
        assert result.risk_level == JournalRiskLevel.MEDIUM
        assert result.doaj_listed is False

    def test_doaj_error_unchecked(self, respx_mock) -> None:
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(500)
        )
        result = check_journal_quality(_ref("Some Journal"))
        assert result.risk_level == JournalRiskLevel.UNCHECKED

    def test_cache_hit_on_second_call(self, tmp_path, respx_mock) -> None:
        cache = CacheStore(path=tmp_path / "c.db")
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(200, json={"results": [{"bibjson": {"title": "X"}}]})
        )
        first = check_journal_quality(_ref("X"), cache=cache)
        assert first.risk_level == JournalRiskLevel.LOW

        # Tear down the mock; if cache works, no HTTP fires.
        respx_mock.reset()
        second = check_journal_quality(_ref("X"), cache=cache)
        assert second.risk_level == JournalRiskLevel.LOW
