"""Unit tests for citecheck.checks.journal_quality (Phase 4 v2 multi-signal)."""

from __future__ import annotations

import httpx
import pytest

from citecheck.checks.cache import CacheStore
from citecheck.checks.journal_quality import (
    _concern_publisher_match,
    _score_journal,
    _score_to_level,
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


# OpenAlex /sources empty response — used by every test that does not care
# about OpenAlex metadata so that check_journal_quality does not try to make
# a real network call against the budget circuit.
_EMPTY_OPENALEX_SOURCES = {"results": [], "meta": {"count": 0}}


@pytest.fixture(autouse=True)
def _no_budget_state(tmp_path, monkeypatch):
    """Isolate the OpenAlex circuit-breaker state file per test."""
    monkeypatch.setenv("CITECHECK_CACHE_DIR", str(tmp_path))
    yield


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

    def test_held_out_target_now_matched(self) -> None:
        # v2 added held-out predatory targets to the concern list.
        assert (
            _concern_publisher_match("Insight Medical Publishing") == "insight medical publishing"
        )
        assert _concern_publisher_match("SciFed Journals") == "scifed"

    def test_contested_entries_removed(self) -> None:
        # v2 dropped Hindawi/MDPI/Frontiers from the static list.
        assert _concern_publisher_match("Frontiers in Psychology") is None
        assert _concern_publisher_match("MDPI Sustainability") is None
        assert _concern_publisher_match("Hindawi Mathematical Problems") is None

    def test_empty(self) -> None:
        assert _concern_publisher_match("") is None
        assert _concern_publisher_match(None) is None  # type: ignore[arg-type]


# ---- scoring function (pure, no network) -----------------------------------


class TestScoreJournal:
    def test_doaj_alone_pushes_low(self) -> None:
        score, _ = _score_journal(
            doaj_listed=True,
            is_indexed_in_scopus=None,
            h_index=None,
            works_count=None,
            cited_by_count=None,
            apc_usd=None,
            concern_match=None,
            portfolio_size=None,
        )
        assert score == pytest.approx(-0.5)
        assert _score_to_level(score) == JournalRiskLevel.LOW

    def test_concern_alone_pushes_high(self) -> None:
        score, _ = _score_journal(
            doaj_listed=None,
            is_indexed_in_scopus=None,
            h_index=None,
            works_count=None,
            cited_by_count=None,
            apc_usd=None,
            concern_match="omics international",
            portfolio_size=None,
        )
        assert score == pytest.approx(0.6)
        assert _score_to_level(score) == JournalRiskLevel.HIGH

    def test_scopus_plus_hindex_low(self) -> None:
        score, _ = _score_journal(
            doaj_listed=None,
            is_indexed_in_scopus=True,
            h_index=45,
            works_count=12000,
            cited_by_count=240000,
            apc_usd=None,
            concern_match=None,
            portfolio_size=None,
        )
        # -0.3 (scopus) -0.3 (h-index) -0.2 (citations/article >= 5) = -0.8
        assert score == pytest.approx(-0.8)
        assert _score_to_level(score) == JournalRiskLevel.LOW

    def test_journal_flood_pattern_high(self) -> None:
        score, signals = _score_journal(
            doaj_listed=None,
            is_indexed_in_scopus=False,
            h_index=2,
            works_count=1500,
            cited_by_count=20,
            apc_usd=None,
            concern_match=None,
            portfolio_size=None,
        )
        assert "journal-flood" in " ".join(signals)
        assert score == pytest.approx(0.4)
        assert _score_to_level(score) == JournalRiskLevel.HIGH

    def test_high_apc_no_doaj_medium_only(self) -> None:
        # +0.3 alone does not cross the HIGH threshold (>= 0.4).
        score, _ = _score_journal(
            doaj_listed=False,
            is_indexed_in_scopus=None,
            h_index=None,
            works_count=None,
            cited_by_count=None,
            apc_usd=2000,
            concern_match=None,
            portfolio_size=None,
        )
        assert score == pytest.approx(0.3)
        assert _score_to_level(score) == JournalRiskLevel.MEDIUM

    def test_neutral_defaults_medium(self) -> None:
        score, signals = _score_journal(
            doaj_listed=None,
            is_indexed_in_scopus=None,
            h_index=None,
            works_count=None,
            cited_by_count=None,
            apc_usd=None,
            concern_match=None,
            portfolio_size=None,
        )
        assert score == 0.0
        assert signals == []
        assert _score_to_level(score) == JournalRiskLevel.MEDIUM


# ---- full check (mocks DOAJ + OpenAlex) ------------------------------------


def _mock_openalex_empty(respx_mock) -> None:
    respx_mock.get(url__regex=r"https://api\.openalex\.org/sources.*").mock(
        return_value=httpx.Response(200, json=_EMPTY_OPENALEX_SOURCES)
    )


def _mock_doaj_empty(respx_mock) -> None:
    respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
        return_value=httpx.Response(200, json={"results": []})
    )


class TestCheckJournalQuality:
    def test_no_journal_unchecked(self) -> None:
        result = check_journal_quality(_ref(None))
        assert result.risk_level == JournalRiskLevel.UNCHECKED
        assert result.risk_score is None

    def test_concern_list_alone_high(self, respx_mock) -> None:
        _mock_doaj_empty(respx_mock)
        _mock_openalex_empty(respx_mock)
        result = check_journal_quality(_ref("OMICS International"))
        assert result.risk_level == JournalRiskLevel.HIGH
        assert result.on_concern_list is True
        assert result.matched_publisher == "omics international"
        assert result.risk_score is not None and result.risk_score >= 0.4

    def test_doaj_listed_pushes_low(self, respx_mock) -> None:
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(
                200,
                json={"results": [{"bibjson": {"title": "Open Journal of Foo"}}]},
            )
        )
        _mock_openalex_empty(respx_mock)
        result = check_journal_quality(_ref("Open Journal of Foo"))
        assert result.doaj_listed is True
        assert result.risk_level == JournalRiskLevel.LOW

    def test_scopus_and_hindex_push_low(self, respx_mock) -> None:
        _mock_doaj_empty(respx_mock)
        # OpenAlex returns a reputable source: Scopus-indexed, h-index 50,
        # ~10 citations / article.
        respx_mock.get(url__regex=r"https://api\.openalex\.org/sources.*").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/S100",
                            "display_name": "Top Finance Journal",
                            "host_organization": "https://openalex.org/P900",
                            "host_organization_name": "Wiley",
                            "country_code": "US",
                            "works_count": 5000,
                            "cited_by_count": 60000,
                            "summary_stats": {"h_index": 50, "2yr_mean_citedness": 4.1},
                            "apc_prices": [],
                            "is_in_doaj": False,
                            "is_indexed_in_scopus": True,
                        }
                    ],
                    "meta": {"count": 1},
                },
            )
        )
        result = check_journal_quality(_ref("Top Finance Journal"))
        assert result.is_indexed_in_scopus is True
        assert result.h_index == 50
        assert result.risk_level == JournalRiskLevel.LOW
        # -0.3 scopus -0.3 h-index -0.2 citations/article = -0.8
        assert result.risk_score is not None and result.risk_score <= -0.5

    def test_held_out_predatory_via_concern_list(self, respx_mock) -> None:
        _mock_doaj_empty(respx_mock)
        _mock_openalex_empty(respx_mock)
        result = check_journal_quality(_ref("Insight Medical Publishing"))
        assert result.risk_level == JournalRiskLevel.HIGH
        assert result.matched_publisher == "insight medical publishing"

    def test_doaj_listed_mdpi_no_longer_falsely_flagged(self, respx_mock) -> None:
        # v1 regression: Frontiers/MDPI/Hindawi static-listed AS predatory.
        # v2 must let DOAJ-listed MDPI journals fall to LOW.
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(
                200, json={"results": [{"bibjson": {"title": "Sustainability"}}]}
            )
        )
        _mock_openalex_empty(respx_mock)
        result = check_journal_quality(_ref("Sustainability"))
        assert result.risk_level == JournalRiskLevel.LOW
        assert result.on_concern_list is False

    def test_doaj_error_uses_openalex_fallback(self, respx_mock) -> None:
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(500)
        )
        # OpenAlex still gives us reputation signals.
        respx_mock.get(url__regex=r"https://api\.openalex\.org/sources.*").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "id": "https://openalex.org/S200",
                            "display_name": "Some Journal",
                            "host_organization": None,
                            "works_count": 100,
                            "cited_by_count": 200,
                            "summary_stats": {"h_index": 25},
                            "apc_prices": [],
                            "is_in_doaj": True,
                            "is_indexed_in_scopus": True,
                        }
                    ],
                    "meta": {"count": 1},
                },
            )
        )
        result = check_journal_quality(_ref("Some Journal"))
        # DOAJ failed but OpenAlex's is_in_doaj fills in -> doaj_listed True.
        assert result.doaj_listed is True
        # -0.5 DOAJ -0.3 Scopus -0.3 h-index >= 20 = -1.1
        assert result.risk_level == JournalRiskLevel.LOW

    def test_cache_hit_on_second_call(self, tmp_path, respx_mock) -> None:
        cache = CacheStore(path=tmp_path / "c.db")
        respx_mock.get(url__regex=r"https://doaj\.org/api/v3/search/journals/.*").mock(
            return_value=httpx.Response(200, json={"results": [{"bibjson": {"title": "X"}}]})
        )
        _mock_openalex_empty(respx_mock)
        first = check_journal_quality(_ref("X"), cache=cache)
        assert first.risk_level == JournalRiskLevel.LOW

        # Tear down the mocks; if cache works, no HTTP fires.
        respx_mock.reset()
        second = check_journal_quality(_ref("X"), cache=cache)
        assert second.risk_level == JournalRiskLevel.LOW
