"""Unit tests for citecheck.metrics and citecheck.budget."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from citecheck import budget, metrics


def _isolate_state(tmp_path, monkeypatch) -> None:
    """Redirect both modules to a clean per-test directory."""
    monkeypatch.setenv("CITECHECK_METRICS_DIR", str(tmp_path))
    monkeypatch.setenv("CITECHECK_CACHE_DIR", str(tmp_path))
    # Reset the lazy module-level connection so each test starts fresh.
    metrics._CONN = None  # type: ignore[attr-defined]


# ---- metrics ---------------------------------------------------------------


class TestMetrics:
    def test_record_and_summary(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        metrics.record("openalex", "/works", cache_hit=False, status=200)
        metrics.record("openalex", "/works", cache_hit=True, status=200)
        metrics.record("crossref", "/works", cache_hit=False, status=200)
        s = metrics.summary(7)
        sources = {(r["source"], r["endpoint"]) for r in s}
        assert ("openalex", "/works") in sources
        assert ("crossref", "/works") in sources

    def test_cache_hit_pct(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        for _ in range(3):
            metrics.record("openalex", "/authors", cache_hit=True, status=200)
        metrics.record("openalex", "/authors", cache_hit=False, status=200)
        rows = [r for r in metrics.summary(7) if r["endpoint"] == "/authors"]
        assert rows and abs(rows[0]["cache_hit_pct"] - 0.75) < 0.01

    def test_no_db_dir_is_tolerated(self, tmp_path, monkeypatch) -> None:
        # Force a path we cannot write to and ensure record() doesn't raise.
        monkeypatch.setenv("CITECHECK_METRICS_DIR", "/nonexistent/path/that/cannot/exist")
        metrics._CONN = None  # type: ignore[attr-defined]
        # Should not raise even though the DB can't be opened.
        metrics.record("openalex", "/works", cache_hit=False, status=200)


# ---- budget ----------------------------------------------------------------


class TestBudget:
    def test_clean_state_is_ok(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        assert budget.openalex_ok() is True
        assert budget.exhausted_until() is None

    def test_mark_exhausted_opens_circuit(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        budget.mark_exhausted()
        assert budget.openalex_ok() is False
        until = budget.exhausted_until()
        assert until is not None
        # Default expiry is next midnight UTC, always > now.
        assert until > datetime.now(UTC)

    def test_auto_reset_after_expiry(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        # Mark exhausted with an expiry already in the past — read should
        # auto-reset.
        past = datetime.now(UTC) - timedelta(seconds=10)
        budget.mark_exhausted(until=past)
        # First read should clear the state.
        assert budget.exhausted_until() is None
        assert budget.openalex_ok() is True

    def test_is_budget_exhausted_response_detection(self) -> None:
        assert budget.is_budget_exhausted_response(
            429, "Insufficient budget. Resets at midnight UTC."
        )
        # Case insensitive substring match.
        assert budget.is_budget_exhausted_response(429, "ERROR: insufficient BUDGET")
        # Different 429 body should not trigger.
        assert not budget.is_budget_exhausted_response(429, "Too many requests")
        # Non-429 should never trigger.
        assert not budget.is_budget_exhausted_response(200, "Insufficient budget")
        assert not budget.is_budget_exhausted_response(503, "Insufficient budget")


# ---- openalex_get wrapper --------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict | None = None) -> _FakeResponse:
        self.calls.append((url, params or {}))
        return self.response


class TestOpenAlexGetWrapper:
    def test_normal_call_records_metric(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        client = _FakeClient(_FakeResponse(200, "ok"))
        resp = budget.openalex_get(client, "/works", search="x")
        assert resp is client.response
        # Metric recorded with status=200.
        rows = [r for r in metrics.summary(7) if r["source"] == "openalex"]
        assert any(r["endpoint"] == "/works" for r in rows)

    def test_short_circuit_when_exhausted(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        budget.mark_exhausted()
        client = _FakeClient(_FakeResponse(200, "ok"))
        resp = budget.openalex_get(client, "/works")
        # No HTTP call made.
        assert resp is None
        assert client.calls == []

    def test_budget_exhaustion_marks_circuit_and_returns_none(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        # Server responds 429 with the budget signal.
        client = _FakeClient(_FakeResponse(429, "Insufficient budget. Resets at midnight UTC."))
        resp = budget.openalex_get(client, "/works")
        # Wrapper recognizes the body, opens the circuit, and returns None.
        assert resp is None
        assert budget.openalex_ok() is False

    def test_other_429_does_not_open_circuit(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        client = _FakeClient(_FakeResponse(429, "Too many requests"))
        resp = budget.openalex_get(client, "/works")
        # Non-budget 429 is returned as-is; circuit stays closed.
        assert resp is client.response
        assert budget.openalex_ok() is True

    def test_endpoint_normalization(self, tmp_path, monkeypatch) -> None:
        _isolate_state(tmp_path, monkeypatch)
        client = _FakeClient(_FakeResponse(200))
        budget.openalex_get(client, "https://api.openalex.org/works/doi:10.1234/abc")
        budget.openalex_get(client, "https://api.openalex.org/works/doi:10.5555/xyz")
        # Both should aggregate to the same /works/doi:* row.
        rows = [
            r for r in metrics.summary(7) if r["source"] == "openalex" and "doi:" in r["endpoint"]
        ]
        assert len(rows) == 1
        assert rows[0]["window"] == 2
