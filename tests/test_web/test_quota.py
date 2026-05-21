"""QuotaMonitor unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from citecheck.web.quota import QuotaMonitor


@pytest.fixture
def quota(tmp_path: Path) -> QuotaMonitor:
    return QuotaMonitor(
        tmp_path / "quota.sqlite3",
        daily_token_limits={"cerebras": 10_000, "ollama": 1_000_000},
    )


def test_fresh_quota_has_zero_usage(quota: QuotaMonitor) -> None:
    state = quota.state("cerebras")
    assert state.requests == 0
    assert state.input_tokens == 0
    assert state.output_tokens == 0
    assert state.remaining == 10_000
    assert state.is_exhausted is False


def test_record_increments(quota: QuotaMonitor) -> None:
    quota.record("cerebras", input_tokens=3000, output_tokens=500)
    quota.record("cerebras", input_tokens=4000, output_tokens=600)
    state = quota.state("cerebras")
    assert state.requests == 2
    assert state.input_tokens == 7000
    assert state.output_tokens == 1100
    assert state.remaining == 10_000 - 7000 - 1100


def test_exhausted_when_remaining_zero(quota: QuotaMonitor) -> None:
    quota.record("cerebras", input_tokens=8000, output_tokens=2000)
    state = quota.state("cerebras")
    assert state.remaining == 0
    assert state.is_exhausted is True


def test_estimate_safe_count(quota: QuotaMonitor) -> None:
    # 10K budget; ~4500 tokens per call; 2 claims fit cleanly.
    est = quota.estimate("cerebras", n_claims=5)
    assert est.n_claims == 5
    assert est.will_exhaust is True
    assert est.safe_claim_count == 2  # 10000 // 4500


def test_estimate_below_budget(quota: QuotaMonitor) -> None:
    est = quota.estimate("cerebras", n_claims=2)
    assert est.will_exhaust is False
    assert est.remaining_after > 0


def test_providers_are_isolated(quota: QuotaMonitor) -> None:
    quota.record("cerebras", input_tokens=8000, output_tokens=2000)
    cere = quota.state("cerebras")
    olla = quota.state("ollama")
    assert cere.is_exhausted is True
    assert olla.is_exhausted is False
    assert olla.requests == 0


def test_health_dict_round_trip(quota: QuotaMonitor) -> None:
    quota.record("cerebras", input_tokens=1500, output_tokens=300)
    d = quota.to_health_dict("cerebras")
    assert d["provider"] == "cerebras"
    assert d["requests"] == 1
    assert d["input_tokens"] == 1500
    assert d["output_tokens"] == 300
    assert d["is_exhausted"] is False
