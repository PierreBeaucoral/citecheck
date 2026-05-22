"""Per-provider LLM quota monitor for the citecheck web app.

Cerebras's free tier caps requests at ~30 RPM with a daily token budget
in the low millions.  HF's free Inference exhausts after a few dozen
requests per month.  Anthropic / OpenAI charge per token.  In every
case the deployed app needs to:

1. Know how much it has spent today.
2. Estimate how much a planned job will cost before launching it.
3. Refuse to start Phase 5 when the projected cost would exceed the
   remaining daily budget.
4. Inform the operator (via the /admin/health endpoint) of the live
   quota state, so they can decide to top up credits or wait for the
   reset.

The monitor is a thin SQLite-backed counter, keyed by (provider, UTC
date).  Each successful LLM call increments the row; the daily window
resets at 00:00 UTC.  Local accounting is best-effort: it overestimates
slightly when provider response headers expose more authoritative
numbers (planned for v1.1).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Provider-specific defaults.  These are conservative estimates; the
# operator can override via WebSettings.  Cerebras's documented daily
# token cap is "around 1M" on the free tier; we use a slightly lower
# threshold for the circuit-breaker so we never actually hit the API's
# hard cap.
_DEFAULT_DAILY_TOKEN_LIMIT = {
    "cerebras": 800_000,  # conservative vs ~1M docs
    "hf": 30_000,  # HF free is monthly-not-daily; we use a
    # daily-equivalent proxy here
    "ollama": 10_000_000,  # local Ollama: no real cap; nominal value
    "anthropic": 50_000_000,  # placeholder for v1.1 when credits land
}

# Per-call upper bound for cost estimation (input + output) at our prompt
# template + 5 retrieved chunks + ~300 output tokens.  Used by pre-flight
# estimate; the live monitor uses actual measured tokens.
ESTIMATED_TOKENS_PER_CALL = 4500


@dataclass(frozen=True)
class QuotaState:
    """Snapshot of the daily quota."""

    provider: str
    date_utc: str
    requests: int
    input_tokens: int
    output_tokens: int
    daily_limit: int
    remaining: int  # daily_limit - (input + output), floored at 0
    is_exhausted: bool


@dataclass(frozen=True)
class CostEstimate:
    """Pre-flight projection for a planned job."""

    n_claims: int
    estimated_tokens: int
    will_exhaust: bool
    remaining_before: int
    remaining_after: int
    safe_claim_count: int  # how many we can run before hitting the cap


_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_usage (
    provider     TEXT NOT NULL,
    date_utc     TEXT NOT NULL,        -- 'YYYY-MM-DD'
    requests     INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, date_utc)
);
"""


class QuotaMonitor:
    """SQLite-backed per-provider daily quota counter.

    One instance per process is fine; methods are short-connection and
    SQLite handles concurrency internally for our write rate (one update
    per LLM call).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        daily_token_limits: dict[str, int] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limits = dict(_DEFAULT_DAILY_TOKEN_LIMIT)
        if daily_token_limits:
            self.limits.update(daily_token_limits)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    # ---- write -----------------------------------------------------------

    def record(
        self,
        provider: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Increment today's counters for the given provider by one request."""
        date = self._today_utc()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO quota_usage (provider, date_utc, requests, input_tokens, output_tokens) "
                "VALUES (?, ?, 1, ?, ?) "
                "ON CONFLICT(provider, date_utc) DO UPDATE SET "
                "  requests = requests + 1, "
                "  input_tokens = input_tokens + excluded.input_tokens, "
                "  output_tokens = output_tokens + excluded.output_tokens",
                (provider, date, input_tokens, output_tokens),
            )

    # ---- read ------------------------------------------------------------

    def state(self, provider: str) -> QuotaState:
        """Return the current QuotaState for `provider` (today's window)."""
        date = self._today_utc()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM quota_usage WHERE provider = ? AND date_utc = ?",
                (provider, date),
            ).fetchone()
        reqs = int(row["requests"]) if row else 0
        in_tok = int(row["input_tokens"]) if row else 0
        out_tok = int(row["output_tokens"]) if row else 0
        limit = self.limits.get(provider, 1_000_000)
        used = in_tok + out_tok
        remaining = max(0, limit - used)
        return QuotaState(
            provider=provider,
            date_utc=date,
            requests=reqs,
            input_tokens=in_tok,
            output_tokens=out_tok,
            daily_limit=limit,
            remaining=remaining,
            is_exhausted=(remaining == 0),
        )

    def estimate(self, provider: str, n_claims: int) -> CostEstimate:
        """Project the cost of running `n_claims` LLM calls on `provider`."""
        st = self.state(provider)
        est_total = n_claims * ESTIMATED_TOKENS_PER_CALL
        safe = max(0, st.remaining // ESTIMATED_TOKENS_PER_CALL)
        return CostEstimate(
            n_claims=n_claims,
            estimated_tokens=est_total,
            will_exhaust=est_total > st.remaining,
            remaining_before=st.remaining,
            remaining_after=max(0, st.remaining - est_total),
            safe_claim_count=safe,
        )

    def to_health_dict(self, provider: str) -> dict[str, Any]:
        """Serialize current state for the /admin/health JSON endpoint."""
        st = self.state(provider)
        return {
            "provider": st.provider,
            "date_utc": st.date_utc,
            "requests": st.requests,
            "input_tokens": st.input_tokens,
            "output_tokens": st.output_tokens,
            "daily_limit": st.daily_limit,
            "remaining": st.remaining,
            "is_exhausted": st.is_exhausted,
        }
