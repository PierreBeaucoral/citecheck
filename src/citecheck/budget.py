"""OpenAlex daily-budget circuit-breaker.

OpenAlex's polite-pool moved to a daily-dollar budget in 2024-2025; once
exhausted, every request returns HTTP 429 with body "Insufficient budget"
until midnight UTC. This module keeps a single flag on disk recording the
expiry of the current circuit-open state so:

  1. Subsequent OpenAlex calls inside the same process skip the HTTP call
     entirely and return a synthetic 429 — avoids piling up retries against
     a known-empty bucket.
  2. The CLI / web report can warn the user "OpenAlex budget exhausted;
     running in Crossref-only fallback mode until HH:MM UTC".

The flag file lives at ~/.citecheck/openalex_exhausted_until.iso. Empty /
missing means the circuit is closed (OpenAlex calls allowed).
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def _state_path() -> Path:
    override = os.environ.get("CITECHECK_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".citecheck"
    return base / "openalex_exhausted_until.iso"


def _next_midnight_utc() -> datetime:
    now = datetime.now(UTC)
    tomorrow = (now + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=UTC)


def mark_exhausted(*, until: datetime | None = None) -> None:
    """Record that OpenAlex is over budget; circuit stays open until midnight UTC."""
    expiry = until or _next_midnight_utc()
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        path.write_text(expiry.isoformat(), encoding="utf-8")
    log.warning("openalex budget marked exhausted until %s", expiry.isoformat())


def exhausted_until() -> datetime | None:
    """Returns the expiry datetime if the circuit is open, otherwise None."""
    path = _state_path()
    if not path.is_file():
        return None
    try:
        expiry = datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    if now >= expiry:
        # Auto-reset on first read after expiry.
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    return expiry


def openalex_ok() -> bool:
    """Cheap check before making an OpenAlex call. False => circuit open, skip."""
    return exhausted_until() is None


def is_budget_exhausted_response(status_code: int, body_excerpt: str) -> bool:
    """Heuristic: OpenAlex 429 with 'Insufficient budget' is the budget signal."""
    return status_code == 429 and "insufficient budget" in (body_excerpt or "").lower()


def openalex_get(client, url: str, **params) -> object | None:
    """Polite wrapper around `client.get` for any OpenAlex endpoint.

    - Short-circuits to None when the circuit is already open (no HTTP made,
      caller falls back to Crossref-only behavior).
    - Records the request in `citecheck.metrics`.
    - Marks the circuit open if the response is a budget-exhaustion 429.

    Returns the `httpx.Response` on success or None when the call was skipped
    or the budget was just newly exhausted (in which case the caller should
    treat as if the call had failed).
    """
    # Import inside to avoid making metrics a hard dependency of budget's
    # type signature (and to dodge any future circular imports).
    import httpx

    from citecheck import metrics

    # Normalize for stable metric aggregation: strip the base URL and any
    # specific identifier so /works/doi:10.x/y and /works/doi:10.a/b both
    # roll up to /works/doi:*.
    raw_endpoint = url.split("?", 1)[0]
    endpoint = raw_endpoint
    for base in ("https://api.openalex.org", "http://api.openalex.org"):
        if endpoint.startswith(base):
            endpoint = endpoint[len(base) :]
            break
    # Collapse /works/doi:* and /authors/A123 to bucket forms.
    import re as _re

    endpoint = _re.sub(r"/doi:.+$", "/doi:*", endpoint)
    endpoint = _re.sub(r"/(W|A|S|I)\d+$", r"/\1*", endpoint)

    if not openalex_ok():
        metrics.record("openalex", endpoint, cache_hit=False, status=429)
        return None
    try:
        resp = client.get(url, params=params)
    except httpx.RequestError:
        metrics.record("openalex", endpoint, cache_hit=False, status=0)
        raise
    metrics.record("openalex", endpoint, cache_hit=False, status=resp.status_code)
    if is_budget_exhausted_response(resp.status_code, resp.text):
        mark_exhausted()
        # Treat THIS response as a circuit-open: caller falls back gracefully
        # instead of using whatever body the 429 came with.
        return None
    return resp
