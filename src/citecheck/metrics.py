"""Per-day API call counter for citecheck.

Records every Crossref / OpenAlex / GROBID HTTP response into a local SQLite
table so an operator can see how much of each provider's daily budget the
tool is consuming and how often the cache is saving the call.

Schema (single table, one row per (date, source, endpoint, cache_hit, status)):

    api_calls(
        date TEXT,         -- 'YYYY-MM-DD' in UTC
        source TEXT,       -- 'crossref' | 'openalex' | 'grobid'
        endpoint TEXT,     -- e.g. '/works', '/authors', '/sources'
        cache_hit INTEGER, -- 0 / 1
        status INTEGER,    -- HTTP status; 0 for transport errors
        count INTEGER
    )

The recorder is best-effort: any exception in the metrics layer is swallowed
to avoid taking down a working check just because the metrics DB is locked.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


def _default_path() -> Path:
    override = os.environ.get("CITECHECK_METRICS_DIR") or os.environ.get("CITECHECK_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".citecheck"
    return base / "metrics.db"


_CONN: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection | None:
    """Lazy-init the metrics connection. Returns None on failure (we never crash)."""
    global _CONN
    if _CONN is not None:
        return _CONN
    try:
        path = _default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(path, check_same_thread=False)
        _CONN.execute(
            "CREATE TABLE IF NOT EXISTS api_calls ("
            "date TEXT, source TEXT, endpoint TEXT, "
            "cache_hit INTEGER, status INTEGER, "
            "count INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY (date, source, endpoint, cache_hit, status))"
        )
        _CONN.commit()
    except (sqlite3.Error, OSError) as exc:
        # Any failure to open the metrics DB is best-effort: don't break the
        # caller. Filesystem errors (read-only path) and sqlite3 errors both
        # land us in the same "metrics unavailable" state.
        log.warning("metrics: cannot open metrics.db: %s", exc)
        _CONN = None
    return _CONN


def record(source: str, endpoint: str, *, cache_hit: bool, status: int = 0) -> None:
    """Bump the counter for one API call. Never raises."""
    conn = _get_conn()
    if conn is None:
        return
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    with contextlib.suppress(sqlite3.Error):
        conn.execute(
            "INSERT INTO api_calls (date, source, endpoint, cache_hit, status, count) "
            "VALUES (?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(date, source, endpoint, cache_hit, status) "
            "DO UPDATE SET count = count + 1",
            (today, source, endpoint, int(cache_hit), status),
        )
        conn.commit()


def summary(days: int = 7) -> list[dict]:
    """Return the last `days` of metrics aggregated by source/endpoint.

    Each row has: source, endpoint, calls_today, calls_7d, cache_hit_pct.
    """
    conn = _get_conn()
    if conn is None:
        return []
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT source, endpoint, "
        "  SUM(CASE WHEN date = ? THEN count ELSE 0 END) AS today, "
        "  SUM(count) AS total, "
        "  SUM(CASE WHEN cache_hit = 1 THEN count ELSE 0 END) AS hits, "
        "  SUM(count) AS all_count "
        "FROM api_calls "
        "WHERE date >= date('now', ?) "
        "GROUP BY source, endpoint "
        "ORDER BY total DESC",
        (today, f"-{days} days"),
    ).fetchall()
    out: list[dict] = []
    for src, ep, today_n, total, hits, all_n in rows:
        out.append(
            {
                "source": src,
                "endpoint": ep,
                "today": today_n or 0,
                "window": total or 0,
                "cache_hit_pct": (hits / all_n) if all_n else 0.0,
            }
        )
    return out
