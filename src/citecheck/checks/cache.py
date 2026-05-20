"""Tiny SQLite key/value cache with TTL.

Used to memoize Crossref and OpenAlex lookups so re-running `citecheck check`
on the same paper does not hammer either API. Default location is
~/.citecheck/cache.db; tests use an in-memory path.

We keep the stdlib sqlite3 driver — no ORM, no extra dependency. Schema:

    cache(key TEXT PRIMARY KEY, payload TEXT, fetched_at INTEGER)

`fetched_at` is unix-epoch seconds; `payload` is a JSON-encoded value.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TTL_S = 7 * 24 * 3600  # 7 days


def default_path() -> Path:
    """Where the cache lives by default. Honors $CITECHECK_CACHE_DIR for tests/CI."""
    override = os.environ.get("CITECHECK_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".citecheck"
    return base / "cache.db"


class CacheStore:
    """Thin SQLite-backed cache. Safe to use as a context manager; close() is idempotent."""

    def __init__(self, path: str | Path | None = None, *, ttl_s: int = DEFAULT_TTL_S) -> None:
        self.path = Path(path) if path else default_path()
        self.ttl_s = ttl_s
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "key TEXT PRIMARY KEY, "
            "payload TEXT NOT NULL, "
            "fetched_at INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        """Return the cached value for key, or None if missing/expired."""
        row = self._conn.execute(
            "SELECT payload, fetched_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        payload, fetched_at = row
        if time.time() - fetched_at > self.ttl_s:
            # Expired — let the caller refetch. Drop the stale row opportunistically.
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            log.warning("cache: dropping corrupt payload for key %r", key)
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None

    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, payload, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, ensure_ascii=False), int(time.time())),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Wipe the cache. Useful in tests and for manual recovery."""
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def close(self) -> None:
        with contextlib.suppress(sqlite3.ProgrammingError):
            self._conn.close()  # idempotent — second close raises ProgrammingError

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
