"""Unit tests for citecheck.checks.cache."""

from __future__ import annotations

import time

import pytest

from citecheck.checks.cache import CacheStore


@pytest.fixture
def cache(tmp_path) -> CacheStore:
    return CacheStore(path=tmp_path / "cache.db")


class TestCacheBasics:
    def test_set_then_get(self, cache: CacheStore) -> None:
        cache.set("k", {"a": 1, "b": [2, 3]})
        assert cache.get("k") == {"a": 1, "b": [2, 3]}

    def test_missing_key_returns_none(self, cache: CacheStore) -> None:
        assert cache.get("nope") is None

    def test_set_replaces(self, cache: CacheStore) -> None:
        cache.set("k", "first")
        cache.set("k", "second")
        assert cache.get("k") == "second"

    def test_clear(self, cache: CacheStore) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_close_idempotent(self, cache: CacheStore) -> None:
        cache.close()
        cache.close()  # no error


class TestCacheTTL:
    def test_expired_entries_return_none(self, tmp_path) -> None:
        # 0-second TTL — anything we just set is already stale.
        cache = CacheStore(path=tmp_path / "c.db", ttl_s=0)
        cache.set("k", "value")
        time.sleep(0.01)
        assert cache.get("k") is None

    def test_expired_entries_are_evicted(self, tmp_path) -> None:
        cache = CacheStore(path=tmp_path / "c.db", ttl_s=0)
        cache.set("k", "value")
        time.sleep(0.01)
        # First read evicts the row; second read confirms it's gone.
        cache.get("k")
        # Re-set with a generous TTL and confirm it sticks.
        cache.ttl_s = 3600
        cache.set("k", "new")
        assert cache.get("k") == "new"


class TestCachePersistence:
    def test_value_survives_reopen(self, tmp_path) -> None:
        path = tmp_path / "persist.db"
        with CacheStore(path=path) as cache:
            cache.set("k", "kept")
        with CacheStore(path=path) as cache:
            assert cache.get("k") == "kept"


class TestCorruptPayload:
    def test_corrupt_json_is_dropped(self, tmp_path) -> None:
        cache = CacheStore(path=tmp_path / "c.db")
        # Manually insert malformed JSON.
        cache._conn.execute(
            "INSERT INTO cache (key, payload, fetched_at) VALUES (?, ?, ?)",
            ("k", "{not valid json", int(time.time())),
        )
        cache._conn.commit()
        assert cache.get("k") is None
        # Drop should be persisted — subsequent get is None for the same reason.
        assert cache.get("k") is None
