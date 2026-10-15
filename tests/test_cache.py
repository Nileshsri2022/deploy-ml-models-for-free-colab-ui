"""The resilience cache: TTL, stale-while-revalidate, pinning, eviction, durability.

These are the behaviours that let a published endpoint keep answering after the
Colab VM dies, so they are tested against both backends.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from gateway.core.cache import (
    CACHE_HIT,
    CACHE_MISS,
    CACHE_STALE,
    MemoryCache,
    SqliteCache,
    build_cache,
)

VALUE = {"output": {"label": "positive"}, "__request__": {"text": "great"}}


@pytest.fixture(params=["memory", "sqlite"])
def cache(request: pytest.FixtureRequest, tmp_path: Path):
    backend = build_cache(
        request.param, data_dir=tmp_path, max_entries=50, max_bytes=1 << 20, persist=False
    )
    yield backend
    backend.close()


def test_set_then_get_is_a_hit(cache) -> None:
    cache.set("k1", VALUE, endpoint_id="ep1", model_version=1, ttl_s=60, swr_s=60)
    lookup = cache.get("k1")
    assert lookup.status == CACHE_HIT
    assert lookup.value == VALUE
    assert lookup.endpoint_id == "ep1"


def test_unknown_key_is_a_miss(cache) -> None:
    assert cache.get("nope").status == CACHE_MISS


def test_expired_entry_becomes_stale_then_disappears(cache) -> None:
    cache.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=0.05, swr_s=0.2)
    assert cache.get("k").status == CACHE_HIT
    time.sleep(0.08)
    lookup = cache.get("k")
    assert lookup.status == CACHE_STALE, "inside the swr window we must still serve"
    assert lookup.stale_for_s > 0
    time.sleep(0.25)
    assert cache.get("k").status == CACHE_MISS


def test_zero_ttl_with_swr_is_immediately_stale(cache) -> None:
    """The offline-serving configuration: never fresh, always available."""
    cache.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=0, swr_s=3600)
    assert cache.get("k").status == CACHE_STALE


def test_peek_does_not_count_a_hit(cache) -> None:
    cache.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=60)
    before = cache.stats().hits
    assert cache.peek("k").status == CACHE_HIT
    assert cache.stats().hits == before
    cache.get("k")
    assert cache.stats().hits == before + 1


def test_hit_ratio_and_counters(cache) -> None:
    cache.set("a", VALUE, endpoint_id="ep", model_version=1, ttl_s=60)
    cache.get("a")
    cache.get("a")
    cache.get("missing")
    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.hit_ratio == pytest.approx(2 / 3, abs=0.01)


def test_lru_eviction_respects_the_entry_cap(cache) -> None:
    for i in range(80):
        cache.set(f"k{i}", {"output": i}, endpoint_id="ep", model_version=1, ttl_s=60)
    stats = cache.stats()
    assert stats.entries <= 50
    assert stats.evictions > 0
    # The most recently written key survives.
    assert cache.get("k79").status == CACHE_HIT


def test_pinned_entries_survive_eviction(cache) -> None:
    cache.set("hot", VALUE, endpoint_id="ep", model_version=1, ttl_s=600, pinned=True)
    for i in range(120):
        cache.set(f"k{i}", {"output": i}, endpoint_id="ep", model_version=1, ttl_s=60)
    assert cache.get("hot").status == CACHE_HIT
    assert cache.stats().pinned == 1


def test_pin_and_unpin(cache) -> None:
    cache.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=60)
    assert cache.pin("k", True) is True
    assert cache.pin("missing", True) is False
    assert cache.stats().pinned == 1
    cache.pin("k", False)
    assert cache.stats().pinned == 0


def test_invalidate_endpoint_only_touches_that_endpoint(cache) -> None:
    cache.set("a", VALUE, endpoint_id="ep1", model_version=1, ttl_s=60)
    cache.set("b", VALUE, endpoint_id="ep2", model_version=1, ttl_s=60)
    assert cache.invalidate_endpoint("ep1") == 1
    assert cache.get("a").status == CACHE_MISS
    assert cache.get("b").status == CACHE_HIT


def test_invalidate_keeps_pinned_entries(cache) -> None:
    """A pinned hot key is the offline answer - invalidation must not remove it."""
    cache.set("a", VALUE, endpoint_id="ep1", model_version=1, ttl_s=60, pinned=True)
    assert cache.invalidate_endpoint("ep1") == 0
    assert cache.get("a").status == CACHE_HIT


def test_overwrite_refreshes_ttl_and_keeps_pin(cache) -> None:
    cache.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=60, pinned=True)
    cache.set("k", {"output": "new"}, endpoint_id="ep", model_version=2, ttl_s=0.05, swr_s=0.2)
    assert cache.peek("k").value == {"output": "new"}
    assert cache.stats().pinned == 1, "pinning is sticky across writes"


def test_hot_keys_are_ordered_by_hits(cache) -> None:
    for name in ("cold", "warm", "hot"):
        cache.set(name, {"output": name, "__request__": {"text": name}},
                  endpoint_id="ep", model_version=1, ttl_s=60)
    for _ in range(3):
        cache.get("hot")
    cache.get("warm")
    keys = [k["key"] for k in cache.hot_keys(limit=3)]
    assert keys[0] == "hot"
    assert keys[1] == "warm"
    assert cache.hot_keys(limit=1)[0]["sample"] == {"text": "hot"}


def test_hot_keys_can_be_scoped_to_an_endpoint(cache) -> None:
    cache.set("a", {"output": 1, "__request__": {"x": 1}}, endpoint_id="ep1", model_version=1, ttl_s=60)
    cache.set("b", {"output": 2, "__request__": {"x": 2}}, endpoint_id="ep2", model_version=1, ttl_s=60)
    keys = {k["key"] for k in cache.hot_keys(endpoint_id="ep2", limit=10)}
    assert keys == {"b"}


def test_entries_listing_reports_metadata(cache) -> None:
    cache.set("k", VALUE, endpoint_id="ep", model_version=7, ttl_s=60, swr_s=30)
    entries = cache.entries(limit=10)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.key == "k"
    assert entry.model_version == 7
    assert entry.size_bytes > 0
    assert entry.preview is not None


def test_memory_cache_snapshot_survives_a_restart(tmp_path: Path) -> None:
    """The state-retention guarantee: a gateway restart must not empty the cache."""
    path = tmp_path / "cache.json"
    first = MemoryCache(max_entries=100, persist_path=path)
    first.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=3600, swr_s=3600)
    first.get("k")
    assert first.save() is True
    first.close()

    second = MemoryCache(max_entries=100, persist_path=path)
    lookup = second.get("k")
    assert lookup.status == CACHE_HIT
    assert lookup.value == VALUE
    assert second.stats().hits >= 1, "counters are restored too"


def test_snapshot_drops_entries_that_died_while_we_were_down(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    first = MemoryCache(max_entries=100, persist_path=path)
    first.set("short", VALUE, endpoint_id="ep", model_version=1, ttl_s=0.01, swr_s=0.01)
    first.set("long", VALUE, endpoint_id="ep", model_version=1, ttl_s=3600, swr_s=0)
    first.save()
    time.sleep(0.05)

    second = MemoryCache(max_entries=100, persist_path=path)
    assert second.get("short").status == CACHE_MISS
    assert second.get("long").status == CACHE_HIT


def test_sqlite_cache_persists_without_a_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    first = SqliteCache(path, max_entries=10)
    first.set("k", VALUE, endpoint_id="ep", model_version=1, ttl_s=3600)
    first.close()

    second = SqliteCache(path, max_entries=10)
    assert second.get("k").status == CACHE_HIT
    assert second.stats().entries == 1
    second.close()


def test_sqlite_cache_evicts_least_hit_first(tmp_path: Path) -> None:
    cache = SqliteCache(tmp_path / "c.sqlite3", max_entries=5)
    for i in range(12):
        cache.set(f"k{i}", {"output": i}, endpoint_id="ep", model_version=1, ttl_s=60)
    cache.get("k11")
    cache.get("k11")
    assert cache.stats().entries <= 5
    assert cache.get("k11").status == CACHE_HIT
    cache.close()


def test_build_cache_honours_the_backend_name(tmp_path: Path) -> None:
    assert build_cache("memory", data_dir=tmp_path).name == "memory"
    assert build_cache("sqlite", data_dir=tmp_path).name == "sqlite"
    assert build_cache("nonsense", data_dir=tmp_path).name == "memory", "unknown falls back safely"


def test_large_values_are_size_accounted(cache) -> None:
    big = {"output": "x" * 100_000}
    cache.set("big", big, endpoint_id="ep", model_version=1, ttl_s=60)
    stats = cache.stats()
    assert stats.size_bytes >= 100_000
    assert cache.get("big").status == CACHE_HIT
