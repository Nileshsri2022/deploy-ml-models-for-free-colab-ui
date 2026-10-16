"""The Redis registry mirror - free-tier durability without a paid disk.

Every test runs against fakeredis, so the suite needs no live Redis. Two
stores sharing one FakeServer simulate a gateway restart; a fresh temp dir
for the second store proves data came from Redis and not from disk.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

fakeredis = pytest.importorskip("fakeredis")

from gateway.core.redis_store import (  # noqa: E402
    RedisSnapshotStore,
    build_registry_store,
    describe_url,
    registry_status,
)
from gateway.core.repository import Registry  # noqa: E402
from gateway.core.store import JsonStore  # noqa: E402


class _DeadClient:
    """A client that fails the way an unreachable Redis does."""

    def __getattr__(self, name: str) -> Any:
        def _raise(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError("redis down")

        return _raise


@pytest.fixture()
def server() -> Any:
    return fakeredis.FakeServer()


@pytest.fixture()
def fake_client(server: Any) -> Any:
    return fakeredis.FakeRedis(server=server, decode_responses=True)


def make_store(
    root: Path, server: Any, *, sync_interval_s: float = 3600.0, client: Any | None = None
) -> RedisSnapshotStore:
    """sync_interval defaults to 1h so nothing syncs unless a test says so."""
    return RedisSnapshotStore(
        root,
        redis_url="redis://fake",
        sync_interval_s=sync_interval_s,
        client=client if client is not None else fakeredis.FakeRedis(server=server, decode_responses=True),
    )


# --------------------------------------------------------------------------- #
# mirror + restore
# --------------------------------------------------------------------------- #
def test_records_survive_a_fresh_disk_via_redis(tmp_path: Path, server: Any) -> None:
    first = make_store(tmp_path / "a", server)
    first.put("models", "m1", {"id": "m1", "name": "clf", "version": 2})
    first.close()

    # A brand-new data dir: the only source of truth left is Redis.
    second = make_store(tmp_path / "b", server)
    assert second.get("models", "m1") == {"id": "m1", "name": "clf", "version": 2}
    assert second.count("models") == 1
    second.close()


def test_restore_materializes_the_local_file(tmp_path: Path, server: Any) -> None:
    first = make_store(tmp_path / "a", server)
    first.put("endpoints", "e1", {"id": "e1", "route": "/score"})
    first.close()

    fresh_dir = tmp_path / "b"
    second = make_store(fresh_dir, server)
    assert second.get("endpoints", "e1") is not None
    assert (fresh_dir / "endpoints.json").exists()  # disk keeps parity
    second.close()


def test_deletes_propagate_across_restarts(tmp_path: Path, server: Any) -> None:
    first = make_store(tmp_path / "a", server)
    first.put("jobs", "j1", {"id": "j1"})
    first.close()

    second = make_store(tmp_path / "b", server)
    assert second.delete("jobs", "j1") is True
    second.close()

    third = make_store(tmp_path / "c", server)
    assert third.get("jobs", "j1") is None
    third.close()


def test_warm_pulls_everything_at_boot(tmp_path: Path, server: Any) -> None:
    first = make_store(tmp_path / "a", server)
    first.put("models", "m1", {"id": "m1"})
    first.put("runtimes", "r1", {"id": "r1"})
    first.put("audit", "x1", {"id": "x1"})
    first.close()

    second = make_store(tmp_path / "b", server)
    assert second.warm(["models", "runtimes", "audit", "inference_usage"]) == 3
    assert second.warm(["models"]) == 0  # already cached: no double fetch
    assert second.count("inference_usage") == 0
    second.close()


def test_graceful_close_flushes_pending_writes(tmp_path: Path, server: Any) -> None:
    store = make_store(tmp_path, server)  # sync interval 1h: nothing synced yet
    store.put("models", "m1", {"id": "m1"})
    keys = list(fakeredis.FakeRedis(server=server, decode_responses=True).keys("*"))
    assert not [k for k in keys if ":models:m1" in k]  # still queued
    store.close()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    assert client.get("cdp:registry:v1:models:m1") == '{"id":"m1"}'


def test_background_sync_coalesces_bursts(tmp_path: Path, server: Any) -> None:
    store = make_store(tmp_path, server, sync_interval_s=0.2)
    for i in range(10):
        store.put("runtimes", "r1", {"id": "r1", "beat": i})
    time.sleep(0.6)  # one sync cycle
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    import json as _json

    stored = _json.loads(client.get("cdp:registry:v1:runtimes:r1"))
    assert stored["beat"] == 9
    store.close()


def test_unreachable_redis_degrades_to_file_store(tmp_path: Path) -> None:
    store = RedisSnapshotStore(
        tmp_path, redis_url="redis://fake", sync_interval_s=3600.0, client=_DeadClient()
    )
    store.put("models", "m1", {"id": "m1"})  # must not raise
    store.close()

    reopened: JsonStore = JsonStore(tmp_path)  # plain store reads the same files
    assert reopened.get("models", "m1") == {"id": "m1"}


def test_concurrent_mutations_all_reach_redis(tmp_path: Path, server: Any) -> None:
    store = make_store(tmp_path, server, sync_interval_s=0.1)

    def write(n: int) -> None:
        for i in range(20):
            store.put("api_keys", f"k{n}-{i}", {"id": f"k{n}-{i}"})

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    store.close()

    reader = fakeredis.FakeRedis(server=server, decode_responses=True)
    keys = [k for k in reader.keys("cdp:registry:v1:api_keys:*") if "__meta__" not in k]
    assert len(keys) == 80


# --------------------------------------------------------------------------- #
# repositories on top of the mirror
# --------------------------------------------------------------------------- #
def test_registry_repositories_over_the_mirror(tmp_path: Path, server: Any) -> None:
    from colabdeploy_shared.schemas import ArtifactSpec, ModelInfo

    store = make_store(tmp_path / "a", server)
    registry = Registry(store)
    registry.models.put(
        ModelInfo(id="mdl_1", name="iris", version=1, artifact=ArtifactSpec(uri="hf://x/y"))
    )
    store.close()

    second_store = make_store(tmp_path / "b", server)
    second = Registry(second_store)
    model = second.models.get("mdl_1")
    assert model is not None and model.name == "iris"
    assert second.models.find(name="iris")[0].version == 1
    second_store.close()


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def test_factory_defaults_to_file_store(data_dir: Path) -> None:
    from gateway.core.config import reset_settings

    settings = reset_settings(data_dir=data_dir)
    assert isinstance(build_registry_store(settings), JsonStore)


def test_factory_falls_back_without_a_url(data_dir: Path) -> None:
    from gateway.core.config import reset_settings

    settings = reset_settings(data_dir=data_dir, registry_backend="redis", redis_url="")
    assert isinstance(build_registry_store(settings), JsonStore)


def test_factory_falls_back_when_redis_lib_missing(data_dir: Path, monkeypatch: Any) -> None:
    from gateway.core.config import reset_settings

    settings = reset_settings(
        data_dir=data_dir, registry_backend="redis", redis_url="redis://fake"
    )
    monkeypatch.setitem(sys.modules, "redis", None)  # `import redis` now raises
    store = build_registry_store(settings)
    assert isinstance(store, JsonStore) and not isinstance(store, RedisSnapshotStore)


def test_factory_builds_mirror_with_a_url(data_dir: Path) -> None:
    from gateway.core.config import reset_settings

    settings = reset_settings(
        data_dir=data_dir, registry_backend="redis", redis_url="redis://fake"
    )
    store = build_registry_store(settings)
    assert isinstance(store, RedisSnapshotStore)
    store.close()  # never connected: _ensure_client degrades quietly


def test_inspect_lists_upstash_keys(tmp_path: Path, server: Any) -> None:
    store = make_store(tmp_path / "a", server, sync_interval_s=0)
    store.put("models", "m1", {"id": "m1", "name": "clf"})
    store.put("endpoints", "e1", {"id": "e1", "route": "/score"})
    view = store.inspect(limit=50)
    assert view["connected"] is True
    assert view["backend"] == "upstash"
    ids = {(row["collection"], row["id"]) for row in view["keys"] if row["kind"] == "record"}
    assert ("models", "m1") in ids
    assert ("endpoints", "e1") in ids
    one = store.inspect_key(store._key("models", "m1"))
    assert one is not None and one["value"]["id"] == "m1"
    assert store.drop_key(store._key("models", "m1")) is True
    assert store.inspect_key(store._key("models", "m1")) is None
    store.close()


# --------------------------------------------------------------------------- #
# connection lifecycle (the non-injected path, i.e. what a real deploy takes)
# --------------------------------------------------------------------------- #
class _RecordingClient:
    """Stands in for a real redis-py client and counts how it was used."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.closed = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture()
def live_redis(monkeypatch: pytest.MonkeyPatch, server: Any) -> dict[str, Any]:
    """Patch ``redis.Redis.from_url`` so the store builds its own client.

    Returns a mutable counter dict: ``opens`` counts sockets created (a leak
    shows up as one per call) and ``fail`` makes the next attempts raise, which
    is how a cold Upstash database or a blocked egress path looks.
    """
    state: dict[str, Any] = {"opens": 0, "fail": None}

    def _from_url(url: str, **kwargs: Any) -> Any:
        state["opens"] += 1
        if state["fail"] is not None:
            raise state["fail"]
        return _RecordingClient(fakeredis.FakeRedis(server=server, decode_responses=True))

    import redis as redis_module

    monkeypatch.setattr(redis_module.Redis, "from_url", staticmethod(_from_url))
    return state


def _store(tmp_path: Path, **kw: Any) -> RedisSnapshotStore:
    return RedisSnapshotStore(tmp_path / "store", redis_url="rediss://default:t@db.upstash.io:6379", **kw)


def test_connected_client_is_reused_not_reopened(tmp_path: Path, live_redis: dict[str, Any]) -> None:
    """One socket, cached on the store, and every inspector call sees it.

    Regression: the client was returned but never cached, so the 30 s reconnect
    backoff was armed by *successful* connects. The background syncer re-armed
    it every window and the Cache page answered "not connected" forever while
    the mirror was actually healthy.
    """
    store = _store(tmp_path, sync_interval_s=30.0)
    store.put("models", "m1", {"id": "m1"})
    for _ in range(4):
        assert store.inspect()["connected"] is True
    assert live_redis["opens"] == 1  # not one per call
    assert store.status()["connected"] is True
    store.close()


def test_repeated_reads_do_not_leak_connections(tmp_path: Path, live_redis: dict[str, Any]) -> None:
    store = _store(tmp_path, sync_interval_s=0)
    for i in range(10):
        store.put("models", f"m{i}", {"id": f"m{i}"})
        store.inspect(limit=5)
    assert live_redis["opens"] == 1
    store.close()


def test_warm_restores_every_collection_at_boot(tmp_path: Path, server: Any, live_redis: dict[str, Any]) -> None:
    """The whole point of the mirror: a cold container rebuilds *all* state."""
    first = make_store(tmp_path / "a", server, sync_interval_s=0)
    first.put("models", "m1", {"id": "m1"})
    first.put("endpoints", "e1", {"id": "e1"})
    first.put("api_keys", "k1", {"id": "k1"})
    first.close()

    second = _store(tmp_path / "b", sync_interval_s=30.0)
    assert second.warm(["models", "endpoints", "api_keys", "audit"]) == 3
    assert second.count("models") == 1 and second.count("audit") == 0
    assert second.status()["restored"] == 3
    second.close()


def test_connect_failure_records_the_error_then_backs_off(
    tmp_path: Path, live_redis: dict[str, Any]
) -> None:
    live_redis["fail"] = ConnectionError("Error 110 connecting to Redis")
    store = _store(tmp_path, sync_interval_s=30.0)
    view = store.inspect()
    assert view["connected"] is False
    assert "Error 110" in view["last_error"]
    assert view["message"]  # a reason, not the generic "set the env vars" text
    assert view["hint"]
    opens = live_redis["opens"]
    store.inspect()  # inside the backoff window: no new probe, still offline
    assert live_redis["opens"] == opens
    assert store.status()["state"] in {"backoff", "unreachable"}
    store.close()


def test_reconnect_clears_the_backoff(tmp_path: Path, live_redis: dict[str, Any]) -> None:
    live_redis["fail"] = ConnectionError("redis down")
    store = _store(tmp_path, sync_interval_s=30.0)
    assert store.status()["connected"] is False
    live_redis["fail"] = None  # the operator woke the Upstash database
    status = store.reconnect()
    assert status["connected"] is True
    assert status["last_error"] == ""
    store.close()


def test_write_through_mode_catches_up_after_a_failed_flush(
    tmp_path: Path, live_redis: dict[str, Any]
) -> None:
    """``CDP_REGISTRY_SYNC_S=0`` flushes inline, so a failed flush needs a retry.

    Without the catch-up loop those keys would sit queued until the *next*
    mutation — and a quiet gateway would never make one.
    """
    live_redis["fail"] = ConnectionError("redis down")
    store = RedisSnapshotStore(
        tmp_path / "store",
        redis_url="rediss://default:t@db.upstash.io:6379",
        sync_interval_s=0,
        connect_backoff_s=0.2,
    )
    store.put("models", "m1", {"id": "m1"})
    assert store.sync_stats()["pending"] == 1  # the inline flush could not run

    live_redis["fail"] = None  # the database came back
    for _ in range(50):
        if store.sync_stats()["pending"] == 0:
            break
        time.sleep(0.05)
    assert store.sync_stats()["pending"] == 0
    assert store._client.get("cdp:registry:v1:models:m1") == '{"id":"m1"}'
    store.close()


def test_close_releases_the_socket(tmp_path: Path, live_redis: dict[str, Any]) -> None:
    store = _store(tmp_path, sync_interval_s=30.0)
    assert store.inspect()["connected"] is True
    client = store._client
    store.close()
    assert client.closed == 1
    assert store._client is None


def test_flush_reports_an_unreachable_mirror(tmp_path: Path, live_redis: dict[str, Any]) -> None:
    """"Synced" must mean the write landed, not that a no-op ran."""
    live_redis["fail"] = ConnectionError("Connection refused")
    store = _store(tmp_path, sync_interval_s=30.0)
    store.put("models", "m1", {"id": "m1"})
    result = store.flush()
    assert result["synced"] is False
    assert result["stats"]["pending"] == 1
    assert result["message"] and result["hint"]


def test_keepalive_uses_the_cached_client(tmp_path: Path, live_redis: dict[str, Any]) -> None:
    store = _store(tmp_path, sync_interval_s=30.0, keepalive_s=1.0)
    assert store._ensure_client() is not None
    store._last_keepalive = 0.0
    store._maybe_keepalive()
    assert store._client.get("cdp:registry:v1:__keepalive__") is not None
    store.close()


# --------------------------------------------------------------------------- #
# URL sanity + why the mirror was skipped
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("url", "problem"),
    [
        ("rediss://default:t@db.upstash.io:6379", ""),
        ("redis://localhost:6379/0", ""),
        ("", "empty"),
        ("https://default:t@db.upstash.io", "REST"),
        ("redis://default:t@db.upstash.io:6379", "TLS"),
        ('"rediss://default:t@db.upstash.io:6379"', "quote"),
        ("db.upstash.io:6379", "not a URL"),
    ],
)
def test_describe_url(url: str, problem: str) -> None:
    out = describe_url(url)
    assert (problem in out["problem"]) is True, out
    if not problem:
        assert out["scheme_ok"] and out["host"]


def test_factory_records_why_it_skipped_the_mirror(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason has to survive the fallback, or the UI can only guess.

    Each case is a real deployment mistake: backend left on `file`, a secret
    that never reached the process, the REST URL pasted instead of the Redis
    one, and an image built without the `[redis]` extra.
    """
    from gateway.core.config import reset_settings

    settings = reset_settings(data_dir=data_dir)
    status = registry_status(build_registry_store(settings), settings)
    assert status["reason"] == "backend_file" and "CDP_REGISTRY_BACKEND" in status["message"]

    settings = reset_settings(data_dir=data_dir, registry_backend="redis", redis_url="")
    status = registry_status(build_registry_store(settings), settings)
    assert status["reason"] == "no_url" and "Secret" in status["hint"]

    settings = reset_settings(data_dir=data_dir, registry_backend="redis", redis_url="https://x.upstash.io")
    status = registry_status(build_registry_store(settings), settings)
    assert status["reason"] == "bad_url" and "REST" in status["hint"]

    settings = reset_settings(data_dir=data_dir, registry_backend="redis", redis_url="redis://fake")
    monkeypatch.setitem(sys.modules, "redis", None)  # `import redis` now raises
    status = registry_status(build_registry_store(settings), settings)
    assert status["reason"] == "library_missing" and "[redis]" in status["hint"]
    reset_settings(data_dir=data_dir)


def test_settings_strip_quotes_from_pasted_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.core.config import reset_settings

    monkeypatch.setenv(
        "CDP_REDIS_URL", '  "rediss://default:t@db.upstash.io:6379"  '
    )
    monkeypatch.setenv("CDP_REGISTRY_BACKEND", " redis ")
    settings = reset_settings(data_dir=tmp_path)
    assert settings.redis_url == "rediss://default:t@db.upstash.io:6379"
    assert settings.registry_backend == "redis"
    store = build_registry_store(settings)
    assert isinstance(store, RedisSnapshotStore)
    assert store.url_ok is True
    store.close()
