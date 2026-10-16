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

from gateway.core.redis_store import RedisSnapshotStore, build_registry_store  # noqa: E402
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
