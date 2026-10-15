"""The JSON document store and typed repositories - the system of record.

The control plane has to survive being killed mid-write (free-tier hosts get
recycled), so atomicity and read-modify-write correctness are tested explicitly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from colabdeploy_shared.enums import AdapterKind, DeploymentState
from colabdeploy_shared.schemas import (
    ArtifactSpec,
    DeploymentInfo,
    EndpointCreate,
    EndpointInfo,
    ModelCreate,
    ModelInfo,
    ModelSchema,
)
from gateway.core.repository import Registry
from gateway.core.store import JsonStore


# --------------------------------------------------------------------------- #
# JsonStore
# --------------------------------------------------------------------------- #
def test_put_get_round_trip(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.put("things", "a", {"id": "a", "n": 1})
    assert store.get("things", "a") == {"id": "a", "n": 1}
    assert store.count("things") == 1


def test_writes_hit_the_disk_immediately(tmp_path: Path) -> None:
    """A crash after ``put`` must not lose the record."""
    store = JsonStore(tmp_path)
    store.put("things", "a", {"id": "a"})
    del store
    reopened = JsonStore(tmp_path)
    assert reopened.get("things", "a") == {"id": "a"}


def test_no_tmp_files_are_left_behind(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    for i in range(20):
        store.put("things", f"k{i}", {"id": i})
    assert not list(tmp_path.glob("*.tmp"))


def test_revision_increments_on_every_mutation(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    before = store.revision
    store.put("things", "a", {"id": "a"})
    assert store.revision > before
    after_put = store.revision
    store.update("things", "a", {"n": 1})
    assert store.revision > after_put
    before_delete = store.revision
    store.delete("things", "a")
    assert store.revision > before_delete


def test_update_supports_a_callable_read_modify_write(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.put("counters", "hits", {"n": 1})
    store.update("counters", "hits", lambda rec: {**rec, "n": rec["n"] + 41})
    assert store.get("counters", "hits")["n"] == 42


def test_update_of_a_missing_key_is_a_no_op(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    assert store.update("things", "nope", {"a": 1}) is None
    assert store.count("things") == 0


def test_delete_reports_whether_it_found_anything(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.put("things", "a", {"id": "a"})
    assert store.delete("things", "a") is True
    assert store.delete("things", "a") is False


def test_find_and_find_one(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.put("m", "1", {"id": "1", "name": "x", "v": 1})
    store.put("m", "2", {"id": "2", "name": "x", "v": 2})
    store.put("m", "3", {"id": "3", "name": "y", "v": 1})
    assert len(store.find("m", name="x")) == 2
    assert store.find_one("m", name="y")["id"] == "3"
    assert store.find_one("m", name="z") is None


def test_corrupt_file_is_quarantined_not_fatal(tmp_path: Path) -> None:
    """A truncated write from a previous crash must not take the whole system down."""
    (tmp_path / "things.json").write_text("{not valid json", encoding="utf-8")
    store = JsonStore(tmp_path)
    assert store.all("things") == []
    store.put("things", "a", {"id": "a"})
    assert store.get("things", "a") == {"id": "a"}
    assert list(tmp_path.glob("things.corrupt-*.json")), "the bad file is kept for forensics"


def test_concurrent_writers_do_not_lose_records(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)

    def writer(worker: int) -> None:
        for i in range(40):
            store.put("things", f"{worker}-{i}", {"worker": worker, "i": i})

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert store.count("things") == 240
    # The file must still be valid JSON after all that contention.
    json.loads((tmp_path / "things.json").read_text(encoding="utf-8"))


def test_collection_names_are_sanitised(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.put("../escape", "a", {"id": "a"})
    assert store.get("../escape", "a") == {"id": "a"}
    assert all(p.parent == tmp_path for p in tmp_path.glob("*.json")), "no path traversal"


def test_clear(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.put("a", "1", {})
    store.put("b", "1", {})
    store.clear("a")
    assert store.count("a") == 0 and store.count("b") == 1
    store.clear()
    assert store.count("b") == 0


# --------------------------------------------------------------------------- #
# Registry / repositories
# --------------------------------------------------------------------------- #
def test_registry_rehydrates_typed_models(tmp_path: Path) -> None:
    registry = Registry(JsonStore(tmp_path))
    model = ModelInfo(
        id="mdl_1",
        name="sentiment",
        version=1,
        adapter=AdapterKind.SKLEARN,
        artifact=ArtifactSpec(uri="file:///content/m.pkl"),
        schema_in=ModelSchema(fields={"text": {"type": "string"}}, required=["text"]),
    )
    registry.models.put(model)
    fetched = registry.models.get("mdl_1")
    assert isinstance(fetched, ModelInfo)
    assert fetched.schema_in.required == ["text"]
    assert fetched.adapter == AdapterKind.SKLEARN.value or fetched.adapter == AdapterKind.SKLEARN


def test_model_versioning_helpers(tmp_path: Path) -> None:
    registry = Registry(JsonStore(tmp_path))
    for version in (1, 2, 3):
        registry.models.put(
            ModelInfo(
                id=f"mdl_{version}",
                name="sentiment",
                version=version,
                artifact=ArtifactSpec(uri=f"file:///m{version}.pkl"),
            )
        )
    registry.models.put(
        ModelInfo(id="mdl_other", name="other", version=1, artifact=ArtifactSpec(uri="file:///o.pkl"))
    )
    assert registry.models.latest_version("sentiment") == 3
    assert registry.models.latest_version("missing") == 0
    assert registry.models.by_name_version("sentiment", 2).id == "mdl_2"
    assert registry.models.by_name_version("sentiment", 9) is None
    assert [m.version for m in registry.models.versions_of("mdl_1")] == [3, 2, 1]


def test_endpoint_lookup_by_route_is_slash_insensitive(tmp_path: Path) -> None:
    registry = Registry(JsonStore(tmp_path))
    registry.endpoints.put(
        EndpointInfo(id="ep_1", model_id="mdl_1", model_version=1, route="sentiment")
    )
    assert registry.endpoints.by_route("sentiment").id == "ep_1"
    assert registry.endpoints.by_route("/sentiment/").id == "ep_1"
    assert registry.endpoints.by_route("nope") is None


def test_deployment_queries_and_ordering(tmp_path: Path) -> None:
    registry = Registry(JsonStore(tmp_path))
    for dep_id, state in (("dep_1", DeploymentState.UNLOADED), ("dep_2", DeploymentState.READY)):
        registry.deployments.put(
            DeploymentInfo(
                id=dep_id, endpoint_id="ep_1", model_id="mdl_1", model_version=1,
                runtime_id="rt_1", state=state,
            )
        )
    assert len(registry.deployments.for_runtime("rt_1")) == 2
    assert registry.deployments.active_for_endpoint("ep_1").id == "dep_2"
    registry.store.update("deployments", "dep_2", {"state": DeploymentState.FAILED.value})
    assert registry.deployments.active_for_endpoint("ep_1") is None


def test_runtime_online_filters(tmp_path: Path) -> None:
    from colabdeploy_shared.schemas import RuntimeInfo

    registry = Registry(JsonStore(tmp_path))
    for rid, state in (("rt_1", "online"), ("rt_2", "degraded"), ("rt_3", "offline")):
        registry.runtimes.put(RuntimeInfo(id=rid, state=state))  # type: ignore[arg-type]
    assert {r.id for r in registry.runtimes.online()} == {"rt_1", "rt_2"}
    assert {r.id for r in registry.runtimes.connected()} == {"rt_1"}


def test_audit_log_is_bounded(tmp_path: Path) -> None:
    registry = Registry(JsonStore(tmp_path))
    registry.audit.max_entries = 25
    for i in range(60):
        registry.audit.record("test.action", i=i)
    assert registry.store.count("audit") <= 25
    recent = registry.audit.recent(10)
    assert len(recent) == 10
    assert recent[0]["detail"]["i"] > recent[-1]["detail"]["i"], "newest first"


def test_job_pruning(tmp_path: Path) -> None:
    from datetime import timedelta

    from colabdeploy_shared.schemas import JobInfo, utcnow

    registry = Registry(JsonStore(tmp_path))
    old = JobInfo(id="job_old", endpoint_id="ep", model_id="m", model_version=1,
                  done_at=utcnow() - timedelta(days=10))
    fresh = JobInfo(id="job_new", endpoint_id="ep", model_id="m", model_version=1,
                    done_at=utcnow())
    registry.jobs.put(old)
    registry.jobs.put(fresh)
    assert registry.prune_jobs(utcnow() - timedelta(days=7)) == 1
    assert registry.jobs.get("job_old") is None
    assert registry.jobs.get("job_new") is not None


def test_repository_ignores_records_it_cannot_parse(tmp_path: Path) -> None:
    """A schema change must not brick the whole collection on boot."""
    store = JsonStore(tmp_path)
    store.put("endpoints", "good", EndpointInfo(id="good", model_id="m", model_version=1,
                                                route="good").model_dump(mode="json"))
    store.put("endpoints", "bad", {"id": "bad", "cache_ttl_s": "not-an-int"})
    registry = Registry(store)
    routes = [e.route for e in registry.endpoints.list()]
    assert routes == ["good"]
