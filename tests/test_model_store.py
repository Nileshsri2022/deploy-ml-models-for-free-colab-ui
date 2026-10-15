"""The agent-side model pool: load, LRU-evict, reconcile, warm, predict.

This is the component that makes "the VM went away and came back" cheap: on
reconnect the store converges on the control plane's desired state, re-using
artifacts from the blob cache and re-warming from hot request keys, without the
gateway ever having to re-upload anything.

Everything here runs on the ``echo`` and ``python_function`` adapters, so no ML
framework needs to be installed to run the suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from colabdeploy.adapters.base import AdapterError
from colabdeploy.artifacts import ArtifactCache, ArtifactResolver
from colabdeploy.model_store import ModelStore, build_artifact_spec


@pytest.fixture()
def store(tmp_path: Path) -> ModelStore:
    cache = ArtifactCache(tmp_path / "artifacts")
    resolver = ArtifactResolver(cache)
    pool = ModelStore(resolver, cache, max_loaded=3)
    yield pool
    pool.unload_all()
    cache.close()


def deployment(
    model_id: str = "mdl_a",
    version: int = 1,
    *,
    adapter: str = "echo",
    artifact: str = 'inline:{"latency_ms": 0}',
    endpoint_id: str = "ep_1",
    entrypoint: str | None = None,
    warmup_sample: dict[str, Any] | None = None,
    filename: str = "m.echo.json",
) -> dict[str, Any]:
    """A desired-state deployment entry, in the wire shape the gateway sends."""
    return {
        "deployment_id": f"dep_{model_id}_{version}",
        "endpoint_id": endpoint_id,
        "model_id": model_id,
        "model_version": version,
        "adapter": adapter,
        "artifact": {"uri": artifact, "filename": filename},
        "entrypoint": entrypoint,
        "schema_in": {"fields": {}, "required": [], "example": {"ping": True}},
        "schema_out": {"fields": {}, "required": []},
        "warmup_sample": warmup_sample,
        "state": "pending",
    }


def desired_state(*deployments: dict[str, Any], revision: int = 1) -> dict[str, Any]:
    return {
        "runtime_id": "rt_test",
        "generated_at": "2026-01-01T00:00:00Z",
        "revision": revision,
        "deployments": list(deployments),
        "warmup_top_k": 5,
        "settings": {},
    }


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_load_deployment_makes_a_model_ready(store: ModelStore) -> None:
    result = store.load_deployment(deployment())

    assert result.state == "ready"
    assert result.error is None
    assert result.adapter == "echo"
    assert store.keys == ["mdl_a@1"]


def test_load_is_idempotent(store: ModelStore) -> None:
    first = store.load_deployment(deployment())
    second = store.load_deployment(deployment())

    assert first.state == second.state == "ready"
    assert store.keys == ["mdl_a@1"]


def test_second_load_reuses_the_cached_artifact(store: ModelStore) -> None:
    """Same bytes, different model id: the content-addressed cache does the work."""
    store.load_deployment(deployment("mdl_a", 1))
    store.unload_all()
    result = store.load_deployment(deployment("mdl_b", 1))

    assert result.state == "ready"
    assert result.from_cache is True
    assert store.counters["cache_hits"] >= 1


def test_loading_a_new_version_replaces_the_old_key(store: ModelStore) -> None:
    store.load_deployment(deployment("mdl_a", 1))
    store.load_deployment(deployment("mdl_a", 2))

    assert sorted(store.keys) == ["mdl_a@1", "mdl_a@2"]


def test_python_function_adapter_runs_supplied_code(store: ModelStore) -> None:
    source = "def predict(payload, model=None, meta=None):\n    return {'doubled': payload['n'] * 2}"
    result = store.load_deployment(
        deployment(
            "mdl_fn",
            adapter="python_function",
            artifact=f"inline:{source}",
            filename="model.py",
            entrypoint=None,
        )
    )
    assert result.state == "ready", result.error

    output = store.predict("mdl_fn", 1, {"n": 21})
    assert output["output"] == {"doubled": 42}
    assert output["adapter"] == "python_function"


def test_broken_custom_code_fails_the_deployment_without_raising(store: ModelStore) -> None:
    """A syntax error in user code must surface as a failed deployment, not a crash."""
    result = store.load_deployment(
        deployment(
            "mdl_bad",
            adapter="python_function",
            artifact="inline:def predict(payload, model=None, meta=None)\n    return 1",
            filename="model.py",
        )
    )
    assert result.state == "failed"
    assert result.error
    # failures() is keyed by deployment id (falling back to the model key), which
    # is what the agent reports to the gateway for the UI's deployment row.
    assert "dep_mdl_bad_1" in store.failures()
    assert "syntax error" in store.failures()["dep_mdl_bad_1"]
    assert store.keys == []


def test_missing_framework_is_reported_not_fatal(store: ModelStore) -> None:
    """No ML frameworks are installed in CI; the store must say so and carry on."""
    result = store.load_deployment(
        deployment("mdl_sk", adapter="sklearn", artifact="inline:{}", filename="m.pkl")
    )
    assert result.state == "failed"
    assert result.error
    assert store.failures()["dep_mdl_sk_1"] == result.error


def test_unload_removes_the_model(store: ModelStore) -> None:
    store.load_deployment(deployment())
    assert store.unload("mdl_a@1") is True
    assert store.keys == []
    assert store.unload("mdl_a@1") is False


def test_unload_all_returns_the_count(store: ModelStore) -> None:
    store.load_deployment(deployment("mdl_a", 1, endpoint_id="ep_1"))
    store.load_deployment(deployment("mdl_b", 1, endpoint_id="ep_2"))
    assert store.unload_all() == 2
    assert store.keys == []


# --------------------------------------------------------------------------- #
# LRU eviction (the VRAM guard)
# --------------------------------------------------------------------------- #
def test_pool_evicts_the_least_recently_used_model(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "artifacts")
    pool = ModelStore(ArtifactResolver(cache), cache, max_loaded=2)
    try:
        for index, model_id in enumerate(("mdl_a", "mdl_b", "mdl_c")):
            result = pool.load_deployment(
                deployment(model_id, 1, endpoint_id=f"ep_{index}", artifact=f'inline:{{"id": "{model_id}"}}')
            )
            assert result.state == "ready", result.error

        assert len(pool.keys) == 2
        assert "mdl_a@1" not in pool.keys
        assert pool.counters["evictions"] >= 1
    finally:
        pool.unload_all()
        cache.close()


def test_touching_a_model_keeps_it_alive(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "artifacts")
    pool = ModelStore(ArtifactResolver(cache), cache, max_loaded=2)
    try:
        pool.load_deployment(deployment("mdl_a", 1, endpoint_id="ep_1", artifact='inline:{"id": "a"}'))
        pool.load_deployment(deployment("mdl_b", 1, endpoint_id="ep_2", artifact='inline:{"id": "b"}'))

        # Using mdl_a makes mdl_b the eviction candidate.
        pool.predict("mdl_a", 1, {"ping": True})
        pool.load_deployment(deployment("mdl_c", 1, endpoint_id="ep_3", artifact='inline:{"id": "c"}'))

        assert "mdl_a@1" in pool.keys
        assert "mdl_b@1" not in pool.keys
    finally:
        pool.unload_all()
        cache.close()


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def test_predict_reports_latency_and_counts(store: ModelStore) -> None:
    store.load_deployment(deployment())

    result = store.predict("mdl_a", 1, {"ping": True})
    assert result["output"]["echo"] == {"ping": True} or "echo" in result["output"]
    assert result["model_ms"] >= 0.0
    assert result["adapter"] == "echo"
    assert store.get("mdl_a", 1).predictions == 1  # type: ignore[union-attr]


def test_predict_for_unknown_model_raises_with_a_hint(store: ModelStore) -> None:
    with pytest.raises(AdapterError, match="not loaded in this runtime"):
        store.predict("mdl_missing", 1, {})


def test_predict_for_endpoint_routes_to_the_right_model(store: ModelStore) -> None:
    store.load_deployment(deployment("mdl_a", 1, endpoint_id="ep_a"))
    store.load_deployment(deployment("mdl_b", 1, endpoint_id="ep_b", artifact='inline:{"id": "b"}'))

    result = store.predict_for_endpoint("ep_b", {"ping": 2})
    assert store.find_by_endpoint("ep_b").meta.model_id == "mdl_b"  # type: ignore[union-attr]
    assert result["adapter"] == "echo"


def test_predict_for_unknown_endpoint_raises(store: ModelStore) -> None:
    with pytest.raises(AdapterError, match="no model loaded for endpoint"):
        store.predict_for_endpoint("ep_nope", {})


def test_model_errors_are_counted_and_wrapped(store: ModelStore) -> None:
    """A failing model must produce a clean AdapterError, not a raw traceback."""
    store.load_deployment(
        deployment("mdl_fail", adapter="echo", artifact='inline:{"fail": "boom"}', endpoint_id="ep_f")
    )
    with pytest.raises(AdapterError, match="boom"):
        store.predict("mdl_fail", 1, {})

    loaded = store.get("mdl_fail", 1)
    assert loaded is not None
    assert loaded.errors == 1
    assert loaded.last_error == "boom"


# --------------------------------------------------------------------------- #
# Reconciliation (the reconnect path)
# --------------------------------------------------------------------------- #
def test_reconcile_loads_everything_wanted(store: ModelStore) -> None:
    results = store.reconcile(
        desired_state(
            deployment("mdl_a", 1, endpoint_id="ep_1"),
            deployment("mdl_b", 1, endpoint_id="ep_2", artifact='inline:{"id": "b"}'),
        )
    )

    assert [r.state for r in results] == ["ready", "ready"]
    assert sorted(store.keys) == ["mdl_a@1", "mdl_b@1"]


def test_reconcile_unloads_what_is_no_longer_wanted(store: ModelStore) -> None:
    store.reconcile(desired_state(deployment("mdl_a", 1), deployment("mdl_b", 1, endpoint_id="ep_2", artifact='inline:{"id": "b"}')))
    assert sorted(store.keys) == ["mdl_a@1", "mdl_b@1"]

    store.reconcile(desired_state(deployment("mdl_b", 1, endpoint_id="ep_2", artifact='inline:{"id": "b"}'), revision=2))
    assert store.keys == ["mdl_b@1"]


def test_reconcile_accepts_a_typed_desired_state(store: ModelStore) -> None:
    from colabdeploy_shared.schemas import DesiredState

    payload = desired_state(deployment("mdl_a", 1))
    results = store.reconcile(DesiredState.model_validate(payload))
    assert results[0].state == "ready"


def test_reconcile_reports_failures_alongside_successes(store: ModelStore) -> None:
    results = store.reconcile(
        desired_state(
            deployment("mdl_ok", 1, endpoint_id="ep_ok"),
            deployment("mdl_sk", 1, adapter="sklearn", endpoint_id="ep_sk", artifact="inline:{}", filename="m.pkl"),
        )
    )

    states = {r.model_id: r.state for r in results}
    assert states["mdl_ok"] == "ready"
    assert states["mdl_sk"] == "failed"
    assert store.failures()["dep_mdl_sk_1"]


def test_reconcile_warms_from_hot_cache_keys(store: ModelStore) -> None:
    """The gateway ships recent request payloads; they are replayed as warm-up."""
    store.reconcile(desired_state(deployment("mdl_a", 1, endpoint_id="ep_1")), warmup_samples=[])
    assert store.counters["warmups"] == 0

    store.reconcile(
        desired_state(deployment("mdl_a", 1, endpoint_id="ep_1")),
        warmup_samples=[
            {"endpoint_id": "ep_1", "payload": {"ping": 1}},
            {"endpoint_id": "ep_1", "payload": {"ping": 2}},
            {"endpoint_id": "ep_unknown", "payload": {"ping": 3}},
        ],
    )

    assert store.counters["warmups"] == 2
    # Warm-up must not look like served traffic.
    assert store.get("mdl_a", 1).predictions == 0  # type: ignore[union-attr]
    assert store.get("mdl_a", 1).warmup_ms > 0  # type: ignore[union-attr]


def test_warmup_failure_is_counted_not_raised(store: ModelStore) -> None:
    store.load_deployment(
        deployment("mdl_fail", adapter="echo", artifact='inline:{"fail": "nope"}', endpoint_id="ep_f")
    )
    store.warm_from_sample({"endpoint_id": "ep_f", "payload": {"x": 1}})
    assert store.counters["warmup_failures"] == 1
    assert store.counters["warmups"] == 0


def test_warmup_sample_from_the_model_definition_runs_on_load(store: ModelStore) -> None:
    result = store.load_deployment(
        deployment("mdl_w", warmup_sample={"inputs": {"ping": True}}, endpoint_id="ep_w")
    )
    assert result.state == "ready"
    assert result.warm is True
    assert (result.warmup_ms or 0) >= 0.0


def test_warmup_accepts_the_inputs_spelling(store: ModelStore) -> None:
    store.load_deployment(deployment("mdl_a", 1, endpoint_id="ep_1"))
    store.warm_from_sample({"endpoint_id": "ep_1", "inputs": {"ping": True}})
    assert store.counters["warmups"] == 1


def test_snapshot_describes_the_pool(store: ModelStore) -> None:
    store.load_deployment(deployment("mdl_a", 1, endpoint_id="ep_1"))
    snapshot = store.snapshot()

    assert len(snapshot) == 1
    entry = snapshot[0]
    assert entry["key"] == "mdl_a@1"
    assert entry["model_id"] == "mdl_a"
    assert entry["adapter"] == "echo"
    assert entry["state"] == "ready"
    assert entry["endpoint_id"] == "ep_1"


def test_build_artifact_spec_helper() -> None:
    spec = build_artifact_spec("/content/models/m.pkl", checksum="abc123")
    assert spec.uri == "/content/models/m.pkl"
    assert spec.checksum == "abc123"
    assert spec.filename == "m.pkl"
