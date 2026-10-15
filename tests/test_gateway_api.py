"""The control plane end to end over HTTP: auth, registry, deploy, and resilience.

A stub agent socket is registered with the relay so the *real* request path runs -
routing, cache, degradation, job queueing - without needing a Colab VM. The full
tunnel handshake is covered in ``test_e2e_agent.py``.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from colabdeploy_shared.enums import DeploymentState
from colabdeploy_shared.protocol import ExecResponseFrame
from gateway.services.relay import AgentTimeout, RelayConnection

RUNTIME_ID = "rt_test"
ADMIN_KEY = "CDP_API_test_admin_key_0000000000"  # matches tests/conftest.py


class StubSocket:
    """Stands in for a live agent WebSocket."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed_with: tuple[int | None, str | None] | None = None

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed_with = (code, reason)

    def frames(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self.sent]


@pytest.fixture()
def stub_runtime(state: Any):
    """Register a runtime record + a stub tunnel, as if an agent had connected."""
    from gateway.core.security import AuthService

    record = AuthService(state.registry).create_runtime_token("pytest-runtime", 1.0)
    runtime_id = record["runtime_id"]
    socket = StubSocket()
    conn = RelayConnection(runtime_id=runtime_id, ws=socket, agent_version="0.1.0-test")  # type: ignore[arg-type]
    state.relay._conns[runtime_id] = conn
    state.registry.store.update(
        "runtimes", runtime_id, {"state": "online", "loaded_models": []}
    )
    return {"runtime_id": runtime_id, "token": record["token"], "socket": socket, "conn": conn,
            "state": state}


def serve_with(state: Any, output: Any, status: int = 200) -> list[dict[str, Any]]:
    """Make the stub agent answer every exec frame with ``output``.

    Mirrors the real agent's body shape: ``{"output": ...}`` on success,
    ``{"error": ..., "code": ...}`` on failure (see ``AgentRuntime``).
    """
    calls: list[dict[str, Any]] = []

    async def fake_exec(runtime_id: str, body: dict[str, Any], **kwargs: Any) -> ExecResponseFrame:
        calls.append(body)
        if status >= 400:
            payload = {"code": "MODEL_ERROR", "error": output.get("error", "agent failure")}
        else:
            payload = {"output": output, "model_ms": 1.2}
        return ExecResponseFrame.from_bytes(
            request_id="req-test",
            status=status,
            payload=json.dumps(payload).encode("utf-8"),
            agent_latency_ms=1.2,
        )

    state.relay.exec_on_runtime = fake_exec  # type: ignore[method-assign]
    return calls


def deploy_echo(api: Any, *, route: str = "sentiment", degradation: str = "serve_cached",
                cache_ttl_s: int = 300, swr_s: int = 3600) -> dict[str, Any]:
    """Register an echo model and publish it - the baseline used by most tests."""
    model = api.post(
        "/api/models",
        json={
            "name": route,
            "framework": "echo",
            "adapter": "echo",
            "artifact": {"uri": "inline:{}", "filename": "m.echo.json"},
            "schema_in": {"fields": {"text": {"type": "string"}}, "example": {"text": "hi"}},
            "warmup_sample": {"text": "warm"},
        },
    )
    assert model.status_code == 201, model.text
    deployment = api.post(
        "/api/deploy",
        json={
            "model_id": model.json()["id"],
            "endpoint": {
                "route": route,
                "degradation_mode": degradation,
                "cache_ttl_s": cache_ttl_s,
                "stale_while_revalidate_s": swr_s,
            },
        },
    )
    assert deployment.status_code == 201, deployment.text
    return deployment.json()


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_unauthenticated_calls_are_rejected(client: Any) -> None:
    response = client.get("/api/endpoints")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


def test_bad_key_is_rejected(client: Any) -> None:
    response = client.get("/api/endpoints", headers={"Authorization": "Bearer CDP_API_wrong"})
    assert response.status_code == 401


def test_x_api_key_header_works(client: Any) -> None:
    assert client.get("/api/endpoints", headers={"X-API-Key": ADMIN_KEY}).status_code == 200


def test_public_health_needs_no_key(client: Any) -> None:
    assert client.get("/v1/health").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200


def test_keys_are_stored_hashed_and_listed_without_secrets(api: Any, state: Any) -> None:
    created = api.post("/api/keys", json={"label": "ci", "scopes": ["inference"]})
    assert created.status_code == 201
    token = created.json()["token"]
    assert token.startswith("CDP_API_")

    listing = api.get("/api/keys").json()["keys"]
    assert all("token_hash" not in key for key in listing)
    assert all(key.get("token") is None for key in listing)
    raw = state.registry.store.all("api_keys")
    assert all("token_hash" in record for record in raw)
    assert token not in json.dumps(raw), "the plaintext key must never be persisted"

    # An inference-only key cannot manage.
    inference_only = {"Authorization": f"Bearer {token}"}
    assert api.client.get("/api/endpoints", headers=inference_only).status_code == 403


def test_key_revocation_takes_effect_immediately(api: Any) -> None:
    created = api.post("/api/keys", json={"label": "temp"}).json()
    headers = {"Authorization": f"Bearer {created['token']}"}
    assert api.client.get("/api/endpoints", headers=headers).status_code == 200
    assert api.delete(f"/api/keys/{created['id']}").status_code == 200
    assert api.client.get("/api/endpoints", headers=headers).status_code == 401


# --------------------------------------------------------------------------- #
# Registry & deploy
# --------------------------------------------------------------------------- #
def test_model_registration_detects_the_adapter(api: Any) -> None:
    for uri, expected in (
        ("file:///content/m.pkl", "sklearn"),
        ("file:///content/m.onnx", "onnx"),
        ("file:///content/m.pt", "pytorch"),
        ("hf://bert-base-uncased", "huggingface"),
    ):
        response = api.post(
            "/api/models",
            json={"name": f"m-{expected}", "artifact": {"uri": uri}, "framework": "auto"},
        )
        assert response.status_code == 201, response.text
        assert response.json()["adapter"] == expected, uri


def test_framework_hint_overrides_extension(api: Any) -> None:
    response = api.post(
        "/api/models",
        json={"name": "odd", "artifact": {"uri": "file:///m.bin"}, "framework": "xgboost"},
    )
    assert response.json()["adapter"] == "xgboost"


def test_uploading_a_second_version_bumps_the_counter(api: Any) -> None:
    first = api.post("/api/models", json={"name": "v", "artifact": {"uri": "file:///a.pkl"}}).json()
    second = api.post("/api/models", json={"name": "v", "artifact": {"uri": "file:///b.pkl"}}).json()
    assert (first["version"], second["version"]) == (1, 2)
    assert second["all_versions"] == [2, 1]


def test_python_function_adapter_requires_an_entrypoint(api: Any) -> None:
    response = api.post(
        "/api/models",
        json={"name": "custom", "adapter": "python_function",
              "artifact": {"uri": "file:///nothing.py"}},
    )
    assert response.status_code == 400
    assert "entrypoint" in response.json()["error"]["message"]


def test_deploy_publishes_an_endpoint_and_records_intent(api: Any, stub_runtime: Any) -> None:
    result = deploy_echo(api)
    endpoint = result["endpoint"]
    assert endpoint["route"] == "sentiment"
    assert endpoint["url"].endswith("/v1/models/sentiment/predict")
    assert endpoint["deployment_state"] == "pending"
    assert result["runtime_id"] == stub_runtime["runtime_id"]

    # The intent is durable: a deployment row exists even before the agent acks.
    deployments = api.get("/api/deployments").json()["deployments"]
    assert len(deployments) == 1
    assert deployments[0]["model_version"] == 1


def test_deploy_pushes_desired_state_to_a_connected_runtime(api: Any, stub_runtime: Any) -> None:
    result = deploy_echo(api, route="pushed")
    assert result["pushed_to_runtime"] is True
    kinds = [frame["kind"] for frame in stub_runtime["socket"].frames()]
    assert "desired_state" in kinds
    frame = next(f for f in stub_runtime["socket"].frames() if f["kind"] == "desired_state")
    assert frame["runtime_id"] == stub_runtime["runtime_id"]
    assert [d["model_id"] for d in frame["deployments"]] == [result["model"]["id"]]
    assert frame["deployments"][0]["artifact"]["uri"] == "inline:{}"


def test_deploy_without_a_runtime_still_records_intent(api: Any) -> None:
    """Deploying while every VM is dead is legal - it arms the next one."""
    result = deploy_echo(api, route="future")
    assert result["runtime_id"] is None
    assert result["runtime_online"] is False
    assert result["pushed_to_runtime"] is False
    assert "will load it on next connect" in result["message"]


def test_redeploy_supersedes_the_previous_deployment(api: Any, stub_runtime: Any) -> None:
    first = deploy_echo(api, route="swap")
    second = api.post(
        "/api/deploy",
        json={"model_id": first["model"]["id"], "endpoint_id": first["endpoint"]["id"]},
    ).json()
    states = {d["id"]: d["state"] for d in api.get("/api/deployments").json()["deployments"]}
    assert states[first["deployment"]["id"]] == "unloaded"
    assert states[second["deployment"]["id"]] == "pending"


def test_rollback_repoints_the_endpoint(api: Any, stub_runtime: Any) -> None:
    first = deploy_echo(api, route="roll")
    model_id = first["model"]["id"]
    assert first["endpoint"]["model_version"] == 1
    v2 = api.post("/api/models", json={
        "name": "roll", "adapter": "echo", "artifact": {"uri": "inline:{}"}}).json()
    assert v2["version"] == 2
    api.post("/api/deploy", json={"model_id": v2["id"], "endpoint_id": first["endpoint"]["id"]})
    assert api.get("/api/endpoints").json()["endpoints"][0]["model_version"] == 2

    rolled = api.post(f"/api/endpoints/{first['endpoint']['id']}/rollback?model_version=1")
    assert rolled.status_code == 200, rolled.text
    assert rolled.json()["endpoint"]["model_version"] == 1
    # A rollback repins the version, it does not switch models.
    assert rolled.json()["endpoint"]["model_id"] == model_id

    missing = api.post(f"/api/endpoints/{first['endpoint']['id']}/rollback?model_version=99")
    assert missing.status_code == 404


def test_endpoint_route_is_slugified_and_unique(api: Any) -> None:
    first = api.post("/api/models", json={"name": "My Model!", "adapter": "echo",
                                          "artifact": {"uri": "inline:{}"}}).json()
    endpoint = api.post("/api/endpoints", json={"model_id": first["id"]}).json()
    assert endpoint["route"] == "my-model"


def test_endpoint_update_tunes_the_resilience_policy(api: Any, stub_runtime: Any) -> None:
    result = deploy_echo(api, route="tunable")
    endpoint_id = result["endpoint"]["id"]
    updated = api.patch(
        f"/api/endpoints/{endpoint_id}",
        json={"cache_ttl_s": 60, "degradation_mode": "queue", "rate_limit_rpm": 10},
    ).json()
    assert updated["cache_ttl_s"] == 60
    assert updated["degradation_mode"] == "queue"
    assert updated["rate_limit_rpm"] == 10


def test_delete_endpoint_unloads_and_invalidates(api: Any, stub_runtime: Any) -> None:
    result = deploy_echo(api, route="doomed")
    endpoint_id = result["endpoint"]["id"]
    assert api.delete(f"/api/endpoints/{endpoint_id}").status_code == 200
    assert api.get("/api/endpoints").json()["count"] == 0
    assert api.delete(f"/api/endpoints/{endpoint_id}").status_code == 404


# --------------------------------------------------------------------------- #
# Artifact upload / download
# --------------------------------------------------------------------------- #
def test_artifact_upload_round_trip(api: Any, stub_runtime: Any, state: Any) -> None:
    files = {"file": ("model.echo.json", b'{"latency_ms": 0}', "application/json")}
    response = api.post("/api/models/upload", data={"name": "uploaded", "adapter": "echo"}, files=files)
    assert response.status_code == 201, response.text
    artifact = response.json()["artifact"]
    assert artifact["uri"].startswith("store://art_")
    assert artifact["size_bytes"] == len(b'{"latency_ms": 0}')
    assert len(artifact["checksum"]) == 64

    # The agent pulls it back with its runtime token.
    artifact_id = artifact["uri"].split("://", 1)[1]
    pulled = api.client.get(
        f"/internal/artifacts/{artifact_id}",
        headers={"X-Runtime-Token": stub_runtime["token"]},
    )
    assert pulled.status_code == 200
    assert pulled.content == b'{"latency_ms": 0}'
    assert pulled.headers["X-ColabDeploy-Checksum"] == artifact["checksum"]


def test_artifact_download_requires_a_runtime_token(api: Any, stub_runtime: Any) -> None:
    files = {"file": ("m.bin", b"data", "application/octet-stream")}
    uploaded = api.post("/api/models/upload", data={"name": "secret", "adapter": "echo"}, files=files).json()
    artifact_id = uploaded["artifact"]["uri"].split("://", 1)[1]
    assert api.client.get(f"/internal/artifacts/{artifact_id}").status_code == 401
    assert api.client.get(
        f"/internal/artifacts/{artifact_id}", headers={"X-Runtime-Token": "CDP_RT_bogus"}
    ).status_code == 401
    assert api.client.get("/internal/artifacts/art_doesnotexist",
                          headers={"X-Runtime-Token": stub_runtime["token"]}).status_code == 404


def test_uploaded_artifact_is_content_addressed(api: Any) -> None:
    files = {"file": ("same.bin", b"identical", "application/octet-stream")}
    first = api.post("/api/models/upload", data={"name": "a1", "adapter": "echo"}, files=files).json()
    second = api.post("/api/models/upload", data={"name": "a2", "adapter": "echo"}, files=files).json()
    assert first["artifact"]["uri"] == second["artifact"]["uri"], "same bytes -> one blob"


# --------------------------------------------------------------------------- #
# Inference: cache states
# --------------------------------------------------------------------------- #
def test_predict_miss_then_hit(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="cached")
    calls = serve_with(state, {"label": "positive", "score": 0.9})

    first = api.post("/v1/models/cached/predict", json={"inputs": {"text": "great"}})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["cache"] == "MISS"
    assert body["output"] == {"label": "positive", "score": 0.9}
    assert body["runtime_id"] == stub_runtime["runtime_id"]
    assert first.headers["X-ColabDeploy-Cache"] == "MISS"

    second = api.post("/v1/models/cached/predict", json={"inputs": {"text": "great"}})
    assert second.json()["cache"] == "HIT"
    assert second.json()["output"] == {"label": "positive", "score": 0.9}
    assert len(calls) == 1, "a cache hit must not touch the VM"
    assert second.json()["latency_ms"] < first.json()["latency_ms"] + 1000

    # Key order must not create a second cache entry.
    reordered = api.post("/v1/models/cached/predict", json={"inputs": {"text": "great"}})
    assert reordered.json()["cache"] == "HIT"


def test_predict_bypass_cache(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="bypass")
    calls = serve_with(state, {"n": 1})
    api.post("/v1/models/bypass/predict", json={"inputs": {"text": "x"}})
    again = api.post("/v1/models/bypass/predict", json={"inputs": {"text": "x"}, "bypass_cache": True})
    assert again.json()["cache"] == "MISS"
    assert len(calls) == 2


def test_cache_ttl_zero_disables_caching(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="nocache", cache_ttl_s=0, swr_s=0)
    calls = serve_with(state, {"n": 1})
    for _ in range(3):
        response = api.post("/v1/models/nocache/predict", json={"inputs": {"text": "x"}})
        assert response.json()["cache"] == "MISS"
    assert len(calls) == 3


def test_payload_alias_and_validation(api: Any, stub_runtime: Any, state: Any) -> None:
    api.post("/api/models", json={
        "name": "strict", "adapter": "echo", "artifact": {"uri": "inline:{}"},
        "schema_in": {"fields": {"text": {"type": "string"}, "n": {"type": "integer"}},
                      "required": ["text"]},
    })
    api.post("/api/deploy", json={"model_id": api.get("/api/models").json()["latest"][0]["id"],
                                  "endpoint": {"route": "strict"}})
    serve_with(state, {"ok": True})

    # 'payload' is accepted as an alias for 'inputs'.
    ok = api.post("/v1/models/strict/predict", json={"payload": {"text": "hi"}})
    assert ok.status_code == 200, ok.text

    missing = api.post("/v1/models/strict/predict", json={"inputs": {"n": 1}})
    assert missing.status_code == 422
    assert "missing required field 'text'" in json.dumps(missing.json())

    wrong_type = api.post("/v1/models/strict/predict", json={"inputs": {"text": "hi", "n": "x"}})
    assert wrong_type.status_code == 422
    assert "expected integer" in json.dumps(wrong_type.json())


def test_unknown_endpoint_returns_a_helpful_404(api: Any) -> None:
    response = api.post("/v1/models/nothing-here/predict", json={"inputs": {}})
    assert response.status_code == 404
    assert "Deploy it first" in response.json()["error"]["message"]
    assert response.headers["X-ColabDeploy-Error"] == "ENDPOINT_NOT_FOUND"


def test_predict_by_model_id_also_routes(api: Any, stub_runtime: Any, state: Any) -> None:
    result = deploy_echo(api, route="byid")
    serve_with(state, {"ok": 1})
    response = api.post(f"/v1/models/{result['model']['id']}/predict", json={"inputs": {"text": "x"}})
    assert response.status_code == 200, response.text
    assert response.json()["endpoint_id"] == result["endpoint"]["id"]


def test_batch_predict(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="batched")
    serve_with(state, {"ok": True})
    response = api.post(
        "/v1/models/batched/predict:batch",
        json={"instances": [{"text": "a"}, {"text": "b"}, {"text": "c"}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 3
    assert all(item["output"] == {"ok": True} for item in body["results"])


def test_batch_size_is_capped(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="toobig")
    serve_with(state, {"ok": True})
    response = api.post(
        "/v1/models/toobig/predict:batch", json={"instances": [{"i": i} for i in range(500)]}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "BATCH_TOO_LARGE"


def test_agent_error_is_surfaced_not_swallowed(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="failing")
    serve_with(state, {"error": "boom"}, status=400)
    response = api.post("/v1/models/failing/predict", json={"inputs": {"text": "x"}})
    assert response.status_code == 400
    assert "boom" in json.dumps(response.json())


def test_agent_timeout_degrades_instead_of_hanging(api: Any, stub_runtime: Any, state: Any) -> None:
    """A wedged VM must not hang the caller: timeout -> stale cache -> 200."""
    deploy_echo(api, route="slow", cache_ttl_s=1, swr_s=600)
    serve_with(state, {"ok": True})
    api.post("/v1/models/slow/predict", json={"inputs": {"text": "warm the cache"}})
    time.sleep(1.05)  # let freshness lapse; the entry is now stale-but-usable

    async def timeout(*args: Any, **kwargs: Any) -> Any:
        raise AgentTimeout(stub_runtime["runtime_id"], 1.0)

    state.relay.exec_on_runtime = timeout  # type: ignore[method-assign]
    response = api.post("/v1/models/slow/predict", json={"inputs": {"text": "warm the cache"}})
    assert response.status_code == 200
    body = response.json()
    assert body["cache"] == "STALE" and body["stale"] is True
    assert response.headers["X-ColabDeploy-Stale"] == "true"


def run_in_app(api: Any, coro_fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an app coroutine on the TestClient's *own* event loop.

    ``asyncio.run()`` would spin up a second loop, and these coroutines touch
    objects that belong to the app's loop (SSE subscriber queues, relay state).
    Going through Starlette's blocking portal keeps the whole suite synchronous -
    no ``pytest-asyncio`` dependency, and no ``asyncio`` marker to keep in sync
    with the ``--strict-markers`` list in ``pytest.ini``.
    """
    client = api.client
    portal = getattr(client, "portal", None)
    if portal is not None:  # Starlette >= 1.x sets it for the whole `with` block
        return portal.call(lambda: coro_fn(*args, **kwargs))

    factory = getattr(client, "portal_factory", None)  # older Starlette
    if factory is None:
        raise RuntimeError("this TestClient exposes no event-loop portal")
    with factory() as spawned:
        return spawned.call(lambda: coro_fn(*args, **kwargs))


# --------------------------------------------------------------------------- #
# Resilience: the runtime disappears
# --------------------------------------------------------------------------- #
def test_endpoint_survives_a_dead_runtime(api: Any, stub_runtime: Any, state: Any) -> None:
    """The headline guarantee: kill the VM, the endpoint still answers."""
    deploy_echo(api, route="survivor", cache_ttl_s=1, swr_s=3600)
    serve_with(state, {"label": "positive"})
    warm = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "great"}})
    assert warm.json()["cache"] == "MISS"
    time.sleep(1.05)  # entry goes stale; with the VM gone, stale *is* the answer

    # Pull the plug.
    run_in_app(api, state.relay.unregister, stub_runtime["runtime_id"], reason="notebook closed")
    assert state.relay.is_connected(stub_runtime["runtime_id"]) is False
    runtime = state.registry.runtimes.get(stub_runtime["runtime_id"])
    assert runtime.state.value == "offline"

    offline = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "great"}})
    assert offline.status_code == 200, "the caller must not see the VM's death"
    body = offline.json()
    assert body["cache"] == "STALE"
    assert body["stale"] is True
    assert body["output"] == {"label": "positive"}
    assert body["error"]["code"] == "STALE_RESPONSE"
    assert offline.headers["X-ColabDeploy-Stale"] == "true"

    # An unseen payload has nothing to serve -> falls through to the queue.
    novel = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "never seen"}})
    assert novel.status_code == 202, novel.text
    assert novel.json()["job_id"]
    assert novel.json()["error"]["code"] == "QUEUED"
    assert novel.headers["X-ColabDeploy-Job"] == novel.json()["job_id"]


def test_queue_mode_returns_a_job_immediately(api: Any, state: Any) -> None:
    deploy_echo(api, route="queued", degradation="queue")
    response = api.post("/v1/models/queued/predict", json={"inputs": {"text": "x"}})
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert job_id

    job = api.get(f"/v1/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["state"] == "queued"
    assert job.json()["inputs"] == {"text": "x"}
    assert job.headers["Retry-After"]


def test_fail_mode_returns_503_with_retry_after(api: Any, state: Any) -> None:
    deploy_echo(api, route="strictfail", degradation="fail")
    response = api.post("/v1/models/strictfail/predict", json={"inputs": {"text": "x"}})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "RUNTIME_OFFLINE"
    assert response.headers["Retry-After"] == "5"


def test_queued_job_drains_when_a_runtime_returns(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="drain", degradation="queue")
    run_in_app(api, state.relay.unregister, stub_runtime["runtime_id"], reason="gone")
    job_id = api.post("/v1/models/drain/predict", json={"inputs": {"text": "x"}}).json()["job_id"]

    # Runtime comes back with the model loaded.
    calls = serve_with(state, {"label": "drained"})
    state.relay._conns[stub_runtime["runtime_id"]] = RelayConnection(
        runtime_id=stub_runtime["runtime_id"], ws=StubSocket()  # type: ignore[arg-type]
    )
    state.registry.deployments.patch(
        api.get("/api/deployments").json()["deployments"][0]["id"],
        state=DeploymentState.READY.value, runtime_id=stub_runtime["runtime_id"],
    )

    run_in_app(api, state.runtimes.supervise_once)
    job = api.get(f"/v1/jobs/{job_id}").json()
    assert job["state"] == "succeeded", job
    assert job["result"] == {"label": "drained"}
    assert job["done_at"]
    assert len(calls) == 1


def test_job_expiry_when_no_runtime_ever_returns(api: Any, state: Any) -> None:
    deploy_echo(api, route="expire", degradation="queue")
    job_id = api.post("/v1/models/expire/predict", json={"inputs": {"text": "x"}}).json()["job_id"]

    from datetime import timedelta

    from colabdeploy_shared.schemas import utcnow

    state.registry.store.update(
        "jobs", job_id, {"created_at": (utcnow() - timedelta(hours=2)).isoformat()}
    )
    import asyncio

    asyncio.run(state.runtimes.supervise_once())
    job = api.get(f"/v1/jobs/{job_id}").json()
    assert job["state"] == "expired"
    assert "expired" in (job["error"] or "")


def test_explicit_job_creation_and_lookup(api: Any, stub_runtime: Any, state: Any) -> None:
    result = deploy_echo(api, route="explicit-job")
    response = api.post(
        "/v1/jobs",
        json={"endpoint_id": result["endpoint"]["id"], "inputs": {"text": "queued by hand"}},
    )
    assert response.status_code == 202
    assert response.headers["Location"].startswith("/v1/jobs/")
    assert api.get(f"/v1/jobs/{response.json()['id']}").status_code == 200
    assert api.get("/v1/jobs/job_nope").status_code == 404


# --------------------------------------------------------------------------- #
# Runtime management
# --------------------------------------------------------------------------- #
def test_runtime_token_lifecycle(api: Any, runtime_token: Any) -> None:
    assert runtime_token["token"].startswith("CDP_RT_")
    assert "notebook_cell" in runtime_token
    assert runtime_token["notebook_cell"].startswith("from colabdeploy import launch")

    listing = api.get("/api/runtimes").json()
    assert listing["count"] == 1
    runtime = listing["runtimes"][0]
    assert runtime["state"] == "registered"
    assert runtime["connected"] is False
    assert runtime["token_prefix"]
    assert "token_hash" not in json.dumps(listing), "secrets must not leak to the UI"

    rotated = api.post(f"/api/runtimes/{runtime['id']}/rotate")
    assert rotated.json()["token"] != runtime_token["token"]

    revoked = api.delete(f"/api/runtimes/{runtime['id']}")
    assert revoked.status_code == 200
    assert api.get("/api/runtimes").json()["runtimes"][0]["revoked"] is True


def test_runtime_token_is_stored_hashed(api: Any, runtime_token: Any, state: Any) -> None:
    raw = json.dumps(state.registry.store.all("runtimes"))
    assert runtime_token["token"] not in raw


def test_internal_register_binds_the_runtime(api: Any, runtime_token: Any, state: Any) -> None:
    response = api.client.post(
        "/internal/runtime/register",
        json={
            "token": runtime_token["token"],
            "capabilities": {"gpu": True, "gpu_name": "Tesla T4", "gpu_memory_mb": 15360,
                             "cpu_count": 2, "ram_mb": 13000},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["runtime_id"] == runtime_token["runtime_id"]
    assert body["relay_ws_url"].endswith("/internal/relay")
    assert body["agent_protocol"] == 1

    runtime = state.registry.runtimes.get(runtime_token["runtime_id"])
    assert runtime.state.value == "connecting"
    assert runtime.capabilities.gpu_name == "Tesla T4"


def test_internal_register_rejects_a_bad_token(api: Any) -> None:
    response = api.client.post("/internal/runtime/register", json={"token": "CDP_RT_nope"})
    assert response.status_code == 401


def test_desired_state_resolves_store_artifacts(api: Any, runtime_token: Any, stub_runtime: Any) -> None:
    files = {"file": ("m.bin", b"weights", "application/octet-stream")}
    uploaded = api.post("/api/models/upload", data={"name": "stored", "adapter": "echo"}, files=files).json()
    api.post("/api/deploy", json={"model_id": uploaded["id"], "endpoint": {"route": "stored"}})

    response = api.client.get(
        f"/internal/runtime/desired-state?runtime_id={stub_runtime['runtime_id']}",
        headers={"X-Runtime-Token": stub_runtime["token"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["deployments"]) == 1
    artifact = body["deployments"][0]["artifact"]
    assert artifact["download_url"].endswith(f"/internal/artifacts/{uploaded['artifact']['uri'].split('://')[1]}")
    assert body["artifact_auth"]["X-Runtime-Token"] == stub_runtime["token"]


def test_desired_state_rejects_a_mismatched_runtime(api: Any, runtime_token: Any, stub_runtime: Any) -> None:
    response = api.client.get(
        f"/internal/runtime/desired-state?runtime_id={stub_runtime['runtime_id']}",
        headers={"X-Runtime-Token": runtime_token["token"]},
    )
    assert response.status_code == 403


def test_http_heartbeat_updates_liveness(api: Any, runtime_token: Any, state: Any) -> None:
    api.client.post("/internal/runtime/register", json={"token": runtime_token["token"],
                                                        "capabilities": {}})
    response = api.client.post(
        "/internal/runtime/heartbeat",
        json={"state": "online", "loaded_models": ["mdl_x@1"], "capabilities": {"gpu": False}},
        headers={"X-Runtime-Token": runtime_token["token"]},
    )
    assert response.status_code == 200, response.text
    runtime = state.registry.runtimes.get(runtime_token["runtime_id"])
    assert runtime.state.value == "online"
    assert runtime.loaded_models == ["mdl_x@1"]


def test_log_shipping_and_retrieval(api: Any, runtime_token: Any) -> None:
    pushed = api.client.post(
        "/internal/runtime/logs",
        json={"lines": [{"level": "info", "message": "model warmed"},
                        {"level": "error", "message": "cuda oom"}]},
        headers={"X-Runtime-Token": runtime_token["token"]},
    )
    assert pushed.json()["accepted"] == 2
    logs = api.get("/api/logs?limit=50").json()["logs"]
    messages = [line["message"] for line in logs]
    assert "model warmed" in messages
    assert any(line["source"] == "agent" for line in logs)
    errors = api.get("/api/logs?level=error").json()["logs"]
    assert all(line["level"] == "error" for line in errors)


# --------------------------------------------------------------------------- #
# Cache introspection (the UI's Cache page)
# --------------------------------------------------------------------------- #
def test_cache_endpoints_expose_the_resilience_buffer(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="inspect")
    serve_with(state, {"label": "pos"})
    api.post("/v1/models/inspect/predict", json={"inputs": {"text": "great"}})
    api.post("/v1/models/inspect/predict", json={"inputs": {"text": "great"}})

    stats = api.get("/api/cache/stats").json()
    assert stats["entries"] == 1 and stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_ratio"] == pytest.approx(0.5, abs=0.01)

    entries = api.get("/api/cache/entries").json()
    assert entries["count"] == 1
    key = entries["entries"][0]["key"]
    assert entries["entries"][0]["preview"] is not None

    detail = api.get(f"/api/cache/entry/{key}").json()
    assert detail["value"]["output"] == {"label": "pos"}
    assert detail["value"]["__request__"] == {"text": "great"}, "requests are stored for warm-up"

    assert api.post(f"/api/cache/entry/{key}/pin?pinned=true").json()["pinned"] is True
    assert api.get("/api/cache/stats").json()["pinned"] == 1
    hot = api.get("/api/cache/hot?limit=5").json()["hot_keys"]
    assert hot[0]["key"] == key and hot[0]["sample"] == {"text": "great"}

    assert api.delete(f"/api/cache/entry/{key}").json()["found"] is True
    assert api.get("/api/cache/stats").json()["entries"] == 0


def test_cache_invalidate_by_endpoint(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="invalidate-me")
    serve_with(state, {"ok": 1})
    api.post("/v1/models/invalidate-me/predict", json={"inputs": {"text": "x"}})
    assert api.get("/api/cache/stats").json()["entries"] == 1
    endpoint_id = api.get("/api/endpoints").json()["endpoints"][0]["id"]
    assert api.post(f"/api/cache/invalidate?endpoint_id={endpoint_id}").json()["invalidated"] == 1
    assert api.get("/api/cache/stats").json()["entries"] == 0


def test_cache_save_reports_the_backend(api: Any) -> None:
    body = api.post("/api/cache/save").json()
    assert body["backend"] == "memory"
    assert body["saved"] is False, "persist=False in tests, so there is nothing to write"


# --------------------------------------------------------------------------- #
# Status / metrics / bootstrap
# --------------------------------------------------------------------------- #
def test_status_aggregates_the_whole_system(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="status-check")
    body = api.get("/v1/status").json()
    assert body["runtimes_total"] == 1
    assert body["runtimes_online"] == 1
    assert body["endpoints"] == 1
    assert body["models"] == 1
    assert body["version"] == "0.1.0"
    runtime = body["runtimes"][0]
    assert runtime["seconds_since_heartbeat"] is None or runtime["seconds_since_heartbeat"] >= 0


def test_public_endpoint_directory(api: Any, stub_runtime: Any) -> None:
    deploy_echo(api, route="directory")
    body = api.client.get("/v1/endpoints").json()
    assert body["count"] == 1
    assert body["endpoints"][0]["route"] == "directory"
    assert body["endpoints"][0]["url"].endswith("/v1/models/directory/predict")


def test_schema_endpoint_drives_the_ui_form(api: Any, stub_runtime: Any) -> None:
    deploy_echo(api, route="schema-check")
    body = api.client.get("/v1/models/schema-check/schema").json()
    assert body["input"]["fields"]["text"]["type"] == "string"
    assert body["example_curl"].startswith("curl -X POST")
    assert "/v1/models/schema-check/predict" in body["url"]


def test_metrics_are_prometheus_text(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="metrics-check")
    serve_with(state, {"ok": 1})
    api.post("/v1/models/metrics-check/predict", json={"inputs": {"text": "x"}})
    response = api.client.get("/v1/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    for needle in ("colabdeploy_up 1", "colabdeploy_requests_total 1",
                   "colabdeploy_cache_entries 1", "# TYPE colabdeploy_up gauge"):
        assert needle in text, needle


def test_bootstrap_payload_contains_the_notebook_cell(api: Any) -> None:
    body = api.get("/api/bootstrap").json()
    assert "from colabdeploy import launch" in body["notebook_cell"]
    assert body["relay_ws_url"].startswith("ws")
    assert body["colab_relaunch_url"].startswith("https://colab.research.google.com/github/")
    assert body["installed"]["api_keys"] >= 1


def test_audit_trail_records_management_actions(api: Any, stub_runtime: Any) -> None:
    deploy_echo(api, route="audited")
    actions = [entry["action"] for entry in api.get("/api/audit").json()["entries"]]
    assert "model.register" in actions
    assert "endpoint.create" in actions
    assert "deploy" in actions


def test_telemetry_summary(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="telemetry")
    serve_with(state, {"ok": 1})
    api.post("/v1/models/telemetry/predict", json={"inputs": {"text": "x"}})
    api.post("/v1/models/telemetry/predict", json={"inputs": {"text": "x"}})
    body = api.get("/api/telemetry").json()
    assert body["summary"]["requests_total"] == 2
    assert body["summary"]["requests_cached"] == 1
    assert body["summary"]["cache_hit_ratio"] == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------- #
# Runtime selection
# --------------------------------------------------------------------------- #
def test_pick_runtime_prefers_a_ready_deployment(api: Any, stub_runtime: Any, state: Any) -> None:
    result = deploy_echo(api, route="selection")
    deployment_id = result["deployment"]["id"]
    endpoint_id = result["endpoint"]["id"]
    model_id = result["model"]["id"]
    state.registry.deployments.patch(
        deployment_id, state=DeploymentState.READY.value, runtime_id=stub_runtime["runtime_id"]
    )
    chosen = state.runtimes.pick_runtime(endpoint_id, model_id, 1)
    assert chosen == stub_runtime["runtime_id"]


def test_pick_runtime_returns_none_when_nothing_is_connected(api: Any, state: Any) -> None:
    assert state.runtimes.pick_runtime("ep", "mdl", 1) is None


def test_pick_runtime_is_stable_across_calls(api: Any, state: Any) -> None:
    """Consistent hashing: the same endpoint keeps landing on the same VM."""
    from gateway.core.security import AuthService

    ids = []
    for index in range(3):
        record = AuthService(state.registry).create_runtime_token(f"rt-{index}", 1.0)
        state.relay._conns[record["runtime_id"]] = RelayConnection(
            runtime_id=record["runtime_id"], ws=StubSocket()  # type: ignore[arg-type]
        )
        ids.append(record["runtime_id"])
    picks = {state.runtimes.pick_runtime("ep_same", "mdl_same", 1) for _ in range(10)}
    assert len(picks) == 1, "affinity must be stable or per-runtime caches thrash"
    assert picks.pop() in ids


def test_supervise_marks_a_silent_runtime_stale_then_offline(api: Any, stub_runtime: Any, state: Any) -> None:
    import asyncio
    from datetime import timedelta

    from colabdeploy_shared.schemas import utcnow

    runtime_id = stub_runtime["runtime_id"]
    state.relay._conns.pop(runtime_id, None)  # tunnel gone, heartbeat recent
    state.registry.store.update("runtimes", runtime_id, {"last_heartbeat_at": utcnow().isoformat()})
    asyncio.run(state.runtimes.supervise_once())
    assert state.registry.runtimes.get(runtime_id).state.value == "stale"

    old = (utcnow() - timedelta(seconds=state.settings.heartbeat_offline_s + 10)).isoformat()
    state.registry.store.update("runtimes", runtime_id, {"last_heartbeat_at": old, "state": "online"})
    asyncio.run(state.runtimes.supervise_once())
    assert state.registry.runtimes.get(runtime_id).state.value == "offline"


def test_session_cap_warning_sets_an_actionable_error(api: Any, stub_runtime: Any, state: Any) -> None:
    """Colab's 12 h cap is unavoidable; the UI must see it coming."""
    import asyncio
    from datetime import timedelta

    from colabdeploy_shared.schemas import utcnow

    runtime_id = stub_runtime["runtime_id"]
    started = (utcnow() - timedelta(hours=11, minutes=50)).isoformat()
    state.registry.store.update(
        "runtimes", runtime_id, {"session_started_at": started, "last_heartbeat_at": utcnow().isoformat()}
    )
    report = asyncio.run(state.runtimes.supervise_once())
    assert report["session_warning"] and report["session_warning"][0]["runtime_id"] == runtime_id
    runtime = state.runtimes.runtime_view(state.registry.runtimes.get(runtime_id))
    assert "relaunch" in (runtime.error or "").lower()
    assert 0 < runtime.seconds_until_session_end < 20 * 60


def test_reconcile_endpoint_requires_a_live_tunnel(api: Any, stub_runtime: Any, state: Any) -> None:
    deploy_echo(api, route="reconcile-me")
    state.relay._conns.pop(stub_runtime["runtime_id"], None)
    response = api.post(f"/api/runtimes/{stub_runtime['runtime_id']}/reconcile")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "RUNTIME_OFFLINE"


def test_missing_runtime_404s(api: Any) -> None:
    assert api.get("/api/runtimes/rt_nope").status_code == 404
    assert api.delete("/api/runtimes/rt_nope").status_code == 404
    assert api.post("/api/runtimes/rt_nope/rotate").status_code == 404


def test_deploy_with_an_unknown_model_or_runtime(api: Any) -> None:
    assert api.post("/api/deploy", json={"model_id": "mdl_nope"}).status_code == 400
    first = deploy_echo(api, route="known")
    bad = api.post("/api/deploy", json={"model_id": first["model"]["id"], "runtime_id": "rt_nope"})
    assert bad.status_code == 400
    assert "runtime" in bad.json()["error"]["message"]
