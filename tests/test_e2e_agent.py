"""End-to-end: a **real agent** in a thread, a **real uvicorn server**, one tunnel.

This is the M1 spike from ``architecture.md`` turned into a regression test - the
single experiment that de-risks the whole project:

    laptop ──HTTP──▶ gateway ══WS══▶ agent thread ──▶ adapter ──▶ JSON back

It covers the handshake, desired-state reconciliation, inference through the
tunnel, the artifact disk cache, and - the point of the whole design - what the
caller sees when the Colab VM disappears and comes back.

These tests are deliberately **synchronous**: the agent owns its own thread and
event loop, so blocking here cannot deadlock the server's loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
import uvicorn

pytestmark = pytest.mark.e2e

from colabdeploy.artifacts import ArtifactCache, ArtifactResolver  # noqa: E402
from colabdeploy.bootstrap import bootstrap  # noqa: E402
from colabdeploy.model_store import ModelStore  # noqa: E402
from colabdeploy.registry_client import GatewayClient  # noqa: E402
from colabdeploy.runtime import AgentRuntime  # noqa: E402
from colabdeploy.state import make_reconciler  # noqa: E402
from colabdeploy.tunnel import TunnelClient, TunnelConfig, ws_url_from_http  # noqa: E402

from tests.support import TEST_ADMIN_KEY, wait_for  # noqa: E402


# --------------------------------------------------------------------------- #
# Live server
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Server(uvicorn.Server):
    def install_signal_handlers(self) -> None:  # pragma: no cover - not a main thread
        return None


@pytest.fixture()
def live(state: Any) -> Iterator[dict[str, Any]]:
    """Boot the gateway on a real port, sharing the fixture's state object."""
    from gateway.main import create_app

    port = _free_port()
    app = create_app(state)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", ws_ping_interval=None
    )
    server = _Server(config)
    thread = threading.Thread(target=server.run, name="cdp-gateway", daemon=True)
    thread.start()

    deadline = time.time() + 20.0
    while time.time() < deadline:
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "gateway did not start in time"

    base = f"http://127.0.0.1:{port}"
    yield {"base_url": base, "port": port, "server": server, "state": state}

    server.should_exit = True
    thread.join(timeout=10.0)


# --------------------------------------------------------------------------- #
# A real agent, minus the notebook
# --------------------------------------------------------------------------- #
class LocalAgent:
    """Everything ``colabdeploy.launch()`` builds, pointed at the live server."""

    def __init__(self, base_url: str, workdir: Path, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.env = bootstrap(workdir, quiet=True)
        self.client = GatewayClient(self.base_url, token)
        self.cache = ArtifactCache(self.env.cache_dir)
        self.resolver = ArtifactResolver(
            self.cache, gateway_url=self.base_url, runtime_token=token
        )
        self.store = ModelStore(self.resolver, self.cache, max_loaded=4)
        self.runtime_id = ""
        self.tunnel: TunnelClient | None = None
        self.registration: Any = None

    def connect(self, timeout_s: float = 20.0) -> str:
        capabilities = self.env.capabilities("0.1.0-test", ["echo", "python_function"]).model_dump(
            mode="json"
        )
        self.registration = self.client.register(capabilities, None)
        self.runtime_id = self.registration.runtime_id
        self.runtime = AgentRuntime(
            self.env, self.store, runtime_id=self.runtime_id, agent_version="0.1.0-test"
        )
        self.tunnel = TunnelClient(
            TunnelConfig(
                relay_url=self.registration.relay_ws_url or ws_url_from_http(self.base_url),
                runtime_id=self.runtime_id,
                token=self.token,
                agent_version="0.1.0-test",
                heartbeat_interval_s=1.0,
                reconnect_min_s=0.2,
                reconnect_max_s=1.0,
            ),
            self.env,
            handle_exec=self.runtime.handle_exec,
            status_fn=self.runtime.status,
            on_desired_state=make_reconciler(self.store, self.runtime_id),
            log=lambda *a, **kw: None,
        )
        self.tunnel.start()
        assert self.tunnel.wait_connected(timeout_s=timeout_s), "agent tunnel never came up"
        return self.runtime_id

    def disconnect(self) -> None:
        """Simulate Colab killing the VM."""
        if self.tunnel is not None:
            self.tunnel.stop(timeout_s=5.0)

    def close(self) -> None:
        self.disconnect()
        self.runtime.close()
        self.client.close()


@pytest.fixture()
def agent_factory(live: dict[str, Any], tmp_path: Path) -> Iterator[Any]:
    created: list[LocalAgent] = []

    def make(api: Any, label: str = "agent") -> LocalAgent:
        token = api.post(f"/api/runtimes/token?label={label}&ttl_hours=1").json()["token"]
        agent = LocalAgent(live["base_url"], tmp_path / label, token)
        created.append(agent)
        return agent

    yield make
    for agent in created:
        with contextlib.suppress(Exception):
            agent.close()


@pytest.fixture()
def api(live: dict[str, Any]) -> Any:
    import httpx

    client = httpx.Client(base_url=live["base_url"], timeout=30.0,
                          headers={"Authorization": f"Bearer {TEST_ADMIN_KEY}"})
    yield client
    client.close()


def deploy_echo(api: Any, artifact_path: Path, *, name: str = "echo-model",
                runtime_id: str | None = None, route: str | None = None,
                degradation: str = "serve_cached", cache_ttl_s: int = 300,
                swr_s: int = 3600) -> dict[str, Any]:
    model = api.post(
        "/api/models",
        json={
            "name": name,
            "adapter": "echo",
            "artifact": {"uri": f"file://{artifact_path}", "filename": artifact_path.name},
            "schema_in": {"fields": {"text": {"type": "string"}}, "example": {"text": "hi"}},
            "warmup_sample": {"text": "warm-up"},
        },
    ).json()
    return api.post(
        "/api/deploy",
        json={
            "model_id": model["id"],
            "runtime_id": runtime_id,
            "endpoint": {
                "route": route or name,
                "degradation_mode": degradation,
                "cache_ttl_s": cache_ttl_s,
                "stale_while_revalidate_s": swr_s,
            },
        },
    ).json()


def wait_until_ready(api: Any, endpoint_id: str, timeout_s: float = 25.0) -> dict[str, Any]:
    def ready() -> Any:
        for dep in api.get("/api/deployments").json()["deployments"]:
            if dep["endpoint_id"] == endpoint_id and dep["state"] == "ready":
                return dep
        return None

    wait_for(ready, timeout_s=timeout_s, message=f"deployment {endpoint_id} to become ready")
    return api.get("/api/endpoints").json()["endpoints"][0]


# --------------------------------------------------------------------------- #
# The tests
# --------------------------------------------------------------------------- #
def test_agent_connects_and_registers(live: dict[str, Any], api: Any, agent_factory: Any) -> None:
    agent = agent_factory(api)
    runtime_id = agent.connect()

    wait_for(
        lambda: api.get("/api/runtimes").json()["runtimes"][0]["connected"],
        timeout_s=10,
        message="gateway to report the tunnel as connected",
    )
    runtime = api.get(f"/api/runtimes/{runtime_id}").json()
    assert runtime["connected"] is True
    assert runtime["state"] in ("online", "connecting")
    assert runtime["tunnel_kind"] == "ws-relay"
    assert runtime["capabilities"]["cpu_count"]
    assert "echo" in runtime["capabilities"]["adapters"]

    # The agent heartbeats on its own; the gateway must observe it.
    wait_for(
        lambda: api.get(f"/api/runtimes/{runtime_id}").json()["last_heartbeat_at"],
        timeout_s=10,
        message="a heartbeat",
    )


def test_full_round_trip_through_the_tunnel(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """Deploy -> VM loads the model -> a real HTTP call is served by the VM."""
    artifact = tmp_path / "sentiment.echo.json"
    artifact.write_text(json.dumps({"latency_ms": 0}), encoding="utf-8")

    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="sentiment", runtime_id=runtime_id)
    endpoint_id = result["endpoint"]["id"]

    # The gateway pushed desired state; the agent hydrated and acked.
    wait_until_ready(api, endpoint_id)
    assert agent.store.keys == [f"{result['model']['id']}@1"]

    response = api.post("/v1/models/sentiment/predict", json={"inputs": {"text": "I love this"}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cache"] == "MISS"
    assert body["runtime_id"] == runtime_id
    assert body["output"]["echo"] == {"text": "I love this"}
    assert body["output"]["adapter"] == "echo"
    assert response.headers["X-ColabDeploy-Cache"] == "MISS"

    # Second identical call is served by the control plane, never touching the VM.
    before = agent.runtime.counters["executions"]
    cached = api.post("/v1/models/sentiment/predict", json={"inputs": {"text": "I love this"}})
    assert cached.json()["cache"] == "HIT"
    assert cached.json()["output"] == body["output"]
    assert agent.runtime.counters["executions"] == before, "a cache hit must not reach the VM"

    # A different payload does reach it.
    other = api.post("/v1/models/sentiment/predict", json={"inputs": {"text": "awful"}})
    assert other.json()["cache"] == "MISS"
    assert agent.runtime.counters["executions"] == before + 1


def test_model_output_is_deterministic_so_cache_is_correct(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    artifact = tmp_path / "det.echo.json"
    artifact.write_text("{}", encoding="utf-8")
    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="deterministic", runtime_id=runtime_id)
    wait_until_ready(api, result["endpoint"]["id"])

    payload = {"inputs": {"text": "same", "n": 3}}
    first = api.post("/v1/models/deterministic/predict", json=payload).json()
    api.post("/api/cache/invalidate", params={"endpoint_id": result["endpoint"]["id"]})
    second = api.post("/v1/models/deterministic/predict", json=payload).json()
    assert first["output"]["sha256"] == second["output"]["sha256"]
    assert first["cache"] == "MISS" and second["cache"] == "MISS"


def test_disconnect_serves_stale_then_reconnect_rearms(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """The resilience guarantee, end to end, with a real socket being torn down."""
    artifact = tmp_path / "survivor.echo.json"
    artifact.write_text("{}", encoding="utf-8")

    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="survivor", runtime_id=runtime_id,
                         cache_ttl_s=1, swr_s=3600)
    endpoint_id = result["endpoint"]["id"]
    wait_until_ready(api, endpoint_id)

    warm = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "great"}})
    assert warm.json()["cache"] == "MISS"
    assert warm.json()["output"]["echo"] == {"text": "great"}
    time.sleep(1.1)  # freshness lapses; the entry stays usable-but-stale

    # --- Colab dies -------------------------------------------------------
    agent.disconnect()
    wait_for(
        lambda: api.get(f"/api/runtimes/{runtime_id}").json()["connected"] is False,
        timeout_s=10,
        message="the gateway to notice the tunnel closed",
    )
    wait_for(
        lambda: api.get(f"/api/runtimes/{runtime_id}").json()["state"] == "offline",
        timeout_s=10,
        message="the runtime to be marked offline",
    )

    offline = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "great"}})
    assert offline.status_code == 200, "the caller must never see the VM's death"
    body = offline.json()
    assert body["cache"] == "STALE" and body["stale"] is True
    assert body["output"]["echo"] == {"text": "great"}
    assert offline.headers["X-ColabDeploy-Stale"] == "true"

    # An unseen payload cannot be invented - it is queued, not dropped.
    queued = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "brand new"}})
    assert queued.status_code == 202
    job_id = queued.json()["job_id"]
    assert api.get(f"/v1/jobs/{job_id}").json()["state"] == "queued"

    # --- Colab comes back -------------------------------------------------
    agent.connect()
    wait_until_ready(api, endpoint_id, timeout_s=30)

    fresh = api.post("/v1/models/survivor/predict", json={"inputs": {"text": "brand new"}})
    assert fresh.json()["cache"] == "MISS"
    assert fresh.json()["output"]["echo"] == {"text": "brand new"}

    wait_for(
        lambda: api.get(f"/v1/jobs/{job_id}").json()["state"] == "succeeded",
        timeout_s=30,
        message="the queued job to drain",
    )
    assert api.get(f"/v1/jobs/{job_id}").json()["result"]["echo"] == {"text": "brand new"}


def test_reconnect_rehydrates_from_the_disk_cache(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """A session *resume* keeps /content, so the second load must not re-download."""
    artifact = tmp_path / "cached-artifact.echo.json"
    artifact.write_text("{}", encoding="utf-8")

    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="hydrated", runtime_id=runtime_id)
    wait_until_ready(api, result["endpoint"]["id"])
    assert agent.cache.stats()["entries"] == 1
    first_load = agent.store.snapshot()[0]
    assert first_load["from_cache"] is False

    # Same workspace, brand-new agent process = a Colab resume.
    agent.disconnect()
    resumed = LocalAgent(live["base_url"], agent.env.workdir, agent.token)
    resumed.connect()
    wait_until_ready(api, result["endpoint"]["id"], timeout_s=30)

    loads = [m for m in resumed.store.snapshot() if m["model_id"] == result["model"]["id"]]
    assert loads and loads[0]["from_cache"] is True, "the artifact cache must survive a resume"
    assert resumed.cache.stats()["hits"] >= 1


def test_a_broken_model_fails_loudly_without_killing_the_runtime(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """One bad artifact must not take down the other models or the tunnel."""
    broken = tmp_path / "broken.py"
    broken.write_text("def predict(payload)\n    return 1\n", encoding="utf-8")
    good = tmp_path / "good.echo.json"
    good.write_text("{}", encoding="utf-8")

    agent = agent_factory(api)
    runtime_id = agent.connect()

    bad_model = api.post("/api/models", json={
        "name": "broken", "adapter": "python_function",
        "artifact": {"uri": f"file://{broken}"},
        "entrypoint": broken.read_text(encoding="utf-8"),
    }).json()
    bad = api.post("/api/deploy", json={
        "model_id": bad_model["id"], "runtime_id": runtime_id,
        "endpoint": {"route": "broken", "degradation_mode": "fail"},
    }).json()

    def failed() -> Any:
        for dep in api.get("/api/deployments").json()["deployments"]:
            if dep["endpoint_id"] == bad["endpoint"]["id"] and dep["state"] == "failed":
                return dep
        return None

    wait_for(failed, timeout_s=25, message="the broken model to report failed")
    assert "syntax error" in (failed()["error"] or "")

    # The tunnel is still up and a second, good model still deploys.
    assert api.get(f"/api/runtimes/{runtime_id}").json()["connected"] is True
    ok = deploy_echo(api, good, name="healthy", runtime_id=runtime_id, route="healthy")
    wait_until_ready(api, ok["endpoint"]["id"])
    response = api.post("/v1/models/healthy/predict", json={"inputs": {"text": "still works"}})
    assert response.status_code == 200
    assert response.json()["output"]["echo"] == {"text": "still works"}

    # And calling the broken endpoint fails fast rather than hanging.
    broken_call = api.post("/v1/models/broken/predict", json={"inputs": {}})
    assert broken_call.status_code in (400, 502, 503)


def test_model_error_message_reaches_the_caller(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    artifact = tmp_path / "fussy.echo.json"
    artifact.write_text(json.dumps({"fail": "model exploded"}), encoding="utf-8")
    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="fussy", runtime_id=runtime_id,
                         degradation="fail")
    wait_until_ready(api, result["endpoint"]["id"])

    response = api.post("/v1/models/fussy/predict", json={"inputs": {"text": "x"}})
    assert response.status_code == 400
    assert "model exploded" in json.dumps(response.json())
    # A failing model must not poison the cache.
    assert api.get("/api/cache/stats").json()["entries"] == 0


def test_warmup_runs_the_deployers_sample(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    artifact = tmp_path / "warm.echo.json"
    artifact.write_text("{}", encoding="utf-8")
    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="warmable", runtime_id=runtime_id)
    wait_until_ready(api, result["endpoint"]["id"])
    loaded = agent.store.snapshot()[0]
    assert loaded["warm"] is True, "the deployer's warm-up sample should have been run"
    assert loaded["warmup_ms"] is not None
    assert agent.store.counters["warmups"] >= 1
    assert agent.store.counters["warmup_failures"] == 0
    # Warm-up must not be mistaken for served traffic.
    assert agent.runtime.counters["executions"] == 0


def test_custom_python_function_serves_real_traffic(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """The escape hatch that makes 'any model' true without a new adapter."""
    code = tmp_path / "rule.py"
    code.write_text(
        "def predict(payload, model=None, meta=None):\n"
        "    text = (payload.get('text') or '').lower()\n"
        "    score = sum(1 for w in ('great', 'love', 'amazing') if w in text)\n"
        "    return {'label': 'positive' if score else 'negative', 'score': score / 3.0}\n",
        encoding="utf-8",
    )
    agent = agent_factory(api)
    runtime_id = agent.connect()
    model = api.post("/api/models", json={
        "name": "rules", "adapter": "python_function",
        "artifact": {"uri": f"file://{code}"},
        "entrypoint": code.read_text(encoding="utf-8"),
        "schema_in": {"fields": {"text": {"type": "string"}}, "required": ["text"]},
    }).json()
    deployed = api.post("/api/deploy", json={
        "model_id": model["id"], "runtime_id": runtime_id, "endpoint": {"route": "rules"},
    }).json()
    wait_until_ready(api, deployed["endpoint"]["id"])

    positive = api.post("/v1/models/rules/predict", json={"inputs": {"text": "I love this, amazing"}})
    assert positive.json()["output"] == {"label": "positive", "score": pytest.approx(2 / 3)}
    negative = api.post("/v1/models/rules/predict", json={"inputs": {"text": "meh"}})
    assert negative.json()["output"]["label"] == "negative"


def test_artifact_uploaded_in_the_ui_is_pulled_by_the_agent(
    live: dict[str, Any], api: Any, agent_factory: Any
) -> None:
    """``store://`` artifacts: UI upload -> control plane -> VM."""
    agent = agent_factory(api)
    runtime_id = agent.connect()

    uploaded = api.post(
        "/api/models/upload",
        data={"name": "uploaded-model", "adapter": "echo"},
        files={"file": ("model.echo.json", b'{"latency_ms": 0}', "application/json")},
    ).json()
    assert uploaded["artifact"]["uri"].startswith("store://")

    deployed = api.post("/api/deploy", json={
        "model_id": uploaded["id"], "runtime_id": runtime_id,
        "endpoint": {"route": "uploaded-model"},
    }).json()
    wait_until_ready(api, deployed["endpoint"]["id"])

    response = api.post("/v1/models/uploaded-model/predict", json={"inputs": {"text": "hi"}})
    assert response.status_code == 200, response.text
    assert response.json()["output"]["echo"] == {"text": "hi"}

    # The bytes really did travel: gateway artifact store -> VM disk cache.
    assert agent.cache.stats()["entries"] == 1
    assert agent.cache.stats()["bytes"] == len(b'{"latency_ms": 0}')


def test_two_runtimes_can_serve_the_same_model(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """The multi-runtime pool (M5): failover needs both VMs to hold the model."""
    artifact = tmp_path / "pooled.echo.json"
    artifact.write_text("{}", encoding="utf-8")

    first = agent_factory(api, "vm-a")
    second = agent_factory(api, "vm-b")
    first.connect()
    result = deploy_echo(api, artifact, name="pooled", runtime_id=first.runtime_id)
    wait_until_ready(api, result["endpoint"]["id"])

    second.connect()
    # Arm the second VM with the same deployment.
    api.post("/api/deploy", json={
        "model_id": result["model"]["id"], "runtime_id": second.runtime_id,
        "endpoint_id": result["endpoint"]["id"],
    })
    wait_for(
        lambda: second.store.keys and second.store.keys[0].endswith("@1"),
        timeout_s=25,
        message="the second runtime to load the model",
    )

    assert api.post("/v1/models/pooled/predict", json={"inputs": {"text": "a"}}).status_code == 200

    # Kill VM A; VM B keeps the endpoint alive with no cache involved.
    api.post("/api/cache/invalidate", params={"endpoint_id": result["endpoint"]["id"]})
    first.disconnect()
    wait_for(
        lambda: api.get(f"/api/runtimes/{first.runtime_id}").json()["connected"] is False,
        timeout_s=10,
        message="VM A to drop",
    )
    # The deployment now points at VM B, which is online.
    api.post("/api/deploy", json={
        "model_id": result["model"]["id"], "runtime_id": second.runtime_id,
        "endpoint_id": result["endpoint"]["id"],
    })
    wait_until_ready(api, result["endpoint"]["id"], timeout_s=30)
    api.post("/api/cache/invalidate", params={"endpoint_id": result["endpoint"]["id"]})
    failover = api.post("/v1/models/pooled/predict", json={"inputs": {"text": "b"}})
    assert failover.status_code == 200, failover.text
    assert failover.json()["runtime_id"] == second.runtime_id
    assert failover.json()["cache"] == "MISS", "served fresh by the surviving VM"


def test_tunnel_recovers_from_a_gateway_restart(
    live: dict[str, Any], api: Any, agent_factory: Any, tmp_path: Path
) -> None:
    """The agent reconnects forever; a dropped socket is not a deploy failure."""
    artifact = tmp_path / "flaky.echo.json"
    artifact.write_text("{}", encoding="utf-8")
    agent = agent_factory(api)
    runtime_id = agent.connect()
    result = deploy_echo(api, artifact, name="flaky", runtime_id=runtime_id)
    wait_until_ready(api, result["endpoint"]["id"])

    tunnel = agent.tunnel
    assert tunnel is not None
    connects_before = tunnel.stats.connects

    # Close the socket from the agent side - the same thing a flaky network or a
    # Colab pre-emption does. The reconnect loop must recover without help.
    loop = tunnel._loop  # noqa: SLF001 - test-only inspection
    assert loop is not None
    asyncio.run_coroutine_threadsafe(tunnel._ws.close(code=1001), loop).result(timeout=10)  # noqa: SLF001

    wait_for(
        lambda: tunnel.stats.connects > connects_before,
        timeout_s=30,
        message="the agent to reconnect on its own",
    )
    wait_until_ready(api, result["endpoint"]["id"], timeout_s=30)
    assert api.post("/v1/models/flaky/predict", json={"inputs": {"text": "back"}}).json()[
        "cache"
    ] in ("MISS", "HIT")
