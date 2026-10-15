"""The agent's local inference server.

The tunnel dispatches ``exec`` frames **in-process** through an explicit route
table - no port is bound inside the Colab VM, so nothing is exposed on the
notebook's network, there is no second web server to supervise, and no dependency
on an HTTP client's transport internals.

The same routes are also mounted as a small FastAPI app (``self.app``), which is
purely a debugging/inspection affordance: you can run it under uvicorn in a
scratch cell and poke ``/health`` with curl.

Routes mirror what the gateway's ``exec`` frames ask for:

    POST /execute   run one prediction
    POST /warmup    prime models with hot cache keys
    POST /reconcile converge towards a desired state
    POST /unload    drop a model from the warm pool
    GET  /models    what is loaded right now
    GET  /health    agent + hardware + pool status
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI

from colabdeploy_shared.schemas import DesiredState

from .adapters import AdapterError
from .artifacts import ArtifactError
from .bootstrap import Environment
from .model_store import ModelStore


class AgentRuntime:
    """Owns the model pool and answers exec frames."""

    def __init__(
        self,
        env: Environment,
        store: ModelStore,
        *,
        runtime_id: str = "",
        agent_version: str = "0.1.0",
        session_started_at: float | None = None,
    ) -> None:
        self.env = env
        self.store = store
        self.runtime_id = runtime_id
        self.agent_version = agent_version
        self.session_started_at = session_started_at or env.session_started_at
        self.started_at = time.time()
        self.counters = {"executions": 0, "errors": 0, "reconciles": 0, "warmups": 0}
        self._routes: dict[str, Callable[[dict[str, Any]], Any]] = {
            "/execute": self.execute,
            "/warmup": self.warmup,
            "/reconcile": self.reconcile,
            "/unload": self.unload,
            "/models": lambda body: self.models(),
            "/health": lambda body: self.status(),
        }
        self.app = self._build_app()

    # ------------------------------------------------------------------ #
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="ColabDeploy Agent", version=self.agent_version, docs_url=None, openapi_url=None)

        from fastapi.responses import JSONResponse

        @app.api_route("/{path:path}", methods=["GET", "POST"])
        async def dispatch(path: str, body: dict[str, Any] | None = None) -> Any:
            """Mirror the in-process route table over HTTP (debugging only)."""
            status, payload = self.handle_exec(f"/{path}", body or {})
            if status == 200:
                return payload
            return JSONResponse(status_code=status, content=payload)

        return app

    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """Everything a heartbeat carries."""
        uptime = time.time() - self.session_started_at
        return {
            "ok": True,
            "runtime_id": self.runtime_id,
            "agent_version": self.agent_version,
            "state": "online" if self.store.keys or not self.store.failures() else "degraded",
            "capabilities": self.env.capabilities(self.agent_version, self.store.adapters_available).model_dump(
                mode="json"
            ),
            "environment": self.env.describe(),
            "loaded_models": self.store.keys,
            "models": self.store.snapshot(),
            "failures": self.store.failures(),
            "artifact_cache": self.store.cache.stats(),
            "counters": {**self.counters, "model_store": dict(self.store.counters)},
            "uptime_s": round(uptime, 1),
            "session_hours_remaining": round(max(0.0, 12.0 - uptime / 3600.0), 2),
            "python": platform.python_version(),
            "time": time.time(),
        }

    # ------------------------------------------------------------------ #
    # Route implementations (plain sync functions - no HTTP involved)
    # ------------------------------------------------------------------ #
    def execute(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Run one prediction. ``400`` = model/payload problem, ``500`` = agent bug."""
        payload = body.get("payload") or {}
        model_id = str(body.get("model_id") or "")
        version = int(body.get("model_version") or 0)
        endpoint_id = str(body.get("endpoint_id") or "")
        started = time.perf_counter()
        try:
            if model_id and version:
                result = self.store.predict(model_id, version, payload)
            elif endpoint_id:
                result = self.store.predict_for_endpoint(endpoint_id, payload)
            else:
                raise AdapterError("execute needs model_id+model_version or endpoint_id")
            self.counters["executions"] += 1
        except (AdapterError, ArtifactError) as exc:
            self.counters["errors"] += 1
            return 400, {
                "code": "MODEL_ERROR",
                "error": str(exc),
                "model_id": model_id,
                "loaded": self.store.keys,
                "agent_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            self.counters["errors"] += 1
            return 500, {
                "code": "AGENT_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "loaded": self.store.keys,
            }
        result["agent_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result["runtime_id"] = self.runtime_id
        return 200, result

    def warmup(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        samples = body.get("samples") or []
        warmed = 0
        for sample in samples if isinstance(samples, list) else []:
            if isinstance(sample, dict):
                self.store.warm_from_sample(sample)
                warmed += 1
        self.counters["warmups"] += warmed
        return 200, {"warmed": warmed, "loaded": self.store.keys}

    def reconcile(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.counters["reconciles"] += 1
        desired = DesiredState.model_validate(body.get("desired_state") or body)
        results = self.store.reconcile(desired, warmup_samples=body.get("warmup_samples"))
        return 200, {
            "revision": desired.revision,
            "results": [r.__dict__ for r in results],
            "loaded": self.store.keys,
            "failed": self.store.failures(),
        }

    def unload(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        key = str(body.get("key") or "")
        if key == "*":
            return 200, {"unloaded": self.store.unload_all(), "loaded": self.store.keys}
        return 200, {"unloaded": int(self.store.unload(key)), "loaded": self.store.keys}

    def models(self) -> dict[str, Any]:
        return {
            "count": len(self.store.keys),
            "models": self.store.snapshot(),
            "failures": self.store.failures(),
        }

    def handle_exec(self, path: str, body: dict[str, Any], method: str = "POST") -> tuple[int, dict[str, Any]]:
        """Synchronous dispatch used by the tunnel's worker thread."""
        handler = self._routes.get(path.split("?", 1)[0])
        if handler is None:
            return 404, {"code": "UNKNOWN_PATH", "error": f"agent has no route '{path}'",
                         "routes": sorted(self._routes)}
        try:
            result = handler(body if isinstance(body, dict) else {})
        except Exception as exc:
            self.counters["errors"] += 1
            return 500, {"code": "AGENT_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, tuple):
            status, payload = result
            return int(status), payload if isinstance(payload, dict) else {"result": payload}
        return 200, result if isinstance(result, dict) else {"result": result}

    def close(self) -> None:
        self.store.unload_all()
        self.store.cache.close()
