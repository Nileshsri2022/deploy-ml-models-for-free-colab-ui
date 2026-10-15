"""Runtime supervision: registration, heartbeats, desired state, and the reconciler.

This module is what makes the Colab VM disposable. It:

* tracks liveness from heartbeats and tunnel state,
* computes each runtime's **desired state** (which model versions it should have
  loaded) so a brand-new VM re-arms itself automatically,
* degrades ``online -> stale -> offline`` on missed heartbeats,
* warns before Colab's ~12 h session cap and produces the one-click relaunch URL,
* drains the persisted job queue when a runtime returns,
* sends hot cache keys as warm-up samples on reconnect (architecture.md §8).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from colabdeploy_shared.enums import DeploymentState, RuntimeState
from colabdeploy_shared.protocol import HelloFrame, HeartbeatFrame, AckFrame, LogFrame
from colabdeploy_shared.schemas import (
    DeploymentInfo,
    DesiredDeployment,
    DesiredState,
    JobInfo,
    RuntimeCapabilities,
    RuntimeInfo,
    utcnow,
)
from gateway.core.cache import CacheBackend
from gateway.core.config import Settings
from gateway.core.logging import Telemetry
from gateway.core.repository import Registry
from gateway.services.relay import Relay, RelayConnection


class JobDispatcher(Protocol):
    """Implemented by the inference service; called to drain queued jobs."""

    async def dispatch_job(self, job: JobInfo) -> bool: ...


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class RuntimeManager:
    def __init__(
        self,
        registry: Registry,
        relay: Relay,
        settings: Settings,
        telemetry: Telemetry,
        cache: CacheBackend,
    ) -> None:
        self.registry = registry
        self.relay = relay
        self.settings = settings
        self.telemetry = telemetry
        self.cache = cache
        self.dispatcher: JobDispatcher | None = None
        self.sse_hub: Any = None
        self._task: asyncio.Task[None] | None = None
        self._desired_revision = 0
        self._last_revision_push: dict[str, int] = {}
        self._supervise_interval_s = 5.0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    async def start_async(self) -> None:
        """Begin periodic supervision. Called from the app lifespan."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._supervise_loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ------------------------------------------------------------------ #
    # Agent connect / disconnect
    # ------------------------------------------------------------------ #
    async def on_hello(self, conn: RelayConnection, hello: HelloFrame) -> RuntimeInfo:
        record = self.registry.store.get("runtimes", conn.runtime_id) or {}
        caps = RuntimeCapabilities.model_validate(
            {**record.get("capabilities", {}), **hello.capabilities}
        )
        session_started = _parse_dt(hello.session_started_at) or _parse_dt(
            record.get("session_started_at")
        ) or utcnow()
        patch: dict[str, Any] = {
            "state": RuntimeState.CONNECTING.value,
            "capabilities": caps.model_dump(mode="json"),
            "tunnel_kind": "ws-relay",
            "tunnel_url": hello.capabilities.get("tunnel_url"),
            "session_started_at": session_started.isoformat(),
            "last_heartbeat_at": utcnow().isoformat(),
            "error": None,
        }
        updated = self.registry.store.update("runtimes", conn.runtime_id, patch)
        info = RuntimeInfo.model_validate(updated or {**record, **patch, "id": conn.runtime_id})
        self.telemetry.log(
            f"runtime connected ({caps.gpu_name or 'CPU'} · agent {hello.agent_version})",
            runtime_id=conn.runtime_id,
            gpu=caps.gpu,
        )
        return info

    async def on_heartbeat(self, frame: HeartbeatFrame) -> None:
        runtime_id = frame.runtime_id
        record = self.registry.store.get("runtimes", runtime_id)
        if record is None:
            return
        caps = RuntimeCapabilities.model_validate(
            {**record.get("capabilities", {}), **frame.capabilities.model_dump(mode="json")}
        )
        state = RuntimeState.ONLINE
        if frame.state in (RuntimeState.DEGRADED.value, RuntimeState.DEGRADED):
            state = RuntimeState.DEGRADED
        self.registry.store.update(
            "runtimes",
            runtime_id,
            {
                "state": state.value,
                "capabilities": caps.model_dump(mode="json"),
                "loaded_models": frame.loaded_models,
                "last_heartbeat_at": utcnow().isoformat(),
                "session_started_at": (
                    frame.session_started_at.isoformat()
                    if frame.session_started_at
                    else record.get("session_started_at")
                ),
                "queue_depth": frame.queue_depth,
                "inflight": frame.inflight,
                "metrics": frame.metrics,
                "error": record.get("error"),
            },
        )
        # Reflect per-deployment state reported by the agent.
        for dep in frame.loaded_models:
            self._mark_deployment_ready_by_key(runtime_id, dep)

    async def on_ack(self, frame: AckFrame) -> None:
        """The agent acks a desired-state push with per-deployment results."""
        for item in frame.deployment_states:
            dep_id = str(item.get("deployment_id", ""))
            if not dep_id:
                continue
            state = str(item.get("state", DeploymentState.READY.value))
            self.registry.store.update(
                "deployments",
                dep_id,
                {
                    "state": state,
                    "error": item.get("error"),
                    "warm": bool(item.get("warm", state == DeploymentState.READY.value)),
                    "warmup_ms": item.get("warmup_ms"),
                    "runtime_id": frame.runtime_id,
                    "updated_at": utcnow().isoformat(),
                },
            )
        level = "error" if frame.message and "fail" in frame.message.lower() else "info"
        self.telemetry.log(
            frame.message or "desired state acked",
            level=level,
            runtime_id=frame.runtime_id,
            deployments=len(frame.deployment_states),
        )

    async def on_agent_log(self, frame: LogFrame) -> None:
        self.telemetry.ingest_agent_logs(frame.runtime_id, frame.lines)
        for line in frame.lines[-20:]:
            self.telemetry_hub_publish(line)

    def telemetry_hub_publish(self, line: dict[str, Any]) -> None:
        hub = self.sse_hub
        if hub is None:
            return
        from colabdeploy_shared.schemas import LogLine

        with contextlib.suppress(Exception):
            hub.publish(LogLine.model_validate(line))

    async def on_disconnect(self, runtime_id: str, reason: str | None) -> None:
        record = self.registry.store.get("runtimes", runtime_id)
        if record is None:
            return
        self.registry.store.update(
            "runtimes",
            runtime_id,
            {"state": RuntimeState.OFFLINE.value, "error": reason, "loaded_models": []},
        )
        for dep in self.registry.deployments.for_runtime(runtime_id):
            if dep.state in (DeploymentState.READY, DeploymentState.LOADING):
                self.registry.store.update(
                    "deployments",
                    dep.id,
                    {
                        "state": DeploymentState.PENDING.value,
                        "warm": False,
                        "updated_at": utcnow().isoformat(),
                    },
                )
        self.telemetry.log(
            f"runtime disconnected: {reason or 'tunnel closed'}",
            level="warning",
            runtime_id=runtime_id,
        )

    def _mark_deployment_ready_by_key(self, runtime_id: str, model_key: str) -> None:
        """``loaded_models`` entries are ``model_id@version``; flip matching deployments."""
        model_id, _, version = model_key.partition("@")
        for dep in self.registry.deployments.for_runtime(runtime_id):
            if dep.model_id == model_id and str(dep.model_version) in ("", version) and (
                version == "" or dep.model_version == int(version or 0)
            ):
                if dep.state != DeploymentState.READY:
                    self.registry.store.update(
                        "deployments",
                        dep.id,
                        {
                            "state": DeploymentState.READY.value,
                            "runtime_id": runtime_id,
                            "warm": True,
                            "updated_at": utcnow().isoformat(),
                        },
                    )

    # ------------------------------------------------------------------ #
    # Desired state
    # ------------------------------------------------------------------ #
    def desired_state_for(self, runtime_id: str) -> DesiredState:
        """Everything this VM should be serving, newest-first."""
        wanted: list[DesiredDeployment] = []
        for dep in self.registry.deployments.list():
            if dep.runtime_id not in (None, runtime_id):
                continue
            if dep.state in (DeploymentState.FAILED, DeploymentState.UNLOADED):
                continue
            model = self.registry.models.get(dep.model_id)
            if model is None:
                continue
            wanted.append(
                DesiredDeployment(
                    deployment_id=dep.id,
                    endpoint_id=dep.endpoint_id,
                    model_id=model.id,
                    model_version=model.version,
                    adapter=model.adapter,
                    artifact=model.artifact,
                    entrypoint=model.entrypoint,
                    schema_in=model.schema_in,
                    schema_out=model.schema_out,
                    warmup_sample=model.warmup_sample,
                    state=dep.state,
                )
            )
        wanted.sort(key=lambda d: (d.model_version, d.model_id), reverse=True)
        self._desired_revision = max(self._desired_revision, self.registry.revision)
        return DesiredState(
            runtime_id=runtime_id,
            revision=self._desired_revision,
            deployments=wanted,
            warmup_top_k=20,
            settings={
                "heartbeat_interval_s": self.settings.heartbeat_interval_s,
                "session_max_hours": self.settings.session_max_hours,
                "max_body_bytes": self.settings.max_body_bytes,
            },
        )

    async def push_desired_state(self, runtime_id: str, *, with_warmup: bool = False) -> bool:
        state = self.desired_state_for(runtime_id)
        payload = [d.model_dump(mode="json") for d in state.deployments]
        ok = await self.relay.push_desired_state(
            runtime_id, state.revision, payload, state.warmup_top_k, state.settings
        )
        if ok:
            self._last_revision_push[runtime_id] = state.revision
        if ok and with_warmup:
            samples = [
                s
                for s in self.cache.hot_keys(limit=state.warmup_top_k)
                if s.get("sample") is not None
            ]
            if samples:
                await self.relay.push_warmup(
                    runtime_id, [{"payload": s["sample"], "endpoint_id": s["endpoint_id"]} for s in samples]
                )
        return ok

    # ------------------------------------------------------------------ #
    # Runtime selection
    # ------------------------------------------------------------------ #
    def pick_runtime(self, endpoint_id: str, model_id: str, model_version: int) -> str | None:
        """Choose a live runtime for a request.

        Preference order:
        1. the runtime that already has this deployment READY (hottest),
        2. any connected runtime that reports the model loaded,
        3. any connected runtime, picked by consistent hash on the endpoint id
           (stable affinity keeps per-runtime caches warm),
        4. ``None`` -> the caller applies the endpoint's degradation mode.
        """
        ready = [
            d
            for d in self.registry.deployments.for_endpoint(endpoint_id)
            if d.model_id == model_id
            and d.model_version == model_version
            and d.state == DeploymentState.READY
            and d.runtime_id
            and self.relay.is_connected(d.runtime_id)
        ]
        if ready:
            return ready[0].runtime_id

        connected = self.relay.connected_runtime_ids
        if not connected:
            return None

        key = f"{model_id}@{model_version}"
        for runtime_id in connected:
            record = self.registry.store.get("runtimes", runtime_id) or {}
            if key in (record.get("loaded_models") or []):
                return runtime_id

        digest = int(hashlib.sha256(f"{endpoint_id}:{runtime_seed(connected)}".encode()).hexdigest(), 16)
        return connected[digest % len(connected)]

    # ------------------------------------------------------------------ #
    # Supervision loop
    # ------------------------------------------------------------------ #
    async def _supervise_loop(self) -> None:
        """Periodic liveness, session-expiry warnings, job draining, cache saving."""
        while True:
            try:
                await self.supervise_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - supervision must never die
                self.telemetry.log(f"supervisor error: {exc}", level="error")
            await asyncio.sleep(self._supervise_interval_s)

    async def supervise_once(self) -> dict[str, Any]:
        report: dict[str, Any] = {"stale": [], "offline": [], "session_warning": [], "drained": 0}
        now = utcnow()

        for runtime in self.registry.runtimes.list():
            connected = self.relay.is_connected(runtime.id)
            last = _parse_dt(runtime.last_heartbeat_at)
            gap = (now - last).total_seconds() if last else float("inf")

            new_state = runtime.state
            if not connected:
                new_state = (
                    RuntimeState.STALE
                    if gap <= self.settings.heartbeat_offline_s
                    else RuntimeState.OFFLINE
                )
                if runtime.state in (RuntimeState.ONLINE, RuntimeState.CONNECTING, RuntimeState.DEGRADED):
                    new_state = RuntimeState.STALE
            elif gap > self.settings.heartbeat_stale_s:
                new_state = RuntimeState.STALE
            elif runtime.state in (RuntimeState.STALE, RuntimeState.OFFLINE, RuntimeState.CONNECTING):
                new_state = RuntimeState.ONLINE

            patch: dict[str, Any] = {}
            if new_state != runtime.state:
                patch["state"] = new_state.value
                if new_state == RuntimeState.OFFLINE and runtime.state != RuntimeState.OFFLINE:
                    self.telemetry.log(
                        f"runtime marked offline after {gap:.0f}s without heartbeat",
                        level="warning",
                        runtime_id=runtime.id,
                    )
                    report["offline"].append(runtime.id)
                elif new_state == RuntimeState.STALE:
                    report["stale"].append(runtime.id)

            # Session-cap watchdog: Colab kills the VM at ~12 h no matter what.
            started = _parse_dt(runtime.session_started_at)
            if started and connected:
                remaining = self.settings.session_max_hours * 3600 - (now - started).total_seconds()
                if 0 < remaining <= self.settings.session_warn_minutes * 60:
                    report["session_warning"].append({"runtime_id": runtime.id, "seconds": int(remaining)})
                    if not runtime.error or "session" not in str(runtime.error):
                        patch["error"] = (
                            f"Colab session ends in ~{int(remaining // 60)} min - relaunch to stay online"
                        )
                elif remaining <= 0 and not str(runtime.error or "").startswith("session expired"):
                    patch["error"] = "session expired (Colab max lifetime reached)"
            if patch:
                self.registry.store.update("runtimes", runtime.id, patch)

        # Drain queued jobs when capacity is back.
        if self.dispatcher is not None and self.relay.connected_runtime_ids:
            for job in self.registry.jobs.queued()[:32]:
                if await self.dispatcher.dispatch_job(job):
                    report["drained"] += 1

        # Expire old jobs so the queue cannot grow forever.
        cutoff = now - timedelta(seconds=self.settings.job_ttl_s)
        for job in self.registry.jobs.list():
            if job.state.value == "queued" and job.created_at < cutoff:
                self.registry.store.update(
                    "jobs",
                    job.id,
                    {
                        "state": "expired",
                        "error": "expired before a runtime became available",
                        "done_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                    },
                )
        self.registry.jobs.prune(now - timedelta(days=7))

        # Periodic cache snapshot: the resilience buffer survives gateway restarts.
        if int(time.time()) % 60 < int(self._supervise_interval_s) + 1:
            with contextlib.suppress(Exception):
                save = getattr(self.cache, "save", None)
                if callable(save):
                    save()
        return report

    # ------------------------------------------------------------------ #
    # Read models for the UI
    # ------------------------------------------------------------------ #
    def runtime_view(self, runtime: RuntimeInfo) -> RuntimeInfo:
        now = utcnow()
        last = _parse_dt(runtime.last_heartbeat_at)
        started = _parse_dt(runtime.session_started_at)
        runtime.seconds_since_heartbeat = round((now - last).total_seconds(), 1) if last else None
        if started:
            ends = started + timedelta(hours=runtime.session_max_hours)
            runtime.seconds_until_session_end = round((ends - now).total_seconds(), 1)
        runtime.deployments = self.registry.deployments.for_runtime(runtime.id)
        return runtime

    def relaunch_url(self, runtime: RuntimeInfo | None = None) -> str:
        """Deep link that re-opens the agent notebook with the token pre-filled."""
        base = self.settings.colab_relaunch_url
        token_hint = ""
        if runtime is not None:
            record = self.registry.store.get("runtimes", runtime.id) or {}
            token_hint = f"&runtimeId={runtime.id}&tokenPrefix={record.get('token_prefix', '')}"
        return f"{base}?tunnel=true{token_hint}"


def runtime_seed(ids: list[str]) -> str:
    """Stable ordering key so consistent hashing does not reshuffle on list churn."""
    return ",".join(sorted(ids))
