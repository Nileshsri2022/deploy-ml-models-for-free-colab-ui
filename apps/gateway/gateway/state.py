"""Application container: one place where every singleton is built and wired.

Routes and tests both go through :func:`get_state`, so there is exactly one
composition root. ``build_state`` lets a test construct a fully wired instance
against a temp directory without touching module globals.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colabdeploy_shared.schemas import LogLine
from gateway.core.cache import CacheBackend, build_cache
from gateway.core.config import Settings, get_settings, reset_settings
from gateway.core.logging import SseHub, Telemetry
from gateway.core.redis_store import RedisSnapshotStore, build_registry_store, registry_status
from gateway.core.repository import COLLECTIONS, Registry
from gateway.core.sentry import initialize_sentry
from gateway.services.deployments import DeploymentService
from gateway.services.inference import InferenceService
from gateway.services.quotas import DailyInferenceQuota
from gateway.services.relay import Relay
from gateway.services.runtime_manager import RuntimeManager
from gateway.services.sentry_issues import SentryIssueService


class AppState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._event_loop: asyncio.AbstractEventLoop | None = None
        settings.ensure_dirs()
        # Initialize before the Redis mirror starts so first-boot failures can
        # be reported safely when Sentry is configured.
        initialize_sentry(settings)
        self.telemetry = Telemetry()
        self.telemetry.logs.capacity = settings.log_buffer_lines
        self.telemetry.metrics.capacity = settings.metrics_buffer_points
        self.sse = SseHub()
        redis_enabled = (settings.registry_backend or "file").strip().lower() == "redis"
        self.registry_mirror_health: dict[str, Any] = {
            "enabled": redis_enabled,
            "status": (
                "misconfigured"
                if redis_enabled and not settings.redis_url
                else "connecting"
                if redis_enabled
                else "disabled"
            ),
            "last_error": None,
            "last_error_at": None,
            "last_success_at": None,
        }
        # The registry survives container recycling by mirroring to Redis when
        # CDP_REGISTRY_BACKEND=redis; otherwise it is the plain file store.
        self.store = build_registry_store(settings, event_callback=self._on_registry_event)
        self.registry_restored = 0
        # Health of the mirror (connected / disabled / unreachable + why), for
        # /healthz and the Cache page. Refreshed on startup and on demand.
        self.registry_status: dict[str, Any] = {}
        if isinstance(self.store, RedisSnapshotStore):
            self.registry_restored = self.store.warm([*COLLECTIONS, "inference_usage"])
        self.registry = Registry(self.store)
        self.inference_quota = DailyInferenceQuota(self.store, settings.daily_inference_quota)
        self.sentry_issues = SentryIssueService(
            token=settings.sentry_read_token,
            org_slug=settings.sentry_org_slug,
            project_slug=settings.sentry_project_slug,
            api_base_url=settings.sentry_api_base_url,
        )
        self.cache: CacheBackend = build_cache(
            settings.cache_backend,
            data_dir=settings.data_dir,
            max_entries=settings.cache_max_entries,
            max_bytes=settings.cache_max_bytes,
            persist=settings.cache_persist,
        )
        self.relay = Relay()
        self.runtimes = RuntimeManager(
            self.registry, self.relay, settings, self.telemetry, self.cache
        )
        self.runtimes.sse_hub = self.sse
        self.deployments = DeploymentService(self.registry, settings, self.telemetry, self.runtimes)
        self.inference = InferenceService(
            self.registry, self.relay, self.cache, settings, self.telemetry, self.runtimes
        )
        # The supervisor drains queued jobs through the inference service.
        self.runtimes.dispatcher = self.inference

        # Relay callbacks -> runtime manager -> telemetry/SSE.
        self.relay.on_heartbeat = self.runtimes.on_heartbeat
        self.relay.on_ack = self.runtimes.on_ack
        self.relay.on_log = self._on_agent_log
        self.relay.on_disconnect = self.runtimes.on_disconnect

        # Cache warm load: a gateway restart keeps its resilience buffer.
        self.cache_loaded_entries = (
            getattr(self.cache, "load", lambda: 0)()
            if settings.cache_persist and settings.cache_backend == "memory"
            else 0
        )

    def _on_registry_event(self, message: str, level: str, data: dict[str, Any]) -> None:
        """Route Redis mirror health events into the existing Logs UI/SSE."""
        if data.get("component") == "upstash":
            now = datetime.now(timezone.utc).isoformat()
            if level == "warning":
                self.registry_mirror_health.update(
                    status="misconfigured"
                    if data.get("operation") == "configuration"
                    else "disconnected",
                    last_error=data.get("reason"),
                    last_error_at=now,
                )
            elif data.get("status") == "connected":
                self.registry_mirror_health.update(
                    status="connected",
                    last_error=None,
                    last_error_at=None,
                    last_success_at=now,
                )
        self.publish_log(
            LogLine(
                level=level,
                source="gateway",
                message=message,
                data=data or None,
            )
        )

    async def _on_agent_log(self, frame: Any) -> None:
        await self.runtimes.on_agent_log(frame)

    # ------------------------------------------------------------------ #
    def publish_log(self, line: Any) -> None:
        """Append to the ring *and* fan out to SSE subscribers safely."""
        self.telemetry.logs.append(line)
        loop = self._event_loop
        if loop is not None and loop.is_running():
            # Redis sync runs on a background thread; SseHub queues belong to
            # the ASGI loop and must only be mutated from that loop.
            loop.call_soon_threadsafe(self.sse.publish, line)
        else:
            self.sse.publish(line)

    async def startup(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        await self.runtimes.start_async()
        mirror = "registry restored from Redis" if self.registry_restored else ""
        status = registry_status(self.store, self.settings)
        if not status["connected"]:
            # Say once at boot what the Cache page will keep repeating: a
            # missing dependency, an empty env var, or an unreachable database.
            mirror = f"registry mirror {status['state']}: {status.get('message') or status.get('last_error') or ''}".strip()
        self.telemetry.log(
            f"control plane up (cache={self.cache.name}, data_dir={self.settings.data_dir})"
            + (f" [{mirror}]" if mirror else ""),
            data={"version": self.settings.version, "env": self.settings.env},
        )
        self.registry_status = status

    async def shutdown(self) -> None:
        await self.runtimes.stop()
        await self.relay.close_all("gateway shutdown")
        await self.sentry_issues.close()
        self.cache.close()
        self.store.close()  # no-op for the file store; flushes the Redis mirror
        self.telemetry.log("control plane stopped")


_state: AppState | None = None


def build_state(settings: Settings | None = None) -> AppState:
    """Construct a fresh, fully wired state (used by tests and by ``lifespan``)."""
    return AppState(settings or get_settings())


def set_state(state: AppState | None) -> None:
    global _state
    _state = state


def get_state() -> AppState:
    """FastAPI dependency + module accessor."""
    global _state
    if _state is None:
        _state = AppState(get_settings())
    return _state


def reset_state(data_dir: str | Path | None = None, **overrides: Any) -> AppState:
    """Rebuild settings + state; the test harness entry point."""
    if data_dir is not None:
        overrides["data_dir"] = Path(data_dir)
    settings = reset_settings(**overrides)
    state = AppState(settings)
    set_state(state)
    return state


# Convenience accessors used by a few helpers that cannot take a dependency.
def get_registry() -> Registry:
    return get_state().registry


def get_cache() -> CacheBackend:
    return get_state().cache


def get_telemetry() -> Telemetry:
    return get_state().telemetry
