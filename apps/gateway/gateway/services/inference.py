"""The request path: validation → cache → runtime → cache → response.

This is where the resilience policy lives (architecture.md §6.2). One function,
:class:`InferenceService.predict`, implements the whole decision tree:

    fresh cache hit        -> 200 HIT                       (~ms, no VM involved)
    stale hit + runtime up -> 200 STALE + background refresh (stale-while-revalidate)
    stale hit + runtime dn -> 200 STALE, flagged             (endpoint survives the VM)
    miss + runtime up      -> forward, cache, 200 MISS
    miss + runtime down    -> degradation_mode: serve_cached | queue | fail

Everything the caller sees is an :class:`InferenceResponse` envelope, so the public
contract stays identical whether the answer came from a GPU or from disk.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from colabdeploy_shared.enums import DegradationMode, JobState
from colabdeploy_shared.schemas import (
    EndpointInfo,
    InferenceError,
    InferenceResponse,
    JobInfo,
    ModelInfo,
    PredictRequest,
    utcnow,
)
from colabdeploy_shared.utils import cache_key as build_cache_key
from gateway.core.cache import CACHE_STALE, CacheBackend, CacheLookup
from gateway.core.config import Settings
from gateway.core.logging import Telemetry
from gateway.core.repository import Registry
from gateway.services.relay import AgentTimeout, Relay, RelayError
from gateway.services.runtime_manager import RuntimeManager


def _unwrap(value: Any) -> Any:
    """Cached values are stored as ``{"output": ..., "__request__": ...}``."""
    if isinstance(value, dict) and "output" in value and "__request__" in value:
        return value["output"]
    return value


class InferenceFailure(Exception):
    """A request could not be served; carries the HTTP status and error code."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 503,
        retry_after: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after
        self.detail = detail or {}

    def to_error(self) -> InferenceError:
        return InferenceError(code=self.code, message=self.message, detail=self.detail or None)


@dataclass
class ResolvedEndpoint:
    endpoint: EndpointInfo
    model: ModelInfo


class InferenceService:
    def __init__(
        self,
        registry: Registry,
        relay: Relay,
        cache: CacheBackend,
        settings: Settings,
        telemetry: Telemetry,
        runtimes: RuntimeManager,
    ) -> None:
        self.registry = registry
        self.relay = relay
        self.cache = cache
        self.settings = settings
        self.telemetry = telemetry
        self.runtimes = runtimes
        self._refresh_tasks: set[asyncio.Task[Any]] = set()

    # ------------------------------------------------------------------ #
    # Resolution helpers
    # ------------------------------------------------------------------ #
    def resolve_by_endpoint(self, endpoint_id: str) -> ResolvedEndpoint:
        endpoint = self.registry.endpoints.get(endpoint_id)
        if endpoint is None:
            raise InferenceFailure("ENDPOINT_NOT_FOUND", f"endpoint '{endpoint_id}' not found", 404)
        model = self._model_for(endpoint)
        return ResolvedEndpoint(endpoint=endpoint, model=model)

    def resolve_by_model(self, model_id: str) -> ResolvedEndpoint:
        """``POST /v1/models/{model_id}/predict`` - route or model id both work."""
        endpoint = self.registry.endpoints.by_route(model_id)
        if endpoint is None:
            for candidate in self.registry.endpoints.list():
                if candidate.model_id == model_id:
                    endpoint = candidate
                    break
        if endpoint is None:
            raise InferenceFailure(
                "ENDPOINT_NOT_FOUND",
                f"no published endpoint for '{model_id}'. Deploy it first: POST /api/deploy",
                404,
            )
        model = self._model_for(endpoint)
        return ResolvedEndpoint(endpoint=endpoint, model=model)

    def _model_for(self, endpoint: EndpointInfo) -> ModelInfo:
        base = self.registry.models.get(endpoint.model_id)
        if base is None:
            raise InferenceFailure(
                "MODEL_NOT_FOUND", f"model '{endpoint.model_id}' is gone", 404
            )
        if endpoint.model_version and endpoint.model_version != base.version:
            for candidate in self.registry.models.versions_of(base.id):
                if candidate.version == endpoint.model_version:
                    return candidate
        return base

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    async def predict(
        self,
        resolved: ResolvedEndpoint,
        request: PredictRequest,
        *,
        request_id: str | None = None,
        allow_degradation: bool = True,
    ) -> InferenceResponse:
        endpoint, model = resolved.endpoint, resolved.model
        request_id = request_id or uuid.uuid4().hex[:16]
        started = time.perf_counter()
        payload = request.data()

        errors = endpoint.schema_in.validate_payload(payload)
        if errors:
            raise InferenceFailure(
                "INVALID_PAYLOAD",
                "payload does not match the endpoint's input schema",
                422,
                detail={"errors": errors, "expected": endpoint.schema_in.model_dump(mode="json")},
            )

        key = build_cache_key(endpoint.id, model.version, payload)
        lookup = self.cache.get(key) if not request.bypass_cache else CacheLookup(key=key)

        runtime_id = self.runtimes.pick_runtime(endpoint.id, model.id, model.version)

        # --- fresh hit: the cheapest possible answer --------------------- #
        if lookup.status == "HIT":
            return self._finish(
                resolved, _unwrap(lookup.value), "HIT", request_id, started, runtime_id=None, key=key
            )

        # --- stale hit -------------------------------------------------- #
        if lookup.status == CACHE_STALE:
            if runtime_id is not None:
                self._spawn_refresh(resolved, payload, key, request_id)
            return self._finish(
                resolved,
                _unwrap(lookup.value),
                "STALE",
                request_id,
                started,
                runtime_id=runtime_id,
                key=key,
                stale=True,
                stale_for_s=lookup.stale_for_s,
            )

        # --- miss: we need a runtime ------------------------------------ #
        if runtime_id is None:
            if not allow_degradation:
                raise InferenceFailure(
                    "RUNTIME_OFFLINE",
                    "no Colab runtime is connected for this endpoint",
                    503,
                    retry_after=self.settings.retry_after_s,
                )
            return await self._degrade(resolved, payload, key, request_id, started)

        # --- forward to the agent --------------------------------------- #
        timeout_s = min(endpoint.timeout_ms / 1000.0, self.settings.default_timeout_ms / 1000.0)
        try:
            frame = await self.relay.exec_on_runtime(
                runtime_id,
                {
                    "endpoint_id": endpoint.id,
                    "model_id": model.id,
                    "model_version": model.version,
                    "payload": payload,
                    "request_id": request_id,
                },
                path="/execute",
                timeout_s=timeout_s,
            )
        except (AgentTimeout, RelayError) as exc:
            self.telemetry.log(
                f"forward failed: {exc.message}", level="warning", runtime_id=runtime_id
            )
            # A timeout usually means the VM is wedged; drop it and degrade.
            if isinstance(exc, AgentTimeout):
                await self.relay.unregister(runtime_id, reason="exec timeout")
            if allow_degradation:
                return await self._degrade(
                    resolved, payload, key, request_id, started, cause=exc.code
                )
            raise InferenceFailure(exc.code, exc.message, 503, self.settings.retry_after_s) from exc

        if frame.status >= 400:
            body = frame.body or {}
            message = str(body.get("error") or frame.error or f"agent returned {frame.status}")
            code = str(body.get("code") or "AGENT_ERROR")
            self.telemetry.log(f"agent error: {message}", level="error", runtime_id=runtime_id)
            # Model-side failures are not retryable elsewhere - surface them.
            status = 502 if frame.status >= 500 else frame.status
            raise InferenceFailure(code, message, status, detail={"runtime_id": runtime_id})

        output = (frame.body or {}).get("output")
        if output is None and isinstance(frame.body, dict):
            output = frame.body.get("result", frame.body)
        out_errors = endpoint.schema_out.validate_payload(output) if isinstance(output, dict) else []
        if out_errors:
            self.telemetry.log(
                f"output schema mismatch: {out_errors}", level="warning", runtime_id=runtime_id
            )

        self._store(key, resolved, payload, output)
        return self._finish(
            resolved, output, "MISS", request_id, started, runtime_id=runtime_id, key=key,
            agent_latency_ms=frame.agent_latency_ms,
        )

    async def predict_batch(
        self, resolved: ResolvedEndpoint, payloads: list[dict[str, Any]], concurrency: int = 4
    ) -> list[InferenceResponse]:
        if len(payloads) > self.settings.max_batch_size:
            raise InferenceFailure(
                "BATCH_TOO_LARGE",
                f"batch size {len(payloads)} exceeds max {self.settings.max_batch_size}",
                413,
            )
        semaphore = asyncio.Semaphore(concurrency)

        async def one(item: dict[str, Any]) -> InferenceResponse:
            async with semaphore:
                return await self.predict(resolved, PredictRequest(inputs=item))

        return list(await asyncio.gather(*(one(p) for p in payloads)))

    # ------------------------------------------------------------------ #
    # Degradation
    # ------------------------------------------------------------------ #
    async def _degrade(
        self,
        resolved: ResolvedEndpoint,
        payload: dict[str, Any],
        key: str,
        request_id: str,
        started: float,
        cause: str = "RUNTIME_OFFLINE",
    ) -> InferenceResponse:
        endpoint = resolved.endpoint
        mode = DegradationMode(endpoint.degradation_mode)

        if mode == DegradationMode.SERVE_CACHED:
            # `get()` already dropped an over-stale entry; peek at the raw store for
            # a last-resort answer so an offline endpoint still responds.
            last = self.cache.peek(key)
            if last.usable:
                self.telemetry.log(
                    f"serving stale cache while offline ({cause})",
                    level="warning",
                    runtime_id=None,
                )
                return self._finish(
                    resolved, _unwrap(last.value), "STALE", request_id, started, stale=True,
                    stale_for_s=last.stale_for_s, degraded=cause,
                )
            # Nothing cached: fall through to queueing if allowed, else fail.
            mode = DegradationMode.QUEUE

        if mode == DegradationMode.QUEUE:
            job = self.enqueue(resolved, payload)
            self.telemetry.record_queued()
            response = self._finish(
                resolved, None, "MISS", request_id, started, stale=False, degraded=cause
            )
            response.job_id = job.id
            response.error = InferenceError(
                code="QUEUED",
                message=(
                    "no runtime online; the request was queued and will run as soon as a "
                    f"Colab runtime reconnects. Poll GET /v1/jobs/{job.id}"
                ),
                detail={"job_id": job.id, "reason": cause},
            )
            return response

        raise InferenceFailure(
            cause,
            "no Colab runtime is connected and this endpoint is configured to fail fast",
            503,
            self.settings.retry_after_s,
            detail={"degradation_mode": endpoint.degradation_mode},
        )

    def enqueue(self, resolved: ResolvedEndpoint, payload: dict[str, Any], webhook: str | None = None) -> JobInfo:
        from colabdeploy_shared.utils import short_id

        job = JobInfo(
            id=short_id("job", 8),
            endpoint_id=resolved.endpoint.id,
            model_id=resolved.model.id,
            model_version=resolved.model.version,
            inputs=payload,
            webhook_url=webhook,
        )
        self.registry.jobs.put(job)
        self.telemetry.log(
            f"queued job {job.id} for offline endpoint {resolved.endpoint.route}", level="warning"
        )
        return job

    async def dispatch_job(self, job: JobInfo) -> bool:
        """Run one queued job. Returns True if it reached a terminal state."""
        try:
            resolved = self.resolve_by_endpoint(job.endpoint_id)
        except InferenceFailure:
            self.registry.store.update(
                "jobs", job.id, {"state": JobState.FAILED.value, "error": "endpoint no longer exists",
                                "done_at": utcnow().isoformat(), "updated_at": utcnow().isoformat()}
            )
            return True
        attempts = job.attempts + 1
        self.registry.store.update(
            "jobs", job.id, {"state": JobState.RUNNING.value, "attempts": attempts,
                             "updated_at": utcnow().isoformat()}
        )
        try:
            response = await self.predict(
                resolved, PredictRequest(inputs=job.inputs), allow_degradation=False
            )
        except InferenceFailure as exc:
            terminal = attempts >= self.settings.job_max_attempts
            self.registry.store.update(
                "jobs",
                job.id,
                {
                    "state": (JobState.FAILED if terminal else JobState.QUEUED).value,
                    "error": exc.message,
                    "attempts": attempts,
                    "updated_at": utcnow().isoformat(),
                    "done_at": utcnow().isoformat() if terminal else None,
                },
            )
            return terminal
        self.registry.store.update(
            "jobs",
            job.id,
            {
                "state": JobState.SUCCEEDED.value,
                "result": response.output,
                "error": None,
                "updated_at": utcnow().isoformat(),
                "done_at": utcnow().isoformat(),
            },
        )
        self.telemetry.log(f"job {job.id} drained successfully")
        return True

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _store(self, key: str, resolved: ResolvedEndpoint, payload: dict[str, Any], output: Any) -> None:
        endpoint = resolved.endpoint
        if endpoint.cache_ttl_s <= 0:
            return
        # Store the request alongside the response so hot keys can double as
        # warm-up samples when a fresh VM reconnects.
        value = {"output": output, "__request__": payload, "endpoint_route": endpoint.route}
        self.cache.set(
            key,
            value,
            endpoint_id=endpoint.id,
            model_version=resolved.model.version,
            ttl_s=endpoint.cache_ttl_s,
            swr_s=endpoint.stale_while_revalidate_s,
        )

    def _spawn_refresh(
        self, resolved: ResolvedEndpoint, payload: dict[str, Any], key: str, request_id: str
    ) -> None:
        """stale-while-revalidate: answer now, refresh in the background."""

        async def refresh() -> None:
            try:
                await self.predict(
                    resolved,
                    PredictRequest(inputs=payload, bypass_cache=True),
                    request_id=f"{request_id}-swr",
                    allow_degradation=False,
                )
            except InferenceFailure:
                pass  # still offline - the stale entry keeps serving
            except Exception as exc:  # noqa: BLE001
                self.telemetry.log(f"swr refresh failed: {exc}", level="warning")

        try:
            task = asyncio.create_task(refresh())
        except RuntimeError:  # pragma: no cover - no running loop (sync tests)
            return
        self._refresh_tasks.add(task)
        task.add_done_callback(self._refresh_tasks.discard)

    def _finish(
        self,
        resolved: ResolvedEndpoint,
        output: Any,
        cache_status: str,
        request_id: str,
        started: float,
        *,
        runtime_id: str | None = None,
        stale: bool = False,
        stale_for_s: float = 0.0,
        degraded: str | None = None,
        key: str | None = None,
        agent_latency_ms: float | None = None,
    ) -> InferenceResponse:
        endpoint, model = resolved.endpoint, resolved.model
        latency = (time.perf_counter() - started) * 1000
        _ = agent_latency_ms  # surfaced in logs/metrics, not in the response envelope
        self.telemetry.record_request(
            cached=cache_status in ("HIT", "STALE"), stale=stale, failed=False, latency_ms=latency
        )
        response = InferenceResponse(
            model_id=model.id,
            endpoint_id=endpoint.id,
            model_version=model.version,
            output=output,
            cache=cache_status,
            latency_ms=round(latency, 2),
            runtime_id=runtime_id,
            request_id=request_id,
            stale=stale,
        )
        if stale or degraded:
            detail: dict[str, Any] = {}
            if stale_for_s:
                detail["stale_for_s"] = round(stale_for_s, 1)
            if degraded:
                detail["reason"] = degraded
            if key:
                detail["cache_key"] = key
            response.error = InferenceError(
                code="STALE_RESPONSE" if stale else "DEGRADED",
                message="served from cache because no runtime was reachable",
                detail=detail or None,
            )
        return response
