"""The public API - ``/v1``.

This is the "common API" that other projects consume. It is deliberately tiny and
stable: one predict route, one batch route, one schema route, one jobs pair, and
health/status. Everything else lives under ``/api`` (management) or ``/internal``
(agent).

Response envelope is always :class:`InferenceResponse`, so a caller can tell a
fresh GPU answer (``cache: MISS``) from a cached one (``HIT``) from a
served-while-offline one (``STALE`` + ``stale: true``) without parsing prose.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from colabdeploy_shared.enums import JobState
from colabdeploy_shared.schemas import (
    HealthResponse,
    InferenceResponse,
    JobCreate,
    JobInfo,
    PredictRequest,
    StatusResponse,
)
from gateway.api.deps import app_state, inference_dependency
from gateway.services.inference import InferenceFailure

router = APIRouter(prefix="/v1", tags=["public"])


def _base_url(request: Request) -> str:
    state = app_state(request)
    if state.settings.public_base_url:
        return state.settings.public_base_url
    return str(request.base_url).rstrip("/")


def _failure_response(exc: InferenceFailure, extra_headers: dict[str, str] | None = None) -> JSONResponse:
    headers = {"X-ColabDeploy-Error": exc.code}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    if extra_headers:
        headers.update(extra_headers)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": exc.code, "message": exc.message, "detail": exc.detail or None}
        },
        headers=headers,
    )


def _response_headers(payload: InferenceResponse) -> dict[str, str]:
    headers = {
        "X-ColabDeploy-Cache": payload.cache,
        "X-ColabDeploy-Request-Id": payload.request_id,
        "X-ColabDeploy-Latency-Ms": f"{payload.latency_ms:.1f}",
    }
    if payload.stale:
        headers["X-ColabDeploy-Stale"] = "true"
        headers["Cache-Control"] = "private, max-age=0, must-revalidate"
    if payload.job_id:
        headers["X-ColabDeploy-Job"] = payload.job_id
    if payload.runtime_id:
        headers["X-ColabDeploy-Runtime"] = payload.runtime_id
    return headers


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@router.post("/models/{identifier}/predict", response_model=InferenceResponse)
async def predict(
    identifier: str,
    body: PredictRequest,
    request: Request,
    response: Response,
    principal: Any = Depends(inference_dependency),
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
) -> Any:
    """Synchronous inference.

    ``identifier`` may be the endpoint **route** (preferred, stable across model
    versions) or the model id. The response always carries ``cache`` =
    ``HIT | MISS | STALE | BYPASS``.
    """
    state = app_state(request)
    try:
        resolved = _resolve(state, identifier)
        payload = await state.inference.predict(resolved, body, request_id=x_request_id)
    except InferenceFailure as exc:
        return _failure_response(exc)
    for key, value in _response_headers(payload).items():
        response.headers[key] = value
    status = 202 if payload.job_id and payload.output is None else 200
    return JSONResponse(status_code=status, content=payload.model_dump(mode="json"),
                        headers=_response_headers(payload))


@router.post("/models/{identifier}/predict:batch")
async def predict_batch(
    identifier: str,
    body: dict[str, Any],
    request: Request,
    principal: Any = Depends(inference_dependency),
) -> Any:
    """Batch inference. Body: ``{"instances": [{...}, {...}]}`` or ``{"inputs_list": [...]}``."""
    state = app_state(request)
    items = body.get("instances") or body.get("inputs_list") or body.get("payloads") or []
    if not isinstance(items, list):
        return _failure_response(
            InferenceFailure("INVALID_PAYLOAD", "'instances' must be a list of objects", 422)
        )
    try:
        resolved = _resolve(state, identifier)
        results = await state.inference.predict_batch(resolved, [i for i in items if isinstance(i, dict)])
    except InferenceFailure as exc:
        return _failure_response(exc)
    return {
        "endpoint_id": resolved.endpoint.id,
        "model_id": resolved.model.id,
        "model_version": resolved.model.version,
        "count": len(results),
        "results": [r.model_dump(mode="json") for r in results],
    }


@router.get("/models/{identifier}/schema")
async def model_schema(identifier: str, request: Request) -> Any:
    """The endpoint's request/response contract - what the UI form generator uses."""
    state = app_state(request)
    try:
        resolved = _resolve(state, identifier)
    except InferenceFailure as exc:
        return _failure_response(exc)
    endpoint, model = resolved.endpoint, resolved.model
    return {
        "endpoint_id": endpoint.id,
        "model_id": model.id,
        "model_name": model.name,
        "model_version": model.version,
        "adapter": model.adapter,
        "route": endpoint.route,
        "url": f"{_base_url(request)}/v1/models/{endpoint.route.strip('/')}/predict",
        "input": endpoint.schema_in.model_dump(mode="json"),
        "output": endpoint.schema_out.model_dump(mode="json"),
        "cache_ttl_s": endpoint.cache_ttl_s,
        "stale_while_revalidate_s": endpoint.stale_while_revalidate_s,
        "degradation_mode": endpoint.degradation_mode,
        "example_curl": state.deployments.endpoint_view(endpoint, _base_url(request))["curl"],
    }


@router.get("/endpoints")
async def list_endpoints(request: Request) -> Any:
    """All published endpoints, with live/degraded status - safe to expose publicly."""
    state = app_state(request)
    base = _base_url(request)
    items = []
    for endpoint in state.registry.endpoints.list():
        view = state.deployments.endpoint_view(endpoint, base)
        items.append(
            {
                "id": view["id"],
                "route": view["route"],
                "url": view["url"],
                "model_name": view["model_name"],
                "model_version": view["model_version"],
                "adapter": view["adapter"],
                "status": view["status"],
                "live": view["live"],
                "runtime_state": view["runtime_state"],
                "cache_ttl_s": view["cache_ttl_s"],
                "degradation_mode": view["degradation_mode"],
            }
        )
    return {"count": len(items), "endpoints": items, "base_url": base}


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@router.post("/jobs", status_code=202)
async def create_job(
    body: JobCreate, request: Request, principal: Any = Depends(inference_dependency)
) -> Any:
    """Queue an inference job explicitly (long-running models, or offline runtimes)."""
    state = app_state(request)
    identifier = body.endpoint_id or body.model_id
    if not identifier:
        return _failure_response(
            InferenceFailure("INVALID_PAYLOAD", "endpoint_id or model_id is required", 422)
        )
    try:
        resolved = state.inference.resolve_by_endpoint(identifier) \
            if body.endpoint_id else _resolve(state, identifier)
    except InferenceFailure as exc:
        return _failure_response(exc)
    job = state.inference.enqueue(resolved, body.inputs, body.webhook_url)
    return JSONResponse(
        status_code=202,
        content=job.model_dump(mode="json"),
        headers={"Location": f"/v1/jobs/{job.id}"},
    )


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> Any:
    state = app_state(request)
    job = state.registry.jobs.get(job_id)
    if job is None:
        return _failure_response(InferenceFailure("JOB_NOT_FOUND", f"job '{job_id}' not found", 404))
    payload = job.model_dump(mode="json")
    headers = {}
    if job.state in (JobState.QUEUED, JobState.RUNNING):
        headers["Retry-After"] = str(max(1, int(state.settings.heartbeat_interval_s)))
    return JSONResponse(content=payload, headers=headers)


# --------------------------------------------------------------------------- #
# Health / status / metrics
# --------------------------------------------------------------------------- #
@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness of the control plane (not of any Colab runtime)."""
    from gateway import __version__

    return HealthResponse(status="ok", version=__version__)


@router.get("/status", response_model=StatusResponse)
async def status(request: Request) -> StatusResponse:
    """One call for the dashboard: runtimes, endpoints, cache, queue."""
    state = app_state(request)
    runtimes = [state.runtimes.runtime_view(r) for r in state.registry.runtimes.list()]
    online = [r for r in runtimes if r.state == "online"]
    degraded = [r for r in runtimes if r.state in ("degraded", "stale")]
    endpoints = state.registry.endpoints.list()
    serving = sum(
        1
        for ep in endpoints
        if (dep := state.registry.deployments.active_for_endpoint(ep.id)) is not None
        and dep.state.value == "ready"
    )
    summary = state.telemetry.summary()
    return StatusResponse(
        healthy=True,
        runtimes_total=len(runtimes),
        runtimes_online=len(online),
        runtimes_degraded=len(degraded),
        models=len({m.name for m in state.registry.models.list()}),
        endpoints=len(endpoints),
        endpoints_serving=serving,
        deployments_ready=sum(
            1 for d in state.registry.deployments.list() if d.state.value == "ready"
        ),
        jobs_queued=len(state.registry.jobs.queued()),
        cache=state.cache.stats(),
        requests_total=summary["requests_total"],
        requests_cached=summary["requests_cached"],
        uptime_s=summary["uptime_s"],
        version=state.settings.version,
        runtimes=sorted(runtimes, key=lambda r: r.registered_at, reverse=True),
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request) -> str:
    """Prometheus text format - scrape this from the free-tier host."""
    state = app_state(request)
    summary = state.telemetry.summary()
    cache = state.cache.stats()
    relay = state.relay.snapshot()
    lines = [
        "# HELP colabdeploy_up Whether the control plane is serving.",
        "# TYPE colabdeploy_up gauge",
        "colabdeploy_up 1",
        "# HELP colabdeploy_runtimes_connected Colab runtimes with a live tunnel.",
        "# TYPE colabdeploy_runtimes_connected gauge",
        f"colabdeploy_runtimes_connected {len(relay['connected'])}",
        "# HELP colabdeploy_requests_total Inference requests served.",
        "# TYPE colabdeploy_requests_total counter",
        f"colabdeploy_requests_total {summary['requests_total']}",
        f"colabdeploy_requests_cached_total {summary['requests_cached']}",
        f"colabdeploy_requests_stale_total {summary['requests_stale']}",
        f"colabdeploy_requests_failed_total {summary['requests_failed']}",
        f"colabdeploy_requests_queued_total {summary['requests_queued']}",
        "# HELP colabdeploy_latency_ms Gateway-observed latency percentiles.",
        "# TYPE colabdeploy_latency_ms gauge",
        f'colabdeploy_latency_ms{{quantile="0.5"}} {summary["latency_p50_ms"]}',
        f'colabdeploy_latency_ms{{quantile="0.95"}} {summary["latency_p95_ms"]}',
        f'colabdeploy_latency_ms{{quantile="0.99"}} {summary["latency_p99_ms"]}',
        "# HELP colabdeploy_cache_entries Cached prediction entries.",
        "# TYPE colabdeploy_cache_entries gauge",
        f"colabdeploy_cache_entries {cache.entries}",
        f"colabdeploy_cache_hit_ratio {cache.hit_ratio}",
        f"colabdeploy_cache_bytes {cache.size_bytes}",
        f"colabdeploy_cache_pinned {cache.pinned}",
        "# HELP colabdeploy_endpoints_total Published endpoints.",
        "# TYPE colabdeploy_endpoints_total gauge",
        f"colabdeploy_endpoints_total {state.registry.endpoints.list().__len__()}",
        f"colabdeploy_jobs_queued {len(state.registry.jobs.queued())}",
        f"colabdeploy_relay_pending_requests {relay['pending_requests']}",
        f"colabdeploy_relay_timeouts_total {relay['timeouts']}",
        f"colabdeploy_uptime_seconds {summary['uptime_s']}",
    ]
    for runtime in state.registry.runtimes.list():
        state_value = {"online": 1, "degraded": 0.5, "stale": 0.25}.get(str(runtime.state.value), 0)
        lines.append(
            f'colabdeploy_runtime_state{{runtime_id="{runtime.id}",label="{runtime.label}"}} {state_value}'
        )
    return "\n".join(lines) + "\n"


def _resolve(state: Any, identifier: str) -> Any:
    """Accept an endpoint route or a model id interchangeably."""
    return state.inference.resolve_by_model(identifier)
