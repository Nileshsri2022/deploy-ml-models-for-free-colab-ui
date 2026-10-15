"""Management API - ``/api``. Everything the UI needs, all behind a ``manage`` key.

Grouped by concern:

* ``/api/bootstrap``  - one-shot setup payload for a fresh install (key + notebook cell)
* ``/api/keys``       - caller API keys
* ``/api/runtimes``   - Colab runtime lifecycle: mint token, watch state, revoke, relaunch
* ``/api/models``     - register model versions (with artifact upload)
* ``/api/endpoints``  - publish, tune (cache TTL / degradation), delete
* ``/api/deploy``     - the deploy wizard's single call, plus rollback
* ``/api/cache``      - inspect the resilience buffer, pin hot keys, invalidate
* ``/api/logs``       - ring buffer + SSE live tail
* ``/api/jobs``       - queued-while-offline work
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from colabdeploy_shared.enums import JobState
from colabdeploy_shared.schemas import (
    ApiKeyCreate,
    CacheEntryInfo,
    CacheStats,
    DeployRequest,
    EndpointCreate,
    EndpointInfo,
    EndpointUpdate,
    JobInfo,
    ModelCreate,
    ModelInfo,
    ModelSchema,
    RuntimeTokenInfo,
)
from gateway.api.deps import app_state, manage_dependency

router = APIRouter(prefix="/api", tags=["management"])

Manage = Depends(manage_dependency)


def _base(request: Request) -> str:
    state = app_state(request)
    return state.settings.public_base_url or str(request.base_url).rstrip("/")


def _err(message: str, code: str = "BAD_REQUEST", status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


# --------------------------------------------------------------------------- #
# Bootstrap: what the UI shows on a fresh install
# --------------------------------------------------------------------------- #
@router.get("/bootstrap")
async def bootstrap(request: Request, principal: Any = Manage) -> dict[str, Any]:
    """Setup payload: the notebook cell to paste into Colab, plus the relaunch URL."""
    state = app_state(request)
    base = _base(request)
    ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
    cell = (
        "# 1) install the agent (one time per session)\n"
        "!pip -q install git+https://github.com/"
        f"{state.settings.colab_repo}"
        "#subdirectory=agent\n\n"
        "# 2) connect this Colab VM to your control plane\n"
        "from colabdeploy import launch\n"
        f'launch(gateway_url="{base}", token="<paste your runtime token>")'
    )
    return {
        "gateway_url": base,
        "relay_ws_url": f"{ws_base}/internal/relay",
        "colab_relaunch_url": state.settings.colab_relaunch_url,
        "notebook_cell": cell,
        "protocol_version": 1,
        "agent_version": state.settings.version,
        "installed": {
            "api_keys": len(state.registry.api_keys.list()),
            "runtimes": len(state.registry.runtimes.list()),
            "models": len(state.registry.models.list()),
            "endpoints": len(state.registry.endpoints.list()),
        },
    }


# --------------------------------------------------------------------------- #
# API keys
# --------------------------------------------------------------------------- #
@router.post("/keys", status_code=201)
async def create_key(body: ApiKeyCreate, request: Request, principal: Any = Manage) -> Any:
    from gateway.core.security import AuthService

    state = app_state(request)
    info, token = AuthService(state.registry).create_api_key(
        body.label, tuple(body.scopes) or ("inference",), body.expires_in_days
    )
    payload = info.model_dump(mode="json")
    payload["token"] = token  # shown exactly once
    return JSONResponse(status_code=201, content=payload)


@router.get("/keys")
async def list_keys(request: Request, principal: Any = Manage) -> Any:
    from gateway.core.security import AuthService

    state = app_state(request)
    return {"keys": [k.model_dump(mode="json") for k in AuthService(state.registry).list_api_keys()]}


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, request: Request, principal: Any = Manage) -> Any:
    from gateway.core.security import AuthService

    state = app_state(request)
    if not AuthService(state.registry).revoke_api_key(key_id):
        return _err(f"api key '{key_id}' not found", "NOT_FOUND", 404)
    return {"revoked": key_id}


# --------------------------------------------------------------------------- #
# Runtimes
# --------------------------------------------------------------------------- #
@router.post("/runtimes/token", status_code=201)
async def mint_runtime_token(
    request: Request,
    label: str = "colab-runtime",
    ttl_hours: float = Query(default=24.0, ge=0.5, le=24 * 30),
    principal: Any = Manage,
) -> Any:
    """Create a runtime record + token. The token is returned **once**."""
    from gateway.core.security import AuthService

    state = app_state(request)
    result = AuthService(state.registry).create_runtime_token(label, ttl_hours)
    payload = RuntimeTokenInfo(**result).model_dump(mode="json")
    payload["notebook_cell"] = (
        "from colabdeploy import launch\n"
        f'launch(gateway_url="{_base(request)}", token="{result["token"]}")'
    )
    payload["colab_relaunch_url"] = state.settings.colab_relaunch_url
    return JSONResponse(status_code=201, content=payload)


@router.get("/runtimes")
async def list_runtimes(request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    items = [
        state.runtimes.runtime_view(r).model_dump(mode="json")
        for r in state.registry.runtimes.list()
    ]
    for item in items:
        record = state.registry.store.get("runtimes", str(item["id"])) or {}
        item["token_prefix"] = record.get("token_prefix")
        item["token_expires_at"] = record.get("token_expires_at")
        item["connected"] = state.relay.is_connected(str(item["id"]))
        item["revoked"] = bool(record.get("revoked"))
    items.sort(key=lambda i: str(i.get("registered_at", "")), reverse=True)
    return {
        "count": len(items),
        "runtimes": items,
        "relay": state.relay.snapshot(),
        "relaunch_url": state.settings.colab_relaunch_url,
    }


@router.get("/runtimes/{runtime_id}")
async def get_runtime(runtime_id: str, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    runtime = state.registry.runtimes.get(runtime_id)
    if runtime is None:
        return _err(f"runtime '{runtime_id}' not found", "NOT_FOUND", 404)
    view = state.runtimes.runtime_view(runtime).model_dump(mode="json")
    view["connected"] = state.relay.is_connected(runtime_id)
    view["desired_state"] = state.runtimes.desired_state_for(runtime_id).model_dump(mode="json")
    view["relaunch_url"] = state.runtimes.relaunch_url(runtime)
    return view


@router.post("/runtimes/{runtime_id}/rotate")
async def rotate_runtime_token(
    runtime_id: str, request: Request, ttl_hours: float = 24.0, principal: Any = Manage
) -> Any:
    from gateway.core.security import AuthService

    state = app_state(request)
    result = AuthService(state.registry).rotate_runtime_token(runtime_id, ttl_hours)
    if result is None:
        return _err(f"runtime '{runtime_id}' not found", "NOT_FOUND", 404)
    return result


@router.delete("/runtimes/{runtime_id}")
async def revoke_runtime(runtime_id: str, request: Request, principal: Any = Manage) -> Any:
    """Revoke a runtime token and drop its tunnel if it is live."""
    from gateway.core.security import AuthService

    state = app_state(request)
    if not AuthService(state.registry).revoke_runtime_token(runtime_id):
        return _err(f"runtime '{runtime_id}' not found", "NOT_FOUND", 404)
    await state.relay.unregister(runtime_id, reason="token revoked")
    return {"revoked": runtime_id}


@router.post("/runtimes/{runtime_id}/reconcile")
async def reconcile_runtime(runtime_id: str, request: Request, principal: Any = Manage) -> Any:
    """Force-push desired state (and hot cache keys as warm-up samples) to a VM."""
    state = app_state(request)
    if not state.relay.is_connected(runtime_id):
        return _err(
            f"runtime '{runtime_id}' has no live tunnel", "RUNTIME_OFFLINE", 503
        )
    ok = await state.runtimes.push_desired_state(runtime_id, with_warmup=True)
    return {"pushed": ok, "runtime_id": runtime_id}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@router.post("/models", status_code=201)
async def register_model(body: ModelCreate, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    try:
        model = state.deployments.register_model(body, owner=principal.label or principal.id)
    except ValueError as exc:
        return _err(str(exc))
    return JSONResponse(status_code=201, content=state.deployments.model_view(model))


@router.post("/models/upload", status_code=201)
async def upload_model(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    framework: str = Form("auto"),
    adapter: str = Form("auto"),
    schema_in: str | None = Form(None),
    schema_out: str | None = Form(None),
    warmup_sample: str | None = Form(None),
    entrypoint: str | None = Form(None),
    tags: str | None = Form(None),
    principal: Any = Manage,
) -> Any:
    """Upload an artifact through the UI and register it in one call."""
    state = app_state(request)
    data = await file.read()
    if len(data) > state.settings.max_body_bytes * 4:
        return _err(
            f"artifact too large ({len(data)} bytes); point at a Drive/HF URI instead",
            "PAYLOAD_TOO_LARGE",
            413,
        )
    artifact = state.deployments.store_artifact(file.filename or "model.bin", data)

    def _schema(raw: str | None) -> ModelSchema:
        if not raw:
            return ModelSchema()
        try:
            return ModelSchema.model_validate(json.loads(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid schema JSON: {exc}") from exc

    try:
        create = ModelCreate(
            name=name,
            framework=framework,
            adapter=adapter,  # type: ignore[arg-type]
            artifact=artifact,
            schema_in=_schema(schema_in),
            schema_out=_schema(schema_out),
            warmup_sample=json.loads(warmup_sample) if warmup_sample else None,
            entrypoint=entrypoint,
            tags=[t.strip() for t in (tags or "").split(",") if t.strip()],
        )
        model = state.deployments.register_model(create, owner=principal.label or principal.id)
    except ValueError as exc:
        return _err(str(exc))
    return JSONResponse(status_code=201, content=state.deployments.model_view(model))


@router.get("/models")
async def list_models(request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    models = state.registry.models.list()
    latest: dict[str, ModelInfo] = {}
    for model in models:
        current = latest.get(model.name)
        if current is None or model.version > current.version:
            latest[model.name] = model
    return {
        "count": len(models),
        "models": [state.deployments.model_view(m) for m in models],
        "latest": [state.deployments.model_view(m) for m in latest.values()],
    }


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    model = state.registry.models.get(model_id) or state.deployments.model_by_name_version(model_id)
    if model is None:
        return _err(f"model '{model_id}' not found", "NOT_FOUND", 404)
    return state.deployments.model_view(model)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post("/endpoints", status_code=201)
async def create_endpoint(body: EndpointCreate, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    try:
        endpoint = state.deployments.create_endpoint(body)
    except ValueError as exc:
        return _err(str(exc))
    return JSONResponse(
        status_code=201, content=state.deployments.endpoint_view(endpoint, _base(request))
    )


@router.get("/endpoints")
async def list_endpoints(request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    base = _base(request)
    return {
        "count": len(state.registry.endpoints.list()),
        "endpoints": [
            state.deployments.endpoint_view(ep, base) for ep in state.registry.endpoints.list()
        ],
    }


@router.patch("/endpoints/{endpoint_id}")
async def update_endpoint(
    endpoint_id: str, body: EndpointUpdate, request: Request, principal: Any = Manage
) -> Any:
    state = app_state(request)
    endpoint = state.deployments.update_endpoint(endpoint_id, **body.model_dump(exclude_none=True))
    if endpoint is None:
        return _err(f"endpoint '{endpoint_id}' not found", "NOT_FOUND", 404)
    if body.model_dump(exclude_none=True).get("cache_ttl_s") == 0:
        state.deployments.cache_invalidate(endpoint_id)
    return state.deployments.endpoint_view(endpoint, _base(request))


@router.delete("/endpoints/{endpoint_id}")
async def delete_endpoint(endpoint_id: str, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    if not state.deployments.delete_endpoint(endpoint_id):
        return _err(f"endpoint '{endpoint_id}' not found", "NOT_FOUND", 404)
    return {"deleted": endpoint_id}


# --------------------------------------------------------------------------- #
# Deploy / rollback
# --------------------------------------------------------------------------- #
@router.post("/deploy", status_code=201)
async def deploy(body: DeployRequest, request: Request, principal: Any = Manage) -> Any:
    """The wizard's single call: pin a model version to a (new or existing) endpoint."""
    state = app_state(request)
    try:
        result = state.deployments.deploy(
            body.model_id,
            model_version=body.model_version,
            endpoint_id=body.endpoint_id,
            runtime_id=body.runtime_id,
            endpoint_spec=body.endpoint,
        )
    except ValueError as exc:
        return _err(str(exc))

    # Push intent to the VM immediately if it is live; otherwise it will be picked
    # up from desired-state the moment it reconnects.
    runtime_id = result["runtime_id"]
    pushed = False
    if runtime_id and state.relay.is_connected(runtime_id):
        pushed = await state.runtimes.push_desired_state(runtime_id)

    endpoint: EndpointInfo = result["endpoint"]  # type: ignore[assignment]
    view = state.deployments.endpoint_view(endpoint, _base(request))
    return JSONResponse(
        status_code=201,
        content={
            "deployment": result["deployment"].model_dump(mode="json"),
            "endpoint": view,
            "model": result["model"].model_dump(mode="json"),
            "runtime_id": runtime_id,
            "runtime_online": result["runtime_online"],
            "pushed_to_runtime": pushed,
            "curl": view["curl"],
            "message": (
                "deployed and pushed to the runtime"
                if pushed
                else "deployed; the runtime will load it on next connect"
            ),
        },
    )


@router.post("/endpoints/{endpoint_id}/rollback")
async def rollback(
    endpoint_id: str, request: Request, model_version: int = Query(..., ge=1), principal: Any = Manage
) -> Any:
    state = app_state(request)
    try:
        result = state.deployments.rollback(endpoint_id, model_version)
    except ValueError as exc:
        return _err(str(exc), "NOT_FOUND", 404)
    state.deployments.cache_invalidate(endpoint_id)
    runtime_id = result["runtime_id"]
    pushed = False
    if runtime_id and state.relay.is_connected(runtime_id):
        pushed = await state.runtimes.push_desired_state(runtime_id)
    return {
        "rolled_back_to": model_version,
        "endpoint": state.deployments.endpoint_view(result["endpoint"], _base(request)),
        "pushed_to_runtime": pushed,
    }


@router.get("/deployments")
async def list_deployments(request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    return {
        "count": len(state.registry.deployments.list()),
        "deployments": [d.model_dump(mode="json") for d in state.registry.deployments.list()],
    }


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
@router.get("/cache/stats", response_model=CacheStats)
async def cache_stats(request: Request, principal: Any = Manage) -> CacheStats:
    return app_state(request).cache.stats()


@router.get("/cache/entries")
async def cache_entries(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    endpoint_id: str | None = None,
    principal: Any = Manage,
) -> Any:
    state = app_state(request)
    entries: list[CacheEntryInfo] = state.cache.entries(limit=limit, endpoint_id=endpoint_id)
    return {
        "count": len(entries),
        "entries": [e.model_dump(mode="json") for e in entries],
        "stats": state.cache.stats().model_dump(mode="json"),
    }


@router.get("/cache/entry/{key}")
async def cache_entry(key: str, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    lookup = state.cache.peek(key)
    if lookup.status == "MISS" and lookup.value is None:
        return _err(f"cache key '{key}' not found", "NOT_FOUND", 404)
    return {
        "key": key,
        "status": lookup.status,
        "value": lookup.value,
        "age_s": round(lookup.age_s, 2),
        "endpoint_id": lookup.endpoint_id,
        "model_version": lookup.model_version,
    }


@router.post("/cache/entry/{key}/pin")
async def pin_cache_entry(
    request: Request, key: str, pinned: bool = Query(default=True), principal: Any = Manage
) -> Any:
    state = app_state(request)
    if not state.cache.pin(key, pinned):
        return _err(f"cache key '{key}' not found", "NOT_FOUND", 404)
    return {"key": key, "pinned": pinned}


@router.delete("/cache/entry/{key}")
async def delete_cache_entry(key: str, request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    return {"deleted": key, "found": state.cache.delete(key)}


@router.post("/cache/invalidate")
async def invalidate_cache(
    request: Request, endpoint_id: str | None = None, principal: Any = Manage
) -> Any:
    state = app_state(request)
    if endpoint_id:
        return {"invalidated": state.cache.invalidate_endpoint(endpoint_id), "scope": endpoint_id}
    cleared = state.cache.stats().entries
    if hasattr(state.cache, "clear"):
        state.cache.clear()  # type: ignore[attr-defined]
    return {"invalidated": cleared, "scope": "all"}


@router.post("/cache/save")
async def save_cache(request: Request, principal: Any = Manage) -> Any:
    """Force a snapshot to disk (memory backend) - the state-retention guarantee."""
    state = app_state(request)
    save = getattr(state.cache, "save", None)
    if not callable(save):
        return {"saved": False, "backend": state.cache.name}
    return {"saved": bool(save()), "backend": state.cache.name, "stats": state.cache.stats().model_dump(mode="json")}


@router.get("/cache/hot")
async def hot_keys(
    request: Request, limit: int = Query(default=20, ge=1, le=200), principal: Any = Manage
) -> Any:
    """Keys sent to an agent as warm-up samples when it reconnects."""
    state = app_state(request)
    return {"count": limit, "hot_keys": state.cache.hot_keys(limit=limit)}


# --------------------------------------------------------------------------- #
# Logs, audit, jobs
# --------------------------------------------------------------------------- #
@router.get("/logs")
async def recent_logs(
    request: Request,
    limit: int = Query(default=200, ge=1, le=2000),
    level: str | None = None,
    runtime_id: str | None = None,
    principal: Any = Manage,
) -> Any:
    state = app_state(request)
    lines = [
        line
        for line in reversed(list(state.telemetry.logs._items))  # noqa: SLF001 - bounded ring read
        if (level is None or line.level == level) and (runtime_id is None or line.runtime_id == runtime_id)
    ][:limit]
    return {"count": len(lines), "logs": [ln.model_dump(mode="json") for ln in lines]}


@router.get("/logs/stream")
async def log_stream(request: Request, principal: Any = Manage) -> StreamingResponse:
    """Server-Sent Events live tail for the UI."""
    state = app_state(request)
    queue = await state.sse.subscribe()

    async def event_source() -> Any:
        try:
            async for chunk in state.sse.stream(queue):
                if await request.is_disconnected():
                    break
                yield chunk
        finally:
            await state.sse.unsubscribe(queue)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/audit")
async def audit(request: Request, limit: int = Query(default=100, ge=1, le=1000), principal: Any = Manage) -> Any:
    state = app_state(request)
    return {"entries": state.registry.audit.recent(limit)}


@router.get("/jobs")
async def list_jobs(
    request: Request,
    state_filter: str | None = Query(default=None, alias="state"),
    limit: int = Query(default=100, ge=1, le=1000),
    principal: Any = Manage,
) -> Any:
    st = app_state(request)
    jobs = st.registry.jobs.list()
    if state_filter:
        jobs = [j for j in jobs if j.state.value == state_filter]
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return {"count": len(jobs[:limit]), "jobs": [j.model_dump(mode="json") for j in jobs[:limit]]}


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request, principal: Any = Manage) -> Any:
    st = app_state(request)
    job = st.registry.jobs.get(job_id)
    if job is None:
        return _err(f"job '{job_id}' not found", "NOT_FOUND", 404)
    if job.state == JobState.QUEUED:
        return {"queued": job_id, "message": "already queued"}
    st.registry.store.update(
        "jobs", job_id, {"state": JobState.QUEUED.value, "error": None, "attempts": 0, "done_at": None}
    )
    dispatched = await st.inference.dispatch_job(JobInfo.model_validate(st.registry.jobs.get(job_id)))
    return {"queued": job_id, "dispatched": dispatched, "job": st.registry.jobs.get(job_id).model_dump(mode="json")}


@router.get("/telemetry")
async def telemetry(request: Request, principal: Any = Manage) -> Any:
    state = app_state(request)
    return {
        "summary": state.telemetry.summary(),
        "relay": state.relay.snapshot(),
        "cache": state.cache.stats().model_dump(mode="json"),
        "sse_subscribers": state.sse.subscribers,
    }


__all__ = ["router"]
