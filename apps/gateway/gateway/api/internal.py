"""Agent-internal API - ``/internal``. Authenticated by **runtime token** only.

These are the endpoints a Colab VM calls. They are deliberately minimal, because
every byte here travels over a free-tier tunnel:

* ``POST /internal/runtime/register`` - claim the token, report hardware
* ``GET  /internal/runtime/desired-state`` - what should be loaded (poll fallback)
* ``POST /internal/runtime/heartbeat`` - liveness + loaded models (HTTP fallback)
* ``POST /internal/runtime/logs`` - batched log shipping
* ``GET  /internal/artifacts/{id}`` - pull an artifact uploaded through the UI
* ``WS   /internal/relay`` - the reverse tunnel itself

The WebSocket route is the interesting one. Sequence:

    agent ──hello{token,runtime_id,capabilities}──▶ gateway
    gateway: verify token → hello_ack → register tunnel → push desired_state
    then: gateway ⇄ agent exec/exec_response frames, heartbeats, logs
    on close: runtime → offline, deployments → pending, SSE notified
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from typing import Any

from fastapi import APIRouter, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response

from colabdeploy_shared.protocol import (
    AGENT_PROTOCOL_VERSION,
    AckFrame,
    ErrorFrame,
    ExecResponseFrame,
    HelloAckFrame,
    HelloFrame,
    LogFrame,
    parse_frame,
)
from colabdeploy_shared.schemas import RuntimeCapabilities, utcnow
from gateway.api.deps import app_state, require_runtime
from gateway.services.relay import RelayConnection

router = APIRouter(prefix="/internal", tags=["agent"])

RELAY_IDLE_TIMEOUT_S = 120.0


def _agent_err(message: str, code: str = "UNAUTHENTICATED", status: int = 401) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


# --------------------------------------------------------------------------- #
# HTTP control endpoints (used by the agent, and as a fallback when WS is down)
# --------------------------------------------------------------------------- #
@router.post("/runtime/register")
async def register_runtime(
    request: Request,
    body: dict[str, Any],
    x_runtime_token: str | None = Header(default=None, alias="X-Runtime-Token"),
) -> Any:
    """Bind a VM to its runtime record. Returns identity + desired state."""
    token = str(body.get("token") or x_runtime_token or "")
    state = app_state(request)
    record = require_runtime(request, token)
    runtime_id = str(record["id"])

    caps = RuntimeCapabilities.model_validate(
        {**record.get("capabilities", {}), **(body.get("capabilities") or {})}
    )
    session_started = body.get("session_started_at") or utcnow().isoformat()
    state.registry.store.update(
        "runtimes",
        runtime_id,
        {
            "state": "connecting",
            "capabilities": caps.model_dump(mode="json"),
            "colab_url": body.get("colab_url") or record.get("colab_url"),
            "session_started_at": session_started,
            "last_heartbeat_at": utcnow().isoformat(),
            "error": None,
        },
    )
    state.telemetry.log(
        f"runtime registered: {caps.gpu_name or 'CPU'} · {caps.cpu_count or '?'} vCPU · "
        f"{caps.ram_mb or '?'} MB",
        runtime_id=runtime_id,
    )
    desired = state.runtimes.desired_state_for(runtime_id)
    return {
        "runtime_id": runtime_id,
        "agent_protocol": AGENT_PROTOCOL_VERSION,
        "server_version": state.settings.version,
        "relay_ws_url": str(request.base_url).rstrip("/").replace("http", "ws") + "/internal/relay",
        "heartbeat_interval_s": state.settings.heartbeat_interval_s,
        "session_max_hours": state.settings.session_max_hours,
        "desired_state": desired.model_dump(mode="json"),
    }


@router.get("/runtime/desired-state")
async def desired_state(
    request: Request,
    runtime_id: str,
    x_runtime_token: str | None = Header(default=None, alias="X-Runtime-Token"),
) -> Any:
    """Poll fallback: what the control plane wants this VM to serve."""
    state = app_state(request)
    record = require_runtime(request, str(x_runtime_token or ""))
    if str(record["id"]) != runtime_id:
        return _agent_err("token does not match runtime_id", "FORBIDDEN", 403)
    desired = state.runtimes.desired_state_for(runtime_id)
    payload = desired.model_dump(mode="json")
    # Resolve store:// artifacts into a URL the agent can actually fetch.
    base = str(request.base_url).rstrip("/")
    for dep in payload["deployments"]:
        artifact = dep.get("artifact") or {}
        uri = str(artifact.get("uri", ""))
        if uri.startswith("store://"):
            artifact["download_url"] = f"{base}/internal/artifacts/{uri.split('://', 1)[1]}"
    payload["artifact_auth"] = {"X-Runtime-Token": str(x_runtime_token or "")}
    return payload


@router.post("/runtime/heartbeat")
async def heartbeat(
    request: Request,
    body: dict[str, Any],
    x_runtime_token: str | None = Header(default=None, alias="X-Runtime-Token"),
) -> Any:
    """HTTP heartbeat - the WS tunnel carries heartbeats too, this is the fallback."""
    from colabdeploy_shared.protocol import HeartbeatFrame

    state = app_state(request)
    record = require_runtime(request, str(x_runtime_token or body.get("token") or ""))
    frame = HeartbeatFrame.model_validate({**body, "runtime_id": str(record["id"])})
    await state.runtimes.on_heartbeat(frame)
    desired_revision = state.runtimes.desired_state_for(str(record["id"])).revision
    return {
        "ok": True,
        "runtime_id": str(record["id"]),
        "server_time": utcnow().isoformat(),
        "desired_state_revision": desired_revision,
        "heartbeat_interval_s": state.settings.heartbeat_interval_s,
    }


@router.post("/runtime/logs")
async def push_logs(
    request: Request,
    body: dict[str, Any],
    x_runtime_token: str | None = Header(default=None, alias="X-Runtime-Token"),
) -> Any:
    state = app_state(request)
    record = require_runtime(request, str(x_runtime_token or ""))
    lines = body.get("lines") or []
    frame = LogFrame(runtime_id=str(record["id"]), lines=[ln for ln in lines if isinstance(ln, dict)])
    added = state.telemetry.ingest_agent_logs(frame.runtime_id, frame.lines)
    for line in frame.lines:
        state.publish_log(_as_log_line(line, frame.runtime_id))
    return {"accepted": added}


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    request: Request,
    x_runtime_token: str | None = Header(default=None, alias="X-Runtime-Token"),
) -> Any:
    """Stream an artifact uploaded through the UI back down to the agent."""
    state = app_state(request)
    require_runtime(request, str(x_runtime_token or ""))
    path = state.deployments.artifact_path(f"store://{artifact_id}")
    if path is None or not path.exists():
        return _agent_err(f"artifact '{artifact_id}' not found", "NOT_FOUND", 404)
    data = path.read_bytes()
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{artifact_id}.bin"',
            "X-ColabDeploy-Checksum": hashlib.sha256(data).hexdigest(),
            "Cache-Control": "private, max-age=86400",
        },
    )


def _as_log_line(raw: dict[str, Any], runtime_id: str) -> Any:
    from colabdeploy_shared.schemas import LogLine

    try:
        return LogLine.model_validate({**raw, "runtime_id": runtime_id, "source": "agent"})
    except Exception:  # noqa: BLE001
        return LogLine(runtime_id=runtime_id, source="agent", message=str(raw)[:2000])


# --------------------------------------------------------------------------- #
# The reverse tunnel
# --------------------------------------------------------------------------- #
@router.websocket("/relay")
async def relay_socket(websocket: WebSocket) -> None:
    """Outbound-dialed WebSocket from the Colab agent.

    The first frame must be a ``hello`` carrying a valid runtime token; anything
    else is closed with 4401 so the agent can surface a clear message in Colab.
    """
    state = app_state(websocket)
    await websocket.accept()
    runtime_id = ""
    reason = "closed"

    def emit(message: str, level: str = "info") -> None:
        """Build a log line once and fan it out to the ring + SSE subscribers."""
        from colabdeploy_shared.schemas import LogLine

        state.publish_log(LogLine(level=level, runtime_id=runtime_id or None, message=message))

    hello: HelloFrame | None = None
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        frame = parse_frame(raw)
        if not isinstance(frame, HelloFrame):
            await _safe_send(websocket, ErrorFrame(code="expected_hello", message="first frame must be hello"))
            await websocket.close(code=4400, reason="expected hello")
            return
        hello = frame
    except (asyncio.TimeoutError, TimeoutError, WebSocketDisconnect, ValueError):
        with contextlib.suppress(Exception):
            await websocket.close(code=4400, reason="bad handshake")
        return

    if hello.protocol != AGENT_PROTOCOL_VERSION:
        await _safe_send(
            websocket,
            ErrorFrame(
                code="protocol_mismatch",
                message=f"agent protocol {hello.protocol}, gateway {AGENT_PROTOCOL_VERSION}",
            ),
        )
        await websocket.close(code=4426, reason="protocol mismatch")
        return

    record = state.registry.store.get("runtimes", hello.runtime_id)
    if record is None or bool(record.get("revoked")):
        await _safe_send(websocket, ErrorFrame(code="unknown_runtime", message="runtime not found"))
        await websocket.close(code=4404, reason="unknown runtime")
        return

    from gateway.core.security import AuthService

    if AuthService(state.registry).verify_runtime_token(hello.token) is None:
        await _safe_send(websocket, ErrorFrame(code="bad_token", message="invalid runtime token"))
        await websocket.close(code=4401, reason="invalid runtime token")
        return
    if str(record["id"]) != hello.runtime_id:
        await _safe_send(websocket, ErrorFrame(code="token_mismatch", message="token/runtime mismatch"))
        await websocket.close(code=4403, reason="token mismatch")
        return

    runtime_id = hello.runtime_id
    conn = RelayConnection(
        runtime_id=hello.runtime_id,
        ws=websocket,
        agent_version=hello.agent_version,
        capabilities=hello.capabilities,
        tunnel_url=str(hello.capabilities.get("tunnel_url") or "") or None,
    )
    await state.relay.register(conn)
    await state.runtimes.on_hello(conn, hello)
    await _safe_send(
        websocket,
        HelloAckFrame(
            ok=True,
            runtime_id=hello.runtime_id,
            desired_state_revision=state.runtimes.desired_state_for(hello.runtime_id).revision,
            server_version=state.settings.version,
        ),
    )
    # Immediately tell the VM what to load, plus hot cache keys to warm with.
    await state.runtimes.push_desired_state(hello.runtime_id, with_warmup=True)
    emit(f"tunnel established (agent {hello.agent_version}, protocol {hello.protocol})")

    try:
        while True:
            message = await _receive_with_idle_timeout(websocket, RELAY_IDLE_TIMEOUT_S)
            if message is None:
                emit(f"tunnel idle > {RELAY_IDLE_TIMEOUT_S:.0f}s, closing", "warning")
                break
            await state.relay.handle_inbound(hello.runtime_id, message)
    except WebSocketDisconnect as exc:
        reason = f"websocket closed ({exc.code})"
    except RuntimeError as exc:  # pragma: no cover - socket torn down mid-read
        reason = f"socket error: {exc}"
    except asyncio.CancelledError:  # pragma: no cover - gateway shutting down
        reason = "gateway cancelled"
        await state.relay.unregister(runtime_id, reason=reason)
        raise
    except Exception as exc:  # noqa: BLE001 - never leak an exception out of the loop
        reason = f"relay error: {exc}"
    finally:
        await state.relay.unregister(runtime_id, reason=reason)
        emit(f"tunnel closed: {reason}", "warning")


async def _receive_with_idle_timeout(websocket: WebSocket, timeout_s: float) -> str | bytes | None:
    """Wait for the next frame; ``None`` means the socket went quiet (dead VM)."""
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout=timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        return None
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(code=int(message.get("code") or 1000))
    if (text := message.get("text")) is not None:
        return str(text)
    if (data := message.get("bytes")) is not None:
        return data  # type: ignore[return-value]
    return ""


async def _safe_send(websocket: WebSocket, frame: Any) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_text(json.dumps(frame.model_dump(mode="json", exclude_none=True), default=str))


# Keep the exec-response type importable from here for tests that patch the relay.
__all__ = ["AckFrame", "ExecResponseFrame", "RELAY_IDLE_TIMEOUT_S", "router"]
