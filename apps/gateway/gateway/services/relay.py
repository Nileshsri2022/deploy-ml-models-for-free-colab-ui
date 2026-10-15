"""The reverse tunnel broker (architecture.md §4.4 / §6).

Colab gives no inbound ingress, so the agent dials out to ``/internal/relay`` and
we multiplex HTTP-shaped request/response pairs over that single socket.

Design notes:

* One socket per runtime, plus a ``ready`` event so the reconciler can react to a
  reconnect immediately instead of waiting for the next poll.
* Requests are tracked in ``_pending`` keyed by frame id; a ``TimeoutError`` is
  raised if the agent does not answer within the caller's budget, which the
  inference service turns into the endpoint's degradation behaviour.
* A per-connection send lock prevents two concurrent requests from interleaving
  frames on the same socket.
* The relay holds no business state - restart the gateway and agents reconnect.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

from colabdeploy_shared.protocol import (
    AGENT_PROTOCOL_VERSION,
    AckFrame,
    DesiredStateFrame,
    ErrorFrame,
    ExecRequestFrame,
    ExecResponseFrame,
    HeartbeatFrame,
    HelloAckFrame,
    HelloFrame,
    LogFrame,
    PingFrame,
    PongFrame,
    parse_frame,
)


class RelayError(RuntimeError):
    """Raised for relay-level failures the caller should surface as an error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RuntimeOffline(RelayError):
    def __init__(self, runtime_id: str) -> None:
        super().__init__("RUNTIME_OFFLINE", f"runtime '{runtime_id}' has no live tunnel")
        self.runtime_id = runtime_id


class AgentTimeout(RelayError):
    def __init__(self, runtime_id: str, timeout_s: float) -> None:
        super().__init__(
            "AGENT_TIMEOUT", f"runtime '{runtime_id}' did not answer within {timeout_s:.1f}s"
        )
        self.runtime_id = runtime_id


@dataclass
class RelayConnection:
    """One live agent socket."""

    runtime_id: str
    ws: WebSocket
    agent_version: str = "0.0.0"
    capabilities: dict[str, Any] = field(default_factory=dict)
    connected_at: float = field(default_factory=time.time)
    last_frame_at: float = field(default_factory=time.time)
    tunnel_url: str | None = None
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def age_s(self) -> float:
        return time.time() - self.connected_at

    async def send(self, frame: Any) -> None:
        payload = frame.model_dump(mode="json", exclude_none=True)
        async with self._send_lock:
            await self.ws.send_text(json.dumps(payload, default=str))


class Relay:
    """Registry of live agent sockets + request multiplexing."""

    def __init__(self) -> None:
        self._conns: dict[str, RelayConnection] = {}
        self._pending: dict[str, asyncio.Future[ExecResponseFrame]] = {}
        self._lock = asyncio.Lock()
        self._ready_events: dict[str, asyncio.Event] = {}
        self.on_heartbeat: Callable[[HeartbeatFrame], Awaitable[None] | None] | None = None
        self.on_log: Callable[[LogFrame], Awaitable[None] | None] | None = None
        self.on_ack: Callable[[AckFrame], Awaitable[None] | None] | None = None
        self.on_disconnect: Callable[[str, str | None], Awaitable[None] | None] | None = None
        self.stats = {"connections": 0, "disconnects": 0, "requests": 0, "timeouts": 0, "errors": 0}

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    async def register(self, conn: RelayConnection) -> None:
        async with self._lock:
            old = self._conns.get(conn.runtime_id)
            if old is not None and old is not conn:
                # A second agent arrived for the same runtime (e.g. the old VM has
                # not noticed it died). Drop the old socket, keep the fresh one.
                with contextlib.suppress(Exception):
                    await old.ws.close(code=4001, reason="superseded")
            self._conns[conn.runtime_id] = conn
            self.stats["connections"] += 1
            event = self._ready_events.setdefault(conn.runtime_id, asyncio.Event())
            event.set()

    async def unregister(self, runtime_id: str, reason: str | None = None) -> None:
        async with self._lock:
            conn = self._conns.pop(runtime_id, None)
            self.stats["disconnects"] += 1
            event = self._ready_events.get(runtime_id)
            if event is not None:
                event.clear()
        # Fail every in-flight request for this runtime instead of letting callers
        # sit on their own timeouts.
        for request_id, future in list(self._pending.items()):
            if getattr(future, "_cdp_runtime", None) == runtime_id and not future.done():
                future.set_exception(RuntimeOffline(runtime_id))
                self._pending.pop(request_id, None)
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.ws.close(code=1000, reason=reason or "closing")
        if self.on_disconnect is not None:
            await _maybe_await(self.on_disconnect(runtime_id, reason))

    def get(self, runtime_id: str) -> RelayConnection | None:
        return self._conns.get(runtime_id)

    def is_connected(self, runtime_id: str) -> bool:
        return runtime_id in self._conns

    @property
    def connected_runtime_ids(self) -> list[str]:
        return sorted(self._conns)

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": [
                {
                    "runtime_id": c.runtime_id,
                    "agent_version": c.agent_version,
                    "connected_s": round(c.age_s, 1),
                    "idle_s": round(time.time() - c.last_frame_at, 1),
                    "tunnel_url": c.tunnel_url,
                }
                for c in self._conns.values()
            ],
            "pending_requests": len(self._pending),
            **self.stats,
        }

    # ------------------------------------------------------------------ #
    # Inbound frames (driven by the WS route's reader loop)
    # ------------------------------------------------------------------ #
    async def handle_inbound(self, runtime_id: str, raw: str | bytes) -> None:
        conn = self._conns.get(runtime_id)
        if conn is not None:
            conn.last_frame_at = time.time()
        try:
            frame = parse_frame(raw)
        except ValueError as exc:
            self.stats["errors"] += 1
            if conn is not None:
                with contextlib.suppress(Exception):
                    await conn.send(ErrorFrame(code="bad_frame", message=str(exc)))
            return

        kind = getattr(frame, "kind", None)
        if kind == "exec_response":
            assert isinstance(frame, ExecResponseFrame)
            self._resolve(frame)
        elif kind == "heartbeat":
            assert isinstance(frame, HeartbeatFrame)
            if self.on_heartbeat is not None:
                await _maybe_await(self.on_heartbeat(frame))
        elif kind == "log":
            assert isinstance(frame, LogFrame)
            if self.on_log is not None:
                await _maybe_await(self.on_log(frame))
        elif kind == "ack":
            assert isinstance(frame, AckFrame)
            if self.on_ack is not None:
                await _maybe_await(self.on_ack(frame))
        elif kind == "ping":
            assert isinstance(frame, PingFrame)
            if conn is not None:
                with contextlib.suppress(Exception):
                    await conn.send(PongFrame(ts=frame.ts))
        elif kind == "error":
            self.stats["errors"] += 1
        # Unknown frames already degrade to ErrorFrame inside parse_frame.

    def _resolve(self, frame: ExecResponseFrame) -> None:
        future = self._pending.pop(frame.request_id, None)
        if future is not None and not future.done():
            future.set_result(frame)

    # ------------------------------------------------------------------ #
    # Outbound: run something on the agent
    # ------------------------------------------------------------------ #
    async def exec_on_runtime(
        self,
        runtime_id: str,
        body: dict[str, Any],
        *,
        path: str = "/execute",
        method: str = "POST",
        timeout_s: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> ExecResponseFrame:
        """Send an exec frame and await the agent's answer."""
        conn = self._conns.get(runtime_id)
        if conn is None:
            raise RuntimeOffline(runtime_id)
        frame = ExecRequestFrame(
            method=method,
            path=path,
            body=body,
            headers=headers or {},
            timeout_ms=int(timeout_s * 1000),
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ExecResponseFrame] = loop.create_future()
        future._cdp_runtime = runtime_id  # type: ignore[attr-defined]
        self._pending[frame.id] = future
        self.stats["requests"] += 1
        try:
            await conn.send(frame)
        except Exception as exc:  # socket died mid-send
            self._pending.pop(frame.id, None)
            await self.unregister(runtime_id, reason=f"send failed: {exc}")
            raise RuntimeOffline(runtime_id) from exc
        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            self._pending.pop(frame.id, None)
            self.stats["timeouts"] += 1
            future.cancel()
            raise AgentTimeout(runtime_id, timeout_s) from exc

    async def push_desired_state(
        self, runtime_id: str, revision: int, deployments: list[dict[str, Any]],
        warmup_top_k: int = 20, settings: dict[str, Any] | None = None,
    ) -> bool:
        conn = self._conns.get(runtime_id)
        if conn is None:
            return False
        frame = DesiredStateFrame(
            runtime_id=runtime_id,
            revision=revision,
            deployments=deployments,
            warmup_top_k=warmup_top_k,
            settings=settings or {},
        )
        try:
            await conn.send(frame)
            return True
        except Exception:  # pragma: no cover - socket died; reader loop will clean up
            return False

    async def push_warmup(self, runtime_id: str, samples: list[dict[str, Any]]) -> bool:
        """Send hot cache keys so the agent primes models with real traffic shapes."""
        conn = self._conns.get(runtime_id)
        if conn is None or not samples:
            return False
        try:
            await conn.send(
                ExecRequestFrame(
                    path="/warmup",
                    body={"samples": samples},
                    timeout_ms=60000,
                    headers={"x-colabdeploy-control": "warmup"},
                )
            )
            return True
        except Exception:  # pragma: no cover
            return False

    async def wait_ready(self, runtime_id: str, timeout_s: float = 0.0) -> bool:
        """Block until the runtime's tunnel is up (used by the deploy flow)."""
        if runtime_id in self._conns:
            return True
        if timeout_s <= 0:
            return False
        event = self._ready_events.setdefault(runtime_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout_s)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False

    async def close_all(self, reason: str = "gateway shutdown") -> None:
        for runtime_id in list(self._conns):
            await self.unregister(runtime_id, reason=reason)


async def _maybe_await(result: Any) -> Any:
    if isinstance(result, Awaitable):
        return await result
    return result


__all__ = [
    "AGENT_PROTOCOL_VERSION",
    "AckFrame",
    "AgentTimeout",
    "DesiredStateFrame",
    "HelloAckFrame",
    "HelloFrame",
    "Relay",
    "RelayConnection",
    "RelayError",
    "RuntimeOffline",
]
