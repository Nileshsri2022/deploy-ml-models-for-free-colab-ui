"""The outbound WebSocket tunnel - how a Colab VM becomes reachable.

Colab has no inbound ingress, so the agent always dials *out*. One socket carries
everything: the hello handshake, ``exec`` requests from the gateway, heartbeats,
desired-state pushes, log batches and warm-up samples.

Threading model (deliberately simple, because it runs inside a notebook):

* one **async task** owns the socket: receive loop + heartbeat sender;
* ``exec`` frames are handed to a small **ThreadPoolExecutor**, so a slow model
  never blocks heartbeats (a blocked heartbeat is how a VM gets marked offline);
* reconnect uses exponential backoff with jitter, forever, until ``stop()``.

Reconnect is the feature: kill the notebook, reopen it, run the same cell, and the
runtime re-registers, re-hydrates from the on-disk artifact cache and drains
whatever the gateway queued while it was gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from colabdeploy_shared.protocol import (
    AGENT_PROTOCOL_VERSION,
    AckFrame,
    DesiredStateFrame,
    ErrorFrame,
    ExecRequestFrame,
    ExecResponseFrame,
    HeartbeatFrame,
    HelloFrame,
    LogFrame,
    PingFrame,
    PongFrame,
    parse_frame,
)
from colabdeploy_shared.schemas import RuntimeCapabilities

from .bootstrap import Environment

LogFn = Callable[[str, str, dict[str, Any]], None]


def _noop_log(message: str, level: str = "info", **data: Any) -> None:
    prefix = {"error": "!! ", "warning": "! ", "debug": ". "}.get(level, "")
    print(f"[colabdeploy] {prefix}{message}")


@dataclass
class TunnelStats:
    connects: int = 0
    disconnects: int = 0
    requests: int = 0
    errors: int = 0
    timeouts: int = 0
    last_connected_at: float | None = None
    last_error: str | None = None
    bytes_in: int = 0
    bytes_out: int = 0
    reconnect_attempts: int = 0

    def as_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        if self.last_connected_at:
            data["connected_s"] = round(time.time() - self.last_connected_at, 1)
        return data


@dataclass
class TunnelConfig:
    relay_url: str
    runtime_id: str
    token: str
    agent_version: str = "0.1.0"
    heartbeat_interval_s: float = 15.0
    reconnect_min_s: float = 1.0
    reconnect_max_s: float = 60.0
    max_workers: int = 4
    max_frame_bytes: int = 32 * 1024 * 1024
    exec_timeout_s: float = 600.0
    log_batch_size: int = 50
    log_flush_s: float = 5.0
    colab_url: str | None = None
    session_started_at: float = field(default_factory=time.time)


class TunnelClient:
    """Owns the socket, the heartbeat and the reconnect loop."""

    def __init__(
        self,
        config: TunnelConfig,
        env: Environment,
        handle_exec: Callable[[str, dict[str, Any], str], tuple[int, dict[str, Any]]],
        status_fn: Callable[[], dict[str, Any]],
        on_desired_state: Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]],
        *,
        log: LogFn | None = None,
    ) -> None:
        self.config = config
        self.env = env
        self._handle_exec = handle_exec
        self._status_fn = status_fn
        self._on_desired_state = on_desired_state
        self.log = log or _noop_log
        self.stats = TunnelStats()
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers, thread_name_prefix="cdp-exec"
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._connected = threading.Event()
        self._ws: Any = None
        self._log_buffer: list[dict[str, Any]] = []
        self._log_lock = threading.Lock()
        self._desired_revision = 0
        # Our own counters: ``ThreadPoolExecutor`` queue internals are private and
        # were renamed across CPython versions, so never introspect them.
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public control
    # ------------------------------------------------------------------ #
    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> None:
        """Run the tunnel in a daemon thread (the notebook cell returns)."""
        if self._thread is not None and self._thread.is_alive():
            self.log("tunnel already running", "warning")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="cdp-tunnel", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(self._close_socket("agent stopping"), self._loop)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._connected.clear()

    def wait_connected(self, timeout_s: float = 30.0) -> bool:
        return self._connected.wait(timeout_s)

    def snapshot(self) -> dict[str, Any]:
        return {
            "connected": self.connected,
            "relay_url": self.config.relay_url,
            "runtime_id": self.config.runtime_id,
            "desired_revision": self._desired_revision,
            **self.stats.as_dict(),
        }

    # ------------------------------------------------------------------ #
    # Logging (batched to keep the tunnel quiet)
    # ------------------------------------------------------------------ #
    def queue_log(self, message: str, level: str = "info", **data: Any) -> None:
        self.log(message, level, **data)
        entry = {"level": level, "message": message, "ts": time.time(), "data": data or None}
        with self._log_lock:
            self._log_buffer.append(entry)

    async def _flush_logs(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.config.log_flush_s)
            with self._log_lock:
                batch, self._log_buffer = self._log_buffer[: self.config.log_batch_size], []
            if batch and self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._send(LogFrame(runtime_id=self.config.runtime_id, lines=batch))

    # ------------------------------------------------------------------ #
    # Thread + loop plumbing
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:  # noqa: BLE001 - the thread must not die silently
            self.log(f"tunnel crashed: {type(exc).__name__}: {exc}", "error")

    async def _run_async(self) -> None:
        self._loop = asyncio.get_running_loop()
        flush_task = asyncio.create_task(self._flush_logs())
        backoff = self.config.reconnect_min_s
        try:
            while not self._stop.is_set():
                try:
                    await self._connect_and_serve()
                    backoff = self.config.reconnect_min_s
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.stats.last_error = f"{type(exc).__name__}: {exc}"
                    self.stats.errors += 1
                    self.log(f"tunnel error: {self.stats.last_error}", "warning")
                finally:
                    self._connected.clear()
                    self._ws = None
                if self._stop.is_set():
                    break
                self.stats.reconnect_attempts += 1
                jitter = random.uniform(0, backoff * 0.25)
                delay = min(backoff + jitter, self.config.reconnect_max_s)
                self.log(f"reconnecting in {delay:.1f}s (attempt {self.stats.reconnect_attempts})", "warning")
                await _sleep_interruptible(delay, self._stop)
                backoff = min(backoff * 2, self.config.reconnect_max_s)
        finally:
            flush_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await flush_task

    async def _connect_and_serve(self) -> None:
        import websockets

        url = self.config.relay_url
        self.log(f"connecting to {url} …")
        async with websockets.connect(
            url,
            max_size=self.config.max_frame_bytes,
            ping_interval=20.0,
            ping_timeout=60.0,
            close_timeout=5.0,
            open_timeout=30.0,
        ) as ws:
            self._ws = ws
            await self._send(
                HelloFrame(
                    protocol=AGENT_PROTOCOL_VERSION,
                    runtime_id=self.config.runtime_id,
                    token=self.config.token,
                    agent_version=self.config.agent_version,
                    capabilities=self._capabilities(),
                    colab_url=self.config.colab_url,
                    session_started_at=_iso(self.config.session_started_at),
                )
            )
            self.stats.connects += 1
            self.stats.last_connected_at = time.time()
            self._connected.set()
            self.log("tunnel established")
            heartbeat = asyncio.create_task(self._heartbeat_loop())

            def _heartbeat_crashed(task: asyncio.Task[Any]) -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    self.stats.last_error = f"heartbeat crashed: {exc!r}"
                    self.log(self.stats.last_error, "error")

            heartbeat.add_done_callback(_heartbeat_crashed)
            try:
                await self._receive_loop(ws)
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
                self.stats.disconnects += 1

    def _capabilities(self) -> dict[str, Any]:
        caps: RuntimeCapabilities = self.env.capabilities(
            self.config.agent_version, self._loaded_adapters()
        )
        data = caps.model_dump(mode="json")
        data["tunnel_kind"] = "ws-relay"
        return data

    def _loaded_adapters(self) -> list[str]:
        try:
            from .adapters import available_adapters

            return available_adapters()
        except Exception:  # noqa: BLE001
            return []

    async def _receive_loop(self, ws: Any) -> None:
        async for raw in ws:
            if self._stop.is_set():
                break
            if isinstance(raw, (bytes, bytearray)):
                self.stats.bytes_in += len(raw)
                raw = raw.decode("utf-8", "replace")
            else:
                self.stats.bytes_in += len(raw)
            try:
                frame = parse_frame(raw)
            except ValueError as exc:
                self.log(f"bad frame from gateway: {exc}", "warning")
                continue
            await self._dispatch(frame)

    async def _dispatch(self, frame: Any) -> None:
        kind = getattr(frame, "kind", None)
        if kind == "exec":
            self.stats.requests += 1
            # Run in the pool: a slow GPU model must not block heartbeats.
            self._submit(self._exec_worker, frame)
        elif kind == "desired_state":
            self._submit(self._desired_state_worker, frame)
        elif kind == "hello_ack":
            self._desired_revision = int(getattr(frame, "desired_state_revision", 0) or 0)
            if not getattr(frame, "ok", True):
                self.log(f"gateway rejected hello: {frame.error}", "error")
        elif kind == "ping":
            await self._send(PongFrame(ts=frame.ts))
        elif kind == "error":
            self.log(f"gateway error [{frame.code}]: {frame.message}", "error")
        elif kind == "exec_response":  # pragma: no cover - gateway-only frame
            pass

    # ------------------------------------------------------------------ #
    # Workers (thread pool)
    # ------------------------------------------------------------------ #
    def _submit(self, worker: Callable[..., None], frame: Any) -> None:
        """Run ``worker`` in the pool while tracking how many are in flight."""
        with self._inflight_lock:
            self._inflight += 1

        def run() -> None:
            try:
                worker(frame)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not kill the tunnel
                self.stats.errors += 1
                self.queue_log(f"worker error: {type(exc).__name__}: {exc}", "error")
            finally:
                with self._inflight_lock:
                    self._inflight -= 1

        self._executor.submit(run)

    @property
    def inflight(self) -> int:
        with self._inflight_lock:
            return self._inflight

    def _exec_worker(self, frame: ExecRequestFrame) -> None:
        started = time.perf_counter()
        body = frame.body or {}
        if frame.body_b64 and not body:
            with contextlib.suppress(Exception):
                body = json.loads(frame.body_bytes().decode("utf-8"))
        try:
            status, payload = self._handle_exec(frame.path, body, frame.method)
        except Exception as exc:  # noqa: BLE001
            status, payload = 500, {"code": "AGENT_ERROR", "error": f"{type(exc).__name__}: {exc}"}
        elapsed = (time.perf_counter() - started) * 1000
        response = ExecResponseFrame.from_bytes(
            request_id=frame.id,
            status=status,
            payload=json.dumps(payload, default=str).encode("utf-8"),
            headers={"x-agent-runtime": self.config.runtime_id},
            agent_latency_ms=round(elapsed, 3),
        )
        self._send_threadsafe(response)
        if status >= 400:
            self.stats.errors += 1
            self.queue_log(f"{frame.path} -> {status}: {str(payload.get('error'))[:200]}", "warning")

    def _desired_state_worker(self, frame: DesiredStateFrame) -> None:
        self._desired_revision = frame.revision
        started = time.perf_counter()
        try:
            results = self._on_desired_state(
                {"revision": frame.revision, "settings": frame.settings,
                 "warmup_top_k": frame.warmup_top_k},
                frame.deployments,
            )
        except Exception as exc:  # noqa: BLE001
            self.queue_log(f"reconcile failed: {exc}", "error")
            results = [{"deployment_id": "", "state": "failed", "error": str(exc)}]
        elapsed = (time.perf_counter() - started) * 1000
        failed = [r for r in results if str(r.get("state")) == "failed"]
        message = (
            f"reconciled {len(results)} deployment(s) in {elapsed:.0f} ms"
            + (f"; {len(failed)} failed" if failed else "")
        )
        self._send_threadsafe(
            AckFrame(
                runtime_id=self.config.runtime_id,
                deployment_states=results,
                message=message,
            )
        )
        self.queue_log(message, "error" if failed else "info")

    # ------------------------------------------------------------------ #
    # Heartbeat
    # ------------------------------------------------------------------ #
    async def _heartbeat_loop(self) -> None:
        interval = max(5.0, self.config.heartbeat_interval_s)
        while not self._stop.is_set():
            try:
                status = self._status_fn()
                frame = HeartbeatFrame(
                    runtime_id=self.config.runtime_id,
                    state=str(status.get("state", "online")),
                    capabilities=RuntimeCapabilities.model_validate(
                        status.get("capabilities") or {}
                    ).model_dump(mode="json"),
                    loaded_models=list(status.get("loaded_models") or []),
                    queue_depth=self.inflight,
                    inflight=self.inflight,
                    session_started_at=_iso(self.config.session_started_at),
                    metrics={
                        "uptime_s": float(status.get("uptime_s") or 0.0),
                        "executions": float(status.get("counters", {}).get("executions", 0)),
                        "errors": float(status.get("counters", {}).get("errors", 0)),
                        "disk_free_mb": float(self.env.disk_free_mb or 0),
                    },
                )
                await self._send(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                # A dead heartbeat is how a healthy VM gets marked offline, so this
                # is loud, and we keep trying instead of giving up on the session.
                self.stats.errors += 1
                self.log(f"heartbeat failed: {type(exc).__name__}: {exc}", "error")
            await _sleep_interruptible(interval, self._stop)

    # ------------------------------------------------------------------ #
    # Send helpers
    # ------------------------------------------------------------------ #
    async def _send(self, frame: Any) -> None:
        ws = self._ws
        if ws is None:
            return
        text = json.dumps(frame.model_dump(mode="json", exclude_none=True), default=str)
        self.stats.bytes_out += len(text)
        await ws.send(text)

    def _send_threadsafe(self, frame: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        future = asyncio.run_coroutine_threadsafe(self._send(frame), loop)
        with contextlib.suppress(Exception):
            future.result(timeout=30.0)

    async def _close_socket(self, reason: str) -> None:
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close(code=1000, reason=reason)


async def _sleep_interruptible(seconds: float, stop: threading.Event) -> None:
    """Sleep in slices so ``stop()`` takes effect quickly."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        if stop.is_set():
            return
        await asyncio.sleep(min(0.25, max(0.0, deadline - time.time())))


def _iso(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def ws_url_from_http(gateway_url: str) -> str:
    """``https://host`` -> ``wss://host/internal/relay``."""
    base = gateway_url.strip().rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    elif base.startswith(("ws://", "wss://")):
        pass
    else:
        base = "wss://" + base
    if base.endswith("/internal/relay"):
        return base
    return base + "/internal/relay"
