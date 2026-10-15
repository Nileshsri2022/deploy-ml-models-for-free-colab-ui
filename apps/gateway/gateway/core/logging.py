"""Bounded in-memory log/metric rings plus a Server-Sent Events fan-out hub.

The UI live-tails runtime logs over SSE (``GET /api/logs/stream``). SSE is used
instead of WebSocket because it survives proxies, needs no client library, and the
data flow is one-directional - exactly what a log tail needs.

Both rings are bounded and drop the oldest entry: a runaway agent must never be
able to exhaust the control plane's memory.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from colabdeploy_shared.schemas import LogLine, MetricSample, utcnow


class RingBuffer:
    """Fixed-capacity deque with cheap filtered reads."""

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = max(16, capacity)
        self._items: deque[Any] = deque(maxlen=self.capacity)

    def append(self, item: Any) -> None:
        self._items.append(item)

    def extend(self, items: list[Any]) -> None:
        for item in items:
            self._items.append(item)

    def snapshot(self, limit: int = 200, **filters: Any) -> list[Any]:
        out = []
        for item in reversed(self._items):
            if limit and len(out) >= limit:
                break
            if filters and not all(getattr(item, k, None) == v for k, v in filters.items()):
                continue
            out.append(item)
        return out

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class Telemetry:
    """Logs + metrics + derived request counters for the dashboard."""

    logs: RingBuffer = field(default_factory=lambda: RingBuffer(5000))
    metrics: RingBuffer = field(default_factory=lambda: RingBuffer(20000))
    requests_total: int = 0
    requests_cached: int = 0
    requests_failed: int = 0
    requests_stale: int = 0
    requests_queued: int = 0
    latency_ms: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    started_at: float = field(default_factory=time.time)

    def log(self, message: str, *, level: str = "info", source: str = "gateway",
            runtime_id: str | None = None, **data: Any) -> LogLine:
        line = LogLine(
            level=level,
            source=source,
            runtime_id=runtime_id,
            message=message,
            data=data or None,
        )
        self.logs.append(line)
        return line

    def ingest_agent_logs(self, runtime_id: str, lines: list[dict[str, Any]]) -> int:
        added = 0
        for raw in lines:
            try:
                line = LogLine.model_validate({**raw, "runtime_id": runtime_id, "source": "agent"})
            except Exception:
                line = LogLine(
                    runtime_id=runtime_id,
                    source="agent",
                    level="info",
                    message=str(raw)[:2000],
                )
            self.logs.append(line)
            added += 1
        return added

    def record_request(self, *, cached: bool, stale: bool, failed: bool, latency_ms: float) -> None:
        self.requests_total += 1
        if cached:
            self.requests_cached += 1
        if stale:
            self.requests_stale += 1
        if failed:
            self.requests_failed += 1
        self.latency_ms.append(max(0.0, latency_ms))
        self.metrics.append(
            MetricSample(name="request_latency_ms", value=latency_ms, labels={"cached": str(cached)})
        )

    def record_queued(self) -> None:
        self.requests_queued += 1

    def percentile(self, p: float) -> float:
        if not self.latency_ms:
            return 0.0
        ordered = sorted(self.latency_ms)
        idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    def summary(self) -> dict[str, Any]:
        return {
            "requests_total": self.requests_total,
            "requests_cached": self.requests_cached,
            "requests_failed": self.requests_failed,
            "requests_stale": self.requests_stale,
            "requests_queued": self.requests_queued,
            "cache_hit_ratio": round(self.requests_cached / self.requests_total, 4)
            if self.requests_total
            else 0.0,
            "latency_p50_ms": self.percentile(50),
            "latency_p95_ms": self.percentile(95),
            "latency_p99_ms": self.percentile(99),
            "uptime_s": round(time.time() - self.started_at, 1),
        }


class SseHub:
    """Fan-out of log lines to any number of SSE subscribers.

    Each subscriber gets its own bounded queue; a slow browser is dropped rather
    than applying back-pressure to the agent that produced the logs.
    """

    def __init__(self, max_subscribers: int = 64, queue_size: int = 500) -> None:
        self.max_subscribers = max_subscribers
        self.queue_size = queue_size
        self._queues: set[asyncio.Queue[str]] = set()
        self._lock = asyncio.Lock()

    @property
    def subscribers(self) -> int:
        return len(self._queues)

    async def subscribe(self) -> asyncio.Queue[str]:
        async with self._lock:
            queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.queue_size)
            if len(self._queues) >= self.max_subscribers:
                # Evict an arbitrary (oldest-inserted) subscriber to stay bounded.
                victim = next(iter(self._queues))
                self._queues.discard(victim)
                victim.put_nowait(_sse({"event": "error", "message": "too many subscribers"}))
            self._queues.add(queue)
            return queue

    async def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        async with self._lock:
            self._queues.discard(queue)

    def publish(self, line: LogLine) -> None:
        payload = _sse(
            {
                "event": "log",
                "ts": line.ts.isoformat(),
                "level": line.level,
                "source": line.source,
                "runtime_id": line.runtime_id,
                "message": line.message,
                "data": line.data,
            }
        )
        for queue in list(self._queues):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                # Drop the oldest event for this slow consumer and retry once.
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                    pass

    async def stream(self, queue: asyncio.Queue[str], heartbeat_s: float = 15.0) -> AsyncIterator[str]:
        try:
            yield _sse({"event": "ready", "ts": utcnow().isoformat()})
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
                except (asyncio.TimeoutError, TimeoutError):
                    yield ": keep-alive\n\n"  # comment frame keeps proxies from closing
                    continue
                yield item
        finally:
            await self.unsubscribe(queue)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
