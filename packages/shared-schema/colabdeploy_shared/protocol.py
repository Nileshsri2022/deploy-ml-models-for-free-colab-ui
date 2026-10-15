"""WebSocket framing between the gateway relay and the Colab agent.

Colab has no inbound ingress, so the agent dials out and we multiplex HTTP-shaped
request/response pairs over a single socket. Frames are JSON objects with a ``kind``
discriminator; bodies are base64 when binary.

    gateway ──exec──▶ agent ──(runs adapter locally)──▶ gateway ──result──▶ caller

Keep the frame set tiny and versioned: ``AGENT_PROTOCOL_VERSION`` is checked on
hello so an old agent gets a clear error instead of undefined behaviour.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

AGENT_PROTOCOL_VERSION = 1

MAX_FRAME_BYTES = 32 * 1024 * 1024


class _Frame(BaseModel):
    model_config = ConfigDict(extra="ignore", use_enum_values=True)

    kind: str
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class HelloFrame(_Frame):
    """First frame the agent sends after the socket opens."""

    kind: Literal["hello"] = "hello"
    protocol: int = AGENT_PROTOCOL_VERSION
    runtime_id: str
    token: str
    agent_version: str = "0.1.0"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    colab_url: str | None = None
    session_started_at: str | None = None


class HelloAckFrame(_Frame):
    kind: Literal["hello_ack"] = "hello_ack"
    ok: bool = True
    runtime_id: str | None = None
    desired_state_revision: int = 0
    error: str | None = None
    server_version: str = "0.1.0"


class ExecRequestFrame(_Frame):
    """An HTTP-ish request the gateway wants the agent to execute locally."""

    kind: Literal["exec"] = "exec"
    method: str = "POST"
    path: str = "/execute"
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    body_b64: str | None = None
    timeout_ms: int = 30000

    def body_bytes(self) -> bytes:
        if self.body_b64:
            return base64.b64decode(self.body_b64)
        if self.body is not None:
            return json.dumps(self.body).encode("utf-8")
        return b""


class ExecResponseFrame(_Frame):
    """The agent's answer to an ``exec`` frame."""

    kind: Literal["exec_response"] = "exec_response"
    request_id: str
    status: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    body_b64: str | None = None
    error: str | None = None
    agent_latency_ms: float = 0.0

    @classmethod
    def from_bytes(
        cls,
        request_id: str,
        status: int,
        payload: bytes,
        headers: dict[str, str] | None = None,
        agent_latency_ms: float = 0.0,
        error: str | None = None,
    ) -> ExecResponseFrame:
        body: dict[str, Any] | None = None
        body_b64: str | None = None
        try:
            decoded = json.loads(payload) if payload else None
            if isinstance(decoded, dict):
                body = decoded
            else:
                body = {"result": decoded}
        except (ValueError, UnicodeDecodeError):
            body_b64 = base64.b64encode(payload).decode("ascii")
        return cls(
            request_id=request_id,
            status=status,
            body=body,
            body_b64=body_b64,
            headers=headers or {},
            agent_latency_ms=agent_latency_ms,
            error=error,
        )

    def payload_bytes(self) -> bytes:
        if self.body_b64:
            return base64.b64decode(self.body_b64)
        if self.body is not None:
            return json.dumps(self.body).encode("utf-8")
        return b""


class HeartbeatFrame(_Frame):
    kind: Literal["heartbeat"] = "heartbeat"
    runtime_id: str
    state: str = "online"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    loaded_models: list[str] = Field(default_factory=list)
    deployments: list[dict[str, Any]] = Field(default_factory=list)
    queue_depth: int = 0
    inflight: int = 0
    session_started_at: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class DesiredStateFrame(_Frame):
    """Gateway → agent: reconcile towards this."""

    kind: Literal["desired_state"] = "desired_state"
    runtime_id: str
    revision: int = 0
    deployments: list[dict[str, Any]] = Field(default_factory=list)
    warmup_top_k: int = 20
    settings: dict[str, Any] = Field(default_factory=dict)


class AckFrame(_Frame):
    kind: Literal["ack"] = "ack"
    runtime_id: str
    deployment_states: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None


class LogFrame(_Frame):
    kind: Literal["log"] = "log"
    runtime_id: str
    lines: list[dict[str, Any]] = Field(default_factory=list)


class PingFrame(_Frame):
    kind: Literal["ping"] = "ping"
    ts: float = 0.0


class PongFrame(_Frame):
    kind: Literal["pong"] = "pong"
    ts: float = 0.0


class ErrorFrame(_Frame):
    kind: Literal["error"] = "error"
    code: str = "error"
    message: str = ""
    runtime_id: str | None = None


Frame = Union[
    HelloFrame,
    HelloAckFrame,
    ExecRequestFrame,
    ExecResponseFrame,
    HeartbeatFrame,
    DesiredStateFrame,
    AckFrame,
    LogFrame,
    PingFrame,
    PongFrame,
    ErrorFrame,
]

FrameKind = type[Frame]

_REGISTRY: dict[str, FrameKind] = {cls.model_fields["kind"].default: cls for cls in Frame.__args__}  # type: ignore[attr-defined]


def parse_frame(raw: str | bytes) -> Frame:
    """Decode a wire frame into its typed model.

    Raises ``ValueError`` for malformed input; unknown kinds become ``ErrorFrame``
    so a version skew degrades instead of dropping the socket.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError(f"frame too large ({len(raw)} bytes)")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"frame is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("frame must be a JSON object")
    kind = data.get("kind")
    cls = _REGISTRY.get(str(kind))
    if cls is None:
        return ErrorFrame(code="unknown_frame", message=f"unknown frame kind: {kind!r}")
    return cls.model_validate(data)


def frame_kinds() -> list[str]:
    return sorted(_REGISTRY)
