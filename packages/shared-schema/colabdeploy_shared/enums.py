"""Enumerations shared across the wire protocol."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """str enum that serialises to its plain value in JSON and in f-strings."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class AdapterKind(StrEnum):
    """Pluggable model adapter implementations (see agent/colabdeploy/adapters)."""

    AUTO = "auto"
    ECHO = "echo"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    PYTORCH = "pytorch"
    KERAS = "keras"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"
    PYTHON_FUNCTION = "python_function"


class RuntimeState(StrEnum):
    """Lifecycle of a Colab VM as seen by the control plane."""

    UNKNOWN = "unknown"
    REGISTERED = "registered"  # token issued, no agent connected yet
    CONNECTING = "connecting"  # agent said hello, hydrating artifacts
    ONLINE = "online"  # heartbeat fresh, models loaded, serving
    DEGRADED = "degraded"  # connected but models failed to load / overloaded
    STALE = "stale"  # heartbeat missed once or twice
    OFFLINE = "offline"  # tunnel closed or heartbeat gap exceeded


class DeploymentState(StrEnum):
    """State of "model version X is served by runtime Y"."""

    PENDING = "pending"  # in desired-state, agent has not acked
    LOADING = "loading"  # agent is downloading/warming
    READY = "ready"  # serving
    FAILED = "failed"  # load error surfaced to the UI
    UNLOADED = "unloaded"


class EndpointVisibility(StrEnum):
    PUBLIC = "public"  # reachable with any valid caller API key
    PRIVATE = "private"  # owner key only


class DegradationMode(StrEnum):
    """What the gateway does when no runtime can serve a request.

    This is the heart of the resilience story: the endpoint survives the VM.
    """

    SERVE_CACHED = "serve_cached"  # return a stale cache hit, flagged as stale
    QUEUE = "queue"  # 202 + job_id, drain when the runtime comes back
    FAIL = "fail"  # 503 + Retry-After


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
