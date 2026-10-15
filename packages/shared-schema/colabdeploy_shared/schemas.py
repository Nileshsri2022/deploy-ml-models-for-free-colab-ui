"""Pydantic models for every payload that crosses a process boundary.

These are the *contract*. The gateway validates incoming requests against
``ModelSchema`` and the agent reports with the same types, so a drift between the
two shows up as a test failure here rather than a runtime 500 in production.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from colabdeploy_shared.enums import (
    AdapterKind,
    DegradationMode,
    DeploymentState,
    EndpointVisibility,
    JobState,
    RuntimeState,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Wire(BaseModel):
    """Base class: strict-ish JSON in/out with a stable shape."""

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
        extra="ignore",
        ser_json_timedelta="float",
    )


# --------------------------------------------------------------------------- #
# Artifacts & models
# --------------------------------------------------------------------------- #


class ArtifactSpec(Wire):
    """Where the model bytes live.

    ``uri`` schemes understood by the agent:

    * ``file://`` or a bare path - already on the VM (e.g. ``/content/model.pkl``)
    * ``http(s)://``             - plain download
    * ``gdrive://<file_id>``     - Google Drive (agent uses ``gdown``)
    * ``hf://<repo>/<filename>`` - HuggingFace Hub
    * ``store://<artifact_id>``  - uploaded through the UI, served by the gateway
    """

    uri: str
    checksum: str | None = Field(default=None, description="sha256 of the artifact, if known")
    size_bytes: int | None = None
    filename: str | None = None

    @field_validator("uri")
    @classmethod
    def _no_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("artifact uri must not be empty")
        return v.strip()


class ModelSchema(Wire):
    """JSON-schema-ish description of an endpoint's request/response contract.

    Deliberately lightweight: ``fields`` drives the UI form generator and the
    gateway's structural validation, ``example`` doubles as the warm-up sample.
    """

    fields: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    example: dict[str, Any] | None = None
    description: str | None = None

    def validate_payload(self, payload: dict[str, Any]) -> list[str]:
        """Return a list of human-readable validation errors (empty == valid)."""
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ["payload must be a JSON object"]
        for name in self.required:
            if name not in payload:
                errors.append(f"missing required field '{name}'")
        for name, value in payload.items():
            spec = self.fields.get(name)
            if spec is None:
                continue  # unknown fields are allowed unless strict mode is on
            expected = spec.get("type") if isinstance(spec, dict) else None
            if expected and not _type_matches(value, expected):
                errors.append(f"field '{name}' expected {expected}, got {type(value).__name__}")
        return errors


def _type_matches(value: Any, expected: str) -> bool:
    mapping: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list, tuple),
        "object": (dict,),
        "any": (object,),
    }
    types = mapping.get(expected.lower(), (object,))
    if expected.lower() == "integer" and isinstance(value, bool):
        return False
    if expected.lower() == "number" and isinstance(value, bool):
        return False
    return isinstance(value, types)


class ModelCreate(Wire):
    """Payload for ``POST /api/models``."""

    name: str = Field(min_length=1, max_length=120)
    framework: str = "auto"
    adapter: AdapterKind = AdapterKind.AUTO
    artifact: ArtifactSpec
    schema_in: ModelSchema = Field(default_factory=ModelSchema)
    schema_out: ModelSchema = Field(default_factory=ModelSchema)
    warmup_sample: dict[str, Any] | None = None
    entrypoint: str | None = Field(
        default=None, description="python_function adapter: callable name or source snippet"
    )
    tags: list[str] = Field(default_factory=list)


class ModelInfo(Wire):
    """A registered model version. Immutable once created."""

    id: str
    name: str
    version: int = 1
    framework: str = "auto"
    adapter: AdapterKind = AdapterKind.AUTO
    artifact: ArtifactSpec
    schema_in: ModelSchema = Field(default_factory=ModelSchema)
    schema_out: ModelSchema = Field(default_factory=ModelSchema)
    warmup_sample: dict[str, Any] | None = None
    entrypoint: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: str = "registered"
    owner_id: str = "default"
    created_at: datetime = Field(default_factory=utcnow)
    notes: str | None = None


# --------------------------------------------------------------------------- #
# Endpoints & deployments
# --------------------------------------------------------------------------- #


class EndpointCreate(Wire):
    model_id: str
    route: str | None = Field(default=None, description="defaults to the model name, slugified")
    visibility: EndpointVisibility = EndpointVisibility.PUBLIC
    cache_ttl_s: int = Field(default=300, ge=0)
    stale_while_revalidate_s: int = Field(default=3600, ge=0)
    degradation_mode: DegradationMode = DegradationMode.SERVE_CACHED
    timeout_ms: int = Field(default=30000, ge=500, le=300000)
    rate_limit_rpm: int = Field(default=600, ge=1, le=100000)
    max_body_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)


class EndpointUpdate(Wire):
    """Partial update - only the fields you send are changed."""

    model_version: int | None = None
    visibility: EndpointVisibility | None = None
    cache_ttl_s: int | None = Field(default=None, ge=0)
    stale_while_revalidate_s: int | None = Field(default=None, ge=0)
    degradation_mode: DegradationMode | None = None
    timeout_ms: int | None = Field(default=None, ge=500, le=300000)
    rate_limit_rpm: int | None = Field(default=None, ge=1, le=100000)
    status: str | None = None


class EndpointInfo(Wire):
    id: str
    model_id: str
    model_version: int = 1
    route: str
    url: str | None = Field(default=None, description="absolute public URL, filled in by the API")
    visibility: EndpointVisibility = EndpointVisibility.PUBLIC
    cache_ttl_s: int = 300
    stale_while_revalidate_s: int = 3600
    degradation_mode: DegradationMode = DegradationMode.SERVE_CACHED
    timeout_ms: int = 30000
    rate_limit_rpm: int = 600
    max_body_bytes: int = 10 * 1024 * 1024
    status: str = "active"
    created_at: datetime = Field(default_factory=utcnow)
    # Denormalised read-model fields the UI needs on the list page:
    model_name: str | None = None
    adapter: AdapterKind | None = None
    serving_runtime_id: str | None = None
    deployment_state: DeploymentState | None = None
    schema_in: ModelSchema = Field(default_factory=ModelSchema)
    schema_out: ModelSchema = Field(default_factory=ModelSchema)


class DeployRequest(Wire):
    """Pin a model version to an endpoint (and optionally to a specific runtime)."""

    model_id: str
    model_version: int | None = Field(default=None, description="defaults to latest")
    endpoint_id: str | None = Field(default=None, description="create one if omitted")
    runtime_id: str | None = Field(default=None, description="defaults to any online runtime")
    endpoint: EndpointCreate | None = None


class DeploymentInfo(Wire):
    id: str
    endpoint_id: str
    model_id: str
    model_version: int
    runtime_id: str | None = None
    state: DeploymentState = DeploymentState.PENDING
    error: str | None = None
    warm: bool = False
    warmup_ms: float | None = None
    deployed_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Runtimes
# --------------------------------------------------------------------------- #


class RuntimeCapabilities(Wire):
    gpu: bool = False
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    cpu_count: int | None = None
    ram_mb: int | None = None
    cuda_version: str | None = None
    python_version: str | None = None
    agent_version: str | None = None
    disk_free_mb: int | None = None
    adapters: list[str] = Field(default_factory=list)


class HeartbeatRequest(Wire):
    runtime_id: str
    state: RuntimeState = RuntimeState.ONLINE
    capabilities: RuntimeCapabilities = Field(default_factory=RuntimeCapabilities)
    loaded_models: list[str] = Field(default_factory=list)
    queue_depth: int = 0
    inflight: int = 0
    session_started_at: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class RuntimeInfo(Wire):
    id: str
    label: str = "colab-runtime"
    state: RuntimeState = RuntimeState.REGISTERED
    capabilities: RuntimeCapabilities = Field(default_factory=RuntimeCapabilities)
    colab_url: str | None = None
    tunnel_kind: str | None = None
    tunnel_url: str | None = None
    last_heartbeat_at: datetime | None = None
    session_started_at: datetime | None = None
    session_max_hours: float = 12.0
    loaded_models: list[str] = Field(default_factory=list)
    deployments: list[DeploymentInfo] = Field(default_factory=list)
    error: str | None = None
    registered_at: datetime = Field(default_factory=utcnow)
    # Derived, filled in by the API for the UI:
    seconds_until_session_end: float | None = None
    seconds_since_heartbeat: float | None = None


class RuntimeTokenInfo(Wire):
    """Returned exactly once when a token is minted - only the hash is stored."""

    runtime_id: str
    token: str
    label: str
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)


class DesiredDeployment(Wire):
    """One unit of work in the runtime's desired state."""

    deployment_id: str
    endpoint_id: str
    model_id: str
    model_version: int
    adapter: AdapterKind
    artifact: ArtifactSpec
    entrypoint: str | None = None
    schema_in: ModelSchema = Field(default_factory=ModelSchema)
    schema_out: ModelSchema = Field(default_factory=ModelSchema)
    warmup_sample: dict[str, Any] | None = None
    state: DeploymentState = DeploymentState.PENDING


class DesiredState(Wire):
    """What the control plane wants a given Colab VM to be doing right now."""

    runtime_id: str
    generated_at: datetime = Field(default_factory=utcnow)
    revision: int = 0
    deployments: list[DesiredDeployment] = Field(default_factory=list)
    warmup_top_k: int = 20
    settings: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


class PredictRequest(Wire):
    """Body of ``POST /v1/models/{model_id}/predict``.

    ``inputs`` is the canonical shape; ``payload`` is accepted as an alias so
    Gradio/HF-style clients work unchanged.
    """

    inputs: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    bypass_cache: bool = False

    def data(self) -> dict[str, Any]:
        if self.inputs is not None:
            return self.inputs
        if self.payload is not None:
            return self.payload
        return {}


class InferenceError(Wire):
    code: str
    message: str
    detail: dict[str, Any] | None = None


class InferenceResponse(Wire):
    """The stable public response envelope."""

    model_id: str
    endpoint_id: str
    model_version: int
    output: Any = None
    cache: str = "MISS"  # HIT | MISS | STALE | BYPASS
    latency_ms: float = 0.0
    runtime_id: str | None = None
    request_id: str
    job_id: str | None = None
    stale: bool = False
    served_at: datetime = Field(default_factory=utcnow)
    error: InferenceError | None = None


# --------------------------------------------------------------------------- #
# Jobs (async / queued-while-offline)
# --------------------------------------------------------------------------- #


class JobCreate(Wire):
    endpoint_id: str | None = None
    model_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    webhook_url: str | None = None


class JobInfo(Wire):
    id: str
    endpoint_id: str
    model_id: str
    model_version: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    state: JobState = JobState.QUEUED
    attempts: int = 0
    result: Any = None
    error: str | None = None
    webhook_url: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    done_at: datetime | None = None


# --------------------------------------------------------------------------- #
# Cache, logs, metrics, keys, health
# --------------------------------------------------------------------------- #


class CacheEntryInfo(Wire):
    key: str
    endpoint_id: str
    model_version: int
    created_at: datetime
    expires_at: datetime
    stale_at: datetime
    pinned: bool = False
    hits: int = 0
    size_bytes: int = 0
    preview: dict[str, Any] | None = None


class CacheStats(Wire):
    backend: str = "memory"
    entries: int = 0
    hits: int = 0
    misses: int = 0
    stale_hits: int = 0
    evictions: int = 0
    hit_ratio: float = 0.0
    size_bytes: int = 0
    max_entries: int = 0
    pinned: int = 0


class LogLine(Wire):
    ts: datetime = Field(default_factory=utcnow)
    runtime_id: str | None = None
    level: str = "info"
    source: str = "agent"  # agent | gateway
    message: str
    data: dict[str, Any] | None = None


class MetricSample(Wire):
    ts: datetime = Field(default_factory=utcnow)
    name: str
    value: float
    labels: dict[str, str] = Field(default_factory=dict)


class ApiKeyCreate(Wire):
    label: str = "default"
    scopes: list[str] = Field(default_factory=lambda: ["inference", "manage"])
    expires_in_days: int | None = Field(default=None, ge=1)


class ApiKeyInfo(Wire):
    id: str
    label: str
    prefix: str
    scopes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked: bool = False
    token: str | None = Field(default=None, description="only present on creation")


class HealthResponse(Wire):
    status: str = "ok"
    version: str = "0.1.0"
    time: datetime = Field(default_factory=utcnow)


class StatusResponse(Wire):
    """Everything the dashboard needs in one call."""

    healthy: bool = True
    runtimes_total: int = 0
    runtimes_online: int = 0
    runtimes_degraded: int = 0
    models: int = 0
    endpoints: int = 0
    endpoints_serving: int = 0
    deployments_ready: int = 0
    jobs_queued: int = 0
    cache: CacheStats = Field(default_factory=CacheStats)
    requests_total: int = 0
    requests_cached: int = 0
    uptime_s: float = 0.0
    version: str = "0.1.0"
    runtimes: list[RuntimeInfo] = Field(default_factory=list)
