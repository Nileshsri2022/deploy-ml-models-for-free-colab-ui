"""The adapter protocol - the only framework-specific code in the system.

Everything else (tunnel, cache, registry, UI) speaks JSON. An adapter turns a
JSON payload into a model call and back, and describes itself so the gateway can
validate requests and the UI can render a form:

    class MyAdapter(ModelAdapter):
        kind = AdapterKind.SKLEARN
        def load(...); def predict(...); def describe(...); ...

Rules that keep "deploy any model" honest:

* **Lazy imports.** ``import torch`` happens inside ``load()``, so a Colab VM
  without TensorFlow can still serve a scikit-learn model.
* **Pure JSON in/out.** No numpy arrays cross the wire; adapters convert.
* **Fail loudly at load time.** A model that cannot load must never look deployed -
  the error travels back to the UI through the ack frame.
* **``detect()`` is cheap.** It inspects names/bytes, never loads the model, so the
  ranked probe in :mod:`colabdeploy.adapters.registry` stays fast.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema


class AdapterError(RuntimeError):
    """Raised when a model cannot be loaded or a payload cannot be served."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return f"{self.message}" + (f" (hint: {self.hint})" if self.hint else "")


@dataclass
class ModelMeta:
    """What the control plane told us about the model we are about to load."""

    model_id: str
    model_version: int
    name: str = ""
    adapter: str = AdapterKind.AUTO.value
    artifact: ArtifactSpec = field(default_factory=lambda: ArtifactSpec(uri=""))
    schema_in: ModelSchema = field(default_factory=ModelSchema)
    schema_out: ModelSchema = field(default_factory=ModelSchema)
    entrypoint: str | None = None
    warmup_sample: dict[str, Any] | None = None
    endpoint_id: str = ""
    deployment_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Identity used in ``loaded_models`` reports and the local model pool."""
        return f"{self.model_id}@{self.model_version}"

    @property
    def adapter_kind(self) -> str:
        """The adapter kind as a plain string, whether it arrived as an enum or str."""
        from colabdeploy_shared.schemas import enum_value

        return enum_value(self.adapter)

    @classmethod
    def from_deployment(cls, dep: dict[str, Any]) -> ModelMeta:
        artifact = dep.get("artifact") or {}
        return cls(
            model_id=str(dep.get("model_id", "")),
            model_version=int(dep.get("model_version", 1)),
            name=str(dep.get("name") or dep.get("model_id", "")),
            adapter=str(dep.get("adapter", AdapterKind.AUTO.value)),
            artifact=ArtifactSpec.model_validate(artifact) if artifact else ArtifactSpec(uri=""),
            schema_in=ModelSchema.model_validate(dep.get("schema_in") or {}),
            schema_out=ModelSchema.model_validate(dep.get("schema_out") or {}),
            entrypoint=dep.get("entrypoint"),
            warmup_sample=dep.get("warmup_sample"),
            endpoint_id=str(dep.get("endpoint_id", "")),
            deployment_id=str(dep.get("deployment_id", "")),
            extra={k: v for k, v in dep.items() if k not in _KNOWN_FIELDS},
        )


_KNOWN_FIELDS = {
    "model_id",
    "model_version",
    "name",
    "adapter",
    "artifact",
    "schema_in",
    "schema_out",
    "entrypoint",
    "warmup_sample",
    "endpoint_id",
    "deployment_id",
}


@dataclass
class Health:
    ok: bool = True
    detail: str = ""
    loaded_at: float | None = None
    warmup_ms: float | None = None
    predictions: int = 0
    errors: int = 0
    last_error: str | None = None


class ModelAdapter(ABC):
    """Base class every adapter extends."""

    kind: AdapterKind = AdapterKind.AUTO
    #: lower = tried first when ``adapter == auto``
    priority: int = 100
    #: file extensions this adapter is confident about
    extensions: tuple[str, ...] = ()
    #: framework modules that must import for this adapter to work. A ``"|"``
    #: inside a name means "any of these" (e.g. ``keras|tensorflow``).
    framework: tuple[str, ...] = ()

    @property
    def kind_value(self) -> str:
        """This adapter's kind as a plain string (``self.kind`` may be an enum)."""
        from colabdeploy_shared.schemas import enum_value

        return enum_value(self.kind)

    def __init__(self) -> None:
        self.model: Any = None
        self.meta: ModelMeta | None = None
        self.artifact_path: Path | None = None
        self.health = Health()
        self._loaded_at: float | None = None

    # ------------------------------------------------------------------ #
    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        """Cheap, load-free guess: does this artifact look like ours?"""
        name = (artifact.filename or artifact.uri.rsplit("/", 1)[-1]).lower()
        return any(name.endswith(ext) for ext in cls.extensions)

    @abstractmethod
    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        """Load the model into memory. Raise :class:`AdapterError` on failure."""

    @abstractmethod
    def predict(self, payload: dict[str, Any]) -> Any:
        """Run inference on one JSON payload and return JSON-serialisable output."""

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[Any]:
        """Default: loop. Override when the framework batches better than that."""
        return [self.predict(p) for p in payloads]

    def describe(self) -> ModelSchema:
        """Input/output contract, used when the deployer did not supply one."""
        return ModelSchema(description=f"{self.kind.value} adapter")

    def warmup(self, sample: dict[str, Any] | None = None) -> float:
        """Prime CUDA graphs / tokenizers / JIT. Returns elapsed milliseconds."""
        if sample is None:
            return 0.0
        started = time.perf_counter()
        try:
            self.predict(sample)
        except Exception as exc:  # noqa: BLE001 - warm-up must never kill a deploy
            self.health.last_error = f"warmup failed: {exc}"
            return round((time.perf_counter() - started) * 1000, 2)
        return round((time.perf_counter() - started) * 1000, 2)

    def unload(self) -> None:
        self.model = None
        self.meta = None
        self._loaded_at = None

    # ------------------------------------------------------------------ #
    def mark_loaded(self, meta: ModelMeta, artifact_path: Path) -> None:
        self.meta = meta
        self.artifact_path = artifact_path
        self._loaded_at = time.time()
        self.health = Health(ok=True, loaded_at=self._loaded_at, detail=f"{self.kind.value} loaded")

    def record_prediction(self, ok: bool, error: str | None = None) -> None:
        if ok:
            self.health.predictions += 1
        else:
            self.health.errors += 1
            self.health.last_error = error

    def to_json(self, value: Any) -> Any:
        """Convert numpy/torch scalars into plain JSON types (recursive)."""
        return jsonable(value)


def jsonable(value: Any) -> Any:
    """Recursively convert model outputs into JSON-safe primitives.

    Handles numpy scalars/arrays, torch tensors, pandas Series/DataFrames and
    anything else exposing ``tolist()``/``item()`` - the single reason adapters
    can stay short.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    # numpy / torch / jax scalars
    item = getattr(value, "item", None)
    if callable(item) and not hasattr(value, "shape"):
        try:
            return jsonable(item())
        except Exception:  # noqa: BLE001
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return jsonable(tolist())
        except Exception:  # noqa: BLE001
            pass

    # pandas DataFrame / Series
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        try:
            return {"columns": list(map(str, value.columns)), "data": jsonable(value.values.tolist())}
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "to_dict") and hasattr(value, "index"):
        try:
            return jsonable(value.to_dict())
        except Exception:  # noqa: BLE001
            pass

    return str(value)


def payload_table(payload: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Normalise the tabular input shapes people actually send.

    Accepted::

        {"columns": ["a","b"], "data": [[1,2]]}
        {"instances": [{"a":1,"b":2}]}
        {"inputs": [{"a":1,"b":2}]}
        [{"a":1,"b":2}]                    # via {"inputs": [...]}
        {"features": [[1,2]]}               # column order from the schema/example
    """
    if "columns" in payload and "data" in payload:
        return list(map(str, payload["columns"])), [list(row) for row in payload["data"]]

    rows: Any = None
    for key in ("instances", "inputs", "data", "features", "rows", "X"):
        if key in payload:
            rows = payload[key]
            break
    if rows is None:
        # A single flat object: treat it as one row.
        rows = [payload]

    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, (list, tuple)):
        raise AdapterError(
            "could not find tabular input; expected 'instances', 'columns'+'data' or a list",
            hint='try {"instances": [{"feature": 1.0}]}',
        )
    if rows and isinstance(rows[0], dict):
        columns = list(rows[0].keys())
        return columns, [[row.get(c) for c in columns] for row in rows]
    if rows and isinstance(rows[0], (list, tuple)):
        columns = [f"f{i}" for i in range(len(rows[0]))]
        return columns, [list(row) for row in rows]
    raise AdapterError("empty or malformed tabular input", hint='try {"instances": [{"x": 1}]}')


def payload_text(payload: dict[str, Any]) -> str:
    """Extract a single text input from the common shapes."""
    for key in ("text", "input", "inputs", "prompt", "query", "sentence", "q"):
        if key in payload:
            value = payload[key]
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    if len(payload) == 1:
        value = next(iter(payload.values()))
        if isinstance(value, str):
            return value
    raise AdapterError(
        "could not find a text input",
        hint='send {"text": "..."}',
    )


def payload_image_bytes(payload: dict[str, Any]) -> bytes:
    """Decode image bytes from a data URI, a base64 string, or a local file path.

    Order matters: ``validate=True`` makes strict base64 reject ordinary paths
    (``/`` and ``.`` are not alphabet characters), so a path is never silently
    mangled into random bytes. No network fetches - a URL must be downloaded by
    the caller, which keeps inference off the open internet.
    """
    import base64

    for key in ("image", "image_b64", "img", "data", "inputs"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue
        if "," in text and text.lower().startswith("data:"):
            text = text.split(",", 1)[1].strip()
        try:
            return base64.b64decode(text, validate=True)
        except Exception:  # noqa: BLE001 - not base64; maybe it is a path
            pass
        candidate = Path(text).expanduser()
        if candidate.is_file():
            return candidate.read_bytes()
    raise AdapterError(
        "could not find image bytes",
        hint='send {"image": "<base64>"}, {"image": "data:image/png;base64,..."} '
        'or {"image": "/content/photo.jpg"}',
    )
