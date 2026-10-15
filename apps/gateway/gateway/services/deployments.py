"""Model registration, endpoint publishing, and the deploy/rollback state machine.

The control plane never loads a model - it records *intent*. Deploying writes a
``DeploymentInfo`` in ``pending`` state; the agent picks it up from its desired
state, hydrates the artifact, warms the model and acks ``ready``. That split is
what lets a deploy survive a VM that is currently dead: the intent is durable, the
execution is retryable.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import (
    AdapterKind,
    DegradationMode,
    DeploymentState,
    EndpointVisibility,
    RuntimeState,
)
from colabdeploy_shared.schemas import (
    ArtifactSpec,
    enum_value,
    DeploymentInfo,
    EndpointCreate,
    EndpointInfo,
    ModelCreate,
    ModelInfo,
    ModelSchema,
    utcnow,
)
from colabdeploy_shared.utils import sha256_file, short_id
from gateway.core.config import Settings
from gateway.core.logging import Telemetry
from gateway.core.repository import Registry
from gateway.services.runtime_manager import RuntimeManager

# Extension -> adapter, ordered most specific first.
_EXTENSION_MAP: list[tuple[tuple[str, ...], AdapterKind]] = [
    ((".onnx",), AdapterKind.ONNX),
    ((".pt", ".pth", ".pt2", ".ckpt"), AdapterKind.PYTORCH),
    ((".h5", ".keras"), AdapterKind.KERAS),
    ((".xgb", ".ubj"), AdapterKind.XGBOOST),
    ((".pkl", ".pickle", ".joblib", ".job"), AdapterKind.SKLEARN),
]

_FRAMEWORK_MAP: dict[str, AdapterKind] = {
    "sklearn": AdapterKind.SKLEARN,
    "scikit-learn": AdapterKind.SKLEARN,
    "scikit": AdapterKind.SKLEARN,
    "xgboost": AdapterKind.XGBOOST,
    "lgbm": AdapterKind.SKLEARN,
    "lightgbm": AdapterKind.SKLEARN,
    "torch": AdapterKind.PYTORCH,
    "pytorch": AdapterKind.PYTORCH,
    "keras": AdapterKind.KERAS,
    "tensorflow": AdapterKind.KERAS,
    "tf": AdapterKind.KERAS,
    "onnx": AdapterKind.ONNX,
    "transformers": AdapterKind.HUGGINGFACE,
    "huggingface": AdapterKind.HUGGINGFACE,
    "hf": AdapterKind.HUGGINGFACE,
    "echo": AdapterKind.ECHO,
    "python": AdapterKind.PYTHON_FUNCTION,
    "python_function": AdapterKind.PYTHON_FUNCTION,
    "custom": AdapterKind.PYTHON_FUNCTION,
}

_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, fallback: str = "model") -> str:
    slug = _SLUG_SAFE.sub("-", (name or "").lower()).strip("-")
    return slug[:48] or fallback


def detect_adapter(create: ModelCreate) -> tuple[AdapterKind, str]:
    """Pick an adapter without loading anything.

    Returns ``(adapter, reason)``. ``AUTO`` means "let the agent try adapters in
    ranked order" - useful for odd artifacts, and always available as a fallback.
    """
    if create.adapter and create.adapter != AdapterKind.AUTO:
        return create.adapter, "explicit adapter"

    framework = (create.framework or "").strip().lower()
    if framework and framework != "auto":
        mapped = _FRAMEWORK_MAP.get(framework)
        if mapped is not None:
            return mapped, f"framework hint '{framework}'"

    uri = create.artifact.uri.lower()
    if uri.startswith("hf://"):
        return AdapterKind.HUGGINGFACE, "huggingface hub uri"
    if create.entrypoint:
        return AdapterKind.PYTHON_FUNCTION, "custom entrypoint supplied"

    filename = (create.artifact.filename or uri.rsplit("/", 1)[-1]).lower()
    for extensions, adapter in _EXTENSION_MAP:
        if filename.endswith(extensions):
            return adapter, f"file extension '{Path(filename).suffix}'"

    # Common SavedModel / directory layouts.
    if "savedmodel" in uri or uri.endswith(".pb"):
        return AdapterKind.KERAS, "tensorflow SavedModel layout"
    if "safetensors" in uri or "pytorch_model.bin" in uri:
        return AdapterKind.HUGGINGFACE, "hf-style weights"
    return AdapterKind.AUTO, "no confident signal - agent will probe adapters"


class DeploymentService:
    def __init__(
        self,
        registry: Registry,
        settings: Settings,
        telemetry: Telemetry,
        runtimes: RuntimeManager,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.telemetry = telemetry
        self.runtimes = runtimes

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #
    def store_artifact(self, filename: str, data: bytes) -> ArtifactSpec:
        """Persist an uploaded file in the control plane's artifact store.

        The agent later fetches it via ``store://<id>`` which the gateway resolves
        to ``GET /internal/artifacts/{id}`` (runtime-token auth).
        """
        assert self.settings.artifact_dir is not None
        digest = hashlib.sha256(data).hexdigest()
        artifact_id = f"art_{digest[:16]}"
        target = self.settings.artifact_dir / artifact_id
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(filename))[:80] or "model.bin"
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        return ArtifactSpec(
            uri=f"store://{artifact_id}",
            checksum=digest,
            size_bytes=len(data),
            filename=safe_name,
        )

    def artifact_path(self, uri: str) -> Path | None:
        if not uri.startswith("store://"):
            return None
        assert self.settings.artifact_dir is not None
        path = self.settings.artifact_dir / uri.split("://", 1)[1]
        return path if path.exists() else None

    # ------------------------------------------------------------------ #
    # Models
    # ------------------------------------------------------------------ #
    def register_model(self, create: ModelCreate, owner: str = "default") -> ModelInfo:
        adapter, reason = detect_adapter(create)
        version = self.registry.models.latest_version(create.name) + 1
        model = ModelInfo(
            id=short_id("mdl", 8),
            name=create.name,
            version=version,
            framework=create.framework or adapter.value,
            adapter=adapter,
            artifact=create.artifact,
            schema_in=create.schema_in or ModelSchema(),
            schema_out=create.schema_out or ModelSchema(),
            warmup_sample=create.warmup_sample or create.schema_in.example,
            entrypoint=create.entrypoint,
            tags=create.tags,
            status="registered",
            owner_id=owner,
            notes=f"adapter chosen: {reason}",
        )
        if adapter == AdapterKind.PYTHON_FUNCTION and not create.entrypoint:
            raise ValueError("adapter 'python_function' requires an entrypoint (source or callable)")
        self.registry.models.put(model)
        self.registry.audit.record(
            "model.register", actor=owner, model_id=model.id, name=model.name, version=version
        )
        self.telemetry.log(
            f"registered model {model.name} v{version} ({adapter.value}: {reason})",
            data={"model_id": model.id},
        )
        return model

    def model_by_name_version(self, name: str, version: int | None = None) -> ModelInfo | None:
        if version is None:
            versions = [m for m in self.registry.models.list() if m.name == name]
            return max(versions, default=None, key=lambda m: m.version)
        return self.registry.models.by_name_version(name, version)

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #
    def create_endpoint(self, spec: EndpointCreate, *, model_version: int | None = None) -> EndpointInfo:
        if not spec.model_id:
            raise ValueError("endpoint needs a model_id")
        model = self.registry.models.get(spec.model_id)
        if model is None:
            raise ValueError(f"model '{spec.model_id}' not found")
        if model_version is not None:
            pinned = self.registry.models.by_name_version(model.name, model_version)
            if pinned is not None:
                model = pinned
        route = slugify(spec.route or model.name)
        existing = self.registry.endpoints.by_route(route)
        if existing is not None and existing.model_id != spec.model_id:
            route = f"{route}-{model.version}"
            if self.registry.endpoints.by_route(route) is not None:
                route = f"{route}-{short_id('ep', 3)[3:]}"
        endpoint = EndpointInfo(
            id=short_id("ep", 8),
            model_id=model.id,
            model_version=model.version,
            route=route,
            visibility=EndpointVisibility(spec.visibility),
            cache_ttl_s=spec.cache_ttl_s,
            stale_while_revalidate_s=spec.stale_while_revalidate_s,
            degradation_mode=DegradationMode(spec.degradation_mode),
            timeout_ms=spec.timeout_ms,
            rate_limit_rpm=spec.rate_limit_rpm,
            max_body_bytes=spec.max_body_bytes,
            status="active",
            model_name=model.name,
            adapter=model.adapter,
            schema_in=model.schema_in,
            schema_out=model.schema_out,
        )
        self.registry.endpoints.put(endpoint)
        self.registry.audit.record("endpoint.create", endpoint_id=endpoint.id, route=route)
        self.telemetry.log(f"published endpoint /v1/models/{route}/predict")
        return endpoint

    def update_endpoint(self, endpoint_id: str, **changes: Any) -> EndpointInfo | None:
        clean = {k: v for k, v in changes.items() if v is not None}
        if not clean:
            return self.registry.endpoints.get(endpoint_id)
        if "route" in clean:
            clean["route"] = slugify(str(clean["route"]))
        updated = self.registry.store.update("endpoints", endpoint_id, clean)
        if updated is None:
            return None
        endpoint = EndpointInfo.model_validate(updated)
        # Keep the denormalised schema copy in sync when the model version moves.
        if "model_version" in clean or "model_id" in clean:
            model = self._resolve_version(endpoint)
            if model is not None:
                endpoint.schema_in = model.schema_in
                endpoint.schema_out = model.schema_out
                endpoint.model_name = model.name
                endpoint.adapter = model.adapter
                self.registry.endpoints.put(endpoint)
        self.registry.audit.record("endpoint.update", endpoint_id=endpoint_id, **clean)
        return endpoint

    def delete_endpoint(self, endpoint_id: str) -> bool:
        removed = 0
        for dep in self.registry.deployments.for_endpoint(endpoint_id):
            self.registry.store.update(
                "deployments", dep.id, {"state": DeploymentState.UNLOADED.value}
            )
            removed += 1
        self.cache_invalidate(endpoint_id)
        ok = self.registry.endpoints.delete(endpoint_id)
        if ok:
            self.registry.audit.record("endpoint.delete", endpoint_id=endpoint_id, deployments=removed)
        return ok

    def cache_invalidate(self, endpoint_id: str) -> int:
        from gateway.state import get_cache  # local import avoids a cycle at module load

        return get_cache().invalidate_endpoint(endpoint_id)

    def _resolve_version(self, endpoint: EndpointInfo) -> ModelInfo | None:
        base = self.registry.models.get(endpoint.model_id)
        if base is None:
            return None
        if endpoint.model_version == base.version:
            return base
        for candidate in self.registry.models.versions_of(base.id):
            if candidate.version == endpoint.model_version:
                return candidate
        return base

    # ------------------------------------------------------------------ #
    # Deploy
    # ------------------------------------------------------------------ #
    def deploy(
        self,
        model_id: str,
        *,
        model_version: int | None = None,
        endpoint_id: str | None = None,
        runtime_id: str | None = None,
        endpoint_spec: EndpointCreate | None = None,
    ) -> dict[str, Any]:
        """Register intent: pin a model version to an endpoint on a runtime."""
        base = self.registry.models.get(model_id)
        if base is None and model_version is None:
            # Allow deploying by name for convenience.
            base = self.model_by_name_version(model_id)
        if base is None:
            raise ValueError(f"model '{model_id}' not found")
        if model_version is not None and model_version != base.version:
            pinned = self.registry.models.by_name_version(base.name, model_version)
            if pinned is None:
                raise ValueError(f"model '{base.name}' has no version {model_version}")
            base = pinned

        endpoint = self.registry.endpoints.get(endpoint_id) if endpoint_id else None
        if endpoint is None:
            spec = endpoint_spec.model_copy(deep=True) if endpoint_spec else EndpointCreate()
            spec.model_id = base.id  # the deploy call is authoritative
            endpoint = self.create_endpoint(spec, model_version=base.version)
            created_endpoint = True
        else:
            created_endpoint = False
            self.registry.store.update(
                "endpoints", endpoint.id, {"model_id": base.id, "model_version": base.version}
            )
            endpoint = self.registry.endpoints.get(endpoint.id) or endpoint

        # Supersede any previous deployment for this endpoint.
        for dep in self.registry.deployments.for_endpoint(endpoint.id):
            if dep.state in (DeploymentState.PENDING, DeploymentState.LOADING, DeploymentState.READY):
                self.registry.store.update(
                    "deployments",
                    dep.id,
                    {"state": DeploymentState.UNLOADED.value, "updated_at": utcnow().isoformat()},
                )

        if runtime_id and not self.registry.runtimes.get(runtime_id):
            raise ValueError(f"runtime '{runtime_id}' not found")
        if runtime_id is None:
            online = self.registry.runtimes.online()
            runtime_id = online[0].id if online else None

        deployment = DeploymentInfo(
            id=short_id("dep", 8),
            endpoint_id=endpoint.id,
            model_id=base.id,
            model_version=base.version,
            runtime_id=runtime_id,
            state=DeploymentState.PENDING,
        )
        self.registry.deployments.put(deployment)
        self.registry.audit.record(
            "deploy", model_id=base.id, version=base.version, endpoint_id=endpoint.id,
            runtime_id=runtime_id,
        )
        self.telemetry.log(
            f"deploy {base.name} v{base.version} -> /v1/models/{endpoint.route}/predict "
            f"(runtime {runtime_id or 'any'})"
        )
        return {
            "deployment": deployment,
            "endpoint": endpoint,
            "model": base,
            "runtime_id": runtime_id,
            "endpoint_created": created_endpoint,
            "runtime_online": bool(runtime_id and self.runtimes.relay.is_connected(runtime_id)),
        }

    def rollback(self, endpoint_id: str, model_version: int) -> dict[str, Any]:
        """Repoint an endpoint at an older version - instant, artifacts stay put."""
        endpoint = self.registry.endpoints.get(endpoint_id)
        if endpoint is None:
            raise ValueError(f"endpoint '{endpoint_id}' not found")
        target = self.registry.models.by_name_version(
            (self.registry.models.get(endpoint.model_id) or ModelInfo(
                id=endpoint.model_id, name="", artifact=ArtifactSpec(uri="")
            )).name,
            model_version,
        )
        if target is None:
            raise ValueError(f"version {model_version} not found for that model")
        return self.deploy(target.id, model_version=model_version, endpoint_id=endpoint_id)

    # ------------------------------------------------------------------ #
    # Read models for the API/UI
    # ------------------------------------------------------------------ #
    def endpoint_view(self, endpoint: EndpointInfo, base_url: str = "") -> dict[str, Any]:
        model = self._resolve_version(endpoint)
        dep = self.registry.deployments.active_for_endpoint(endpoint.id)
        runtime = self.registry.runtimes.get(dep.runtime_id) if dep and dep.runtime_id else None
        path = f"/v1/models/{endpoint.route.strip('/')}/predict"
        return {
            **endpoint.model_dump(mode="json"),
            "model_name": model.name if model else endpoint.model_name,
            "adapter": (model.adapter if model else endpoint.adapter) or endpoint.adapter,
            "url": (base_url.rstrip("/") + path) if base_url else path,
            "path": path,
            "serving_runtime_id": dep.runtime_id if dep else None,
            "deployment_state": enum_value(dep.state) if dep else None,
            "runtime_state": enum_value(runtime.state) if runtime else None,
            "live": bool(runtime and runtime.state == RuntimeState.ONLINE),
            "schema_in": (model.schema_in if model else endpoint.schema_in).model_dump(mode="json"),
            "schema_out": (model.schema_out if model else endpoint.schema_out).model_dump(mode="json"),
            "curl": curl_snippet(base_url, endpoint.route, model.schema_in if model else None),
        }

    def model_view(self, model: ModelInfo) -> dict[str, Any]:
        versions = [m.version for m in self.registry.models.versions_of(model.id)]
        endpoints = [e for e in self.registry.endpoints.list() if e.model_id == model.id]
        return {
            **model.model_dump(mode="json"),
            "all_versions": sorted(versions, reverse=True),
            "endpoints": [e.id for e in endpoints],
        }


def curl_snippet(base_url: str, route: str, schema_in: ModelSchema | None) -> str:
    base = (base_url or "http://localhost:8000").rstrip("/")
    example = (schema_in.example if schema_in and schema_in.example else None) or {"inputs": {}}
    body = {"inputs": example}
    import json

    pretty = json.dumps(body, indent=2, default=str)
    return (
        f"curl -X POST '{base}/v1/models/{route.strip('/')}/predict' \\\n"
        "  -H 'Authorization: Bearer $CDP_API_KEY' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{json.dumps(body, default=str)}'\n\n"
        f"# formatted body:\n{pretty}"
    )


def local_artifact_spec(path: str | Path) -> ArtifactSpec:
    """Helper for CLI/demo use: reference a file already on the VM."""
    p = Path(path)
    return ArtifactSpec(
        uri=f"file://{p.resolve()}" if p.is_absolute() else str(p),
        checksum=sha256_file(str(p)) if p.exists() else None,
        size_bytes=p.stat().st_size if p.exists() else None,
        filename=p.name,
    )


def now_ms() -> int:
    return int(time.time() * 1000)
