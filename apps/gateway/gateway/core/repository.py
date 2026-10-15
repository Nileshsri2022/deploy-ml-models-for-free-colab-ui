"""Typed repositories over :class:`~gateway.core.store.JsonStore`.

Every domain object is stored as a JSON dict and rehydrated into its shared-schema
Pydantic model on read, so the API layer never hand-rolls ``dict.get`` chains.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from colabdeploy_shared.enums import DeploymentState, JobState, RuntimeState
from colabdeploy_shared.schemas import (
    ApiKeyInfo,
    DeploymentInfo,
    DesiredDeployment,
    EndpointInfo,
    JobInfo,
    ModelInfo,
    RuntimeInfo,
    utcnow,
)
from gateway.core.store import JsonStore

M = TypeVar("M", bound=BaseModel)

COLLECTIONS = (
    "models",
    "endpoints",
    "deployments",
    "runtimes",
    "api_keys",
    "jobs",
    "audit",
)


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


class Repository:
    """Base CRUD helper for one collection + one Pydantic type."""

    collection = "items"
    model: Type[BaseModel] = BaseModel

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def list(self) -> list[Any]:
        out = []
        for raw in self.store.all(self.collection):
            obj = self._parse(raw)
            if obj is not None:
                out.append(obj)
        return out

    def get(self, key: str) -> Any | None:
        raw = self.store.get(self.collection, key)
        return self._parse(raw) if raw else None

    def put(self, obj: BaseModel) -> BaseModel:
        data = _dump(obj)
        key = str(data.get("id"))
        self.store.put(self.collection, key, data)
        return obj

    def patch(self, key: str, **changes: Any) -> Any | None:
        raw = self.store.update(self.collection, key, changes)
        return self._parse(raw) if raw else None

    def delete(self, key: str) -> bool:
        return self.store.delete(self.collection, key)

    def find(self, **equals: Any) -> list[Any]:
        return [o for o in self.list() if all(getattr(o, k, None) == v for k, v in equals.items())]

    def _parse(self, raw: dict[str, Any] | None) -> Any | None:
        if raw is None:
            return None
        try:
            return self.model.model_validate(raw)
        except ValidationError:
            return None


class ModelRepository(Repository):
    collection = "models"
    model = ModelInfo

    def latest_version(self, name: str) -> int:
        versions = [m.version for m in self.list() if m.name == name]
        return max(versions, default=0)

    def by_name_version(self, name: str, version: int) -> ModelInfo | None:
        for m in self.list():
            if m.name == name and m.version == version:
                return m
        return None

    def versions_of(self, model_id: str) -> list[ModelInfo]:
        base = self.get(model_id)
        if base is None:
            return []
        return sorted(
            (m for m in self.list() if m.name == base.name), key=lambda m: m.version, reverse=True
        )


class EndpointRepository(Repository):
    collection = "endpoints"
    model = EndpointInfo

    def by_route(self, route: str) -> EndpointInfo | None:
        route = route.strip("/")
        for ep in self.list():
            if ep.route.strip("/") == route:
                return ep
        return None


class DeploymentRepository(Repository):
    collection = "deployments"
    model = DeploymentInfo

    def for_runtime(self, runtime_id: str) -> list[DeploymentInfo]:
        return [d for d in self.list() if d.runtime_id == runtime_id]

    def for_endpoint(self, endpoint_id: str) -> list[DeploymentInfo]:
        return sorted(
            (d for d in self.list() if d.endpoint_id == endpoint_id),
            key=lambda d: d.updated_at,
            reverse=True,
        )

    def active_for_endpoint(self, endpoint_id: str) -> DeploymentInfo | None:
        for state in (DeploymentState.READY, DeploymentState.LOADING, DeploymentState.PENDING):
            for dep in self.for_endpoint(endpoint_id):
                if dep.state == state:
                    return dep
        return None


class RuntimeRepository(Repository):
    collection = "runtimes"
    model = RuntimeInfo

    def online(self) -> list[RuntimeInfo]:
        return [r for r in self.list() if r.state in (RuntimeState.ONLINE, RuntimeState.DEGRADED)]

    def connected(self) -> list[RuntimeInfo]:
        return [r for r in self.list() if r.state == RuntimeState.ONLINE]


class ApiKeyRepository(Repository):
    collection = "api_keys"
    model = ApiKeyInfo

    def by_prefix(self, prefix: str) -> ApiKeyInfo | None:
        for key in self.list():
            if key.prefix == prefix:
                return key
        return None


class JobRepository(Repository):
    collection = "jobs"
    model = JobInfo

    def queued(self) -> list[JobInfo]:
        return sorted(
            (j for j in self.list() if j.state == JobState.QUEUED), key=lambda j: j.created_at
        )

    def prune(self, older_than: datetime) -> int:
        removed = 0
        for job in self.list():
            if job.done_at and job.done_at < older_than:
                self.delete(job.id)
                removed += 1
        return removed


class AuditLog:
    """Append-only, bounded audit trail of management actions."""

    collection = "audit"
    max_entries = 2000

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def record(self, action: str, actor: str = "system", **detail: Any) -> None:
        from colabdeploy_shared.utils import short_id

        entry = {
            "id": short_id("aud", 6),
            "ts": utcnow().isoformat(),
            "action": action,
            "actor": actor,
            "detail": detail,
        }
        self.store.put(self.collection, entry["id"], entry)
        self._trim()

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        entries = sorted(self.store.all(self.collection), key=lambda e: e.get("ts", ""), reverse=True)
        return entries[:limit]

    def _trim(self) -> None:
        entries = self.store.all(self.collection)
        if len(entries) <= self.max_entries:
            return
        for entry in sorted(entries, key=lambda e: e.get("ts", ""))[: len(entries) - self.max_entries]:
            self.store.delete(self.collection, str(entry.get("id")))


class Registry:
    """Facade holding every repository - one object to inject into services."""

    def __init__(self, store: JsonStore) -> None:
        self.store = store
        self.models = ModelRepository(store)
        self.endpoints = EndpointRepository(store)
        self.deployments = DeploymentRepository(store)
        self.runtimes = RuntimeRepository(store)
        self.api_keys = ApiKeyRepository(store)
        self.jobs = JobRepository(store)
        self.audit = AuditLog(store)

    @property
    def revision(self) -> int:
        return self.store.revision

    def prune_jobs(self, older_than: datetime) -> int:
        return self.jobs.prune(older_than)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def newest(items: Iterable[Any], key: str = "created_at") -> Any | None:
    return max(items, default=None, key=lambda i: getattr(i, key, datetime.min))
