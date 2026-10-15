"""The in-VM model pool and the desired-state reconciler.

The control plane says *what* should be loaded; this module decides *how*:

* resolve the artifact (cache-first, see :mod:`colabdeploy.artifacts`),
* pick/build an adapter (auto-detect with probing, see ``adapters``),
* warm it up with the deployer's sample (or a hot cache key),
* keep an LRU pool so a runtime serving 12 models on a 15 GB T4 does not OOM,
* unload what is no longer wanted,
* report a per-deployment state the gateway can show in the UI.

Every failure is captured per-deployment: one bad model must never stop the other
eleven from serving. That is why ``reconcile`` returns a list of results instead of
raising.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from colabdeploy_shared.enums import AdapterKind, DeploymentState
from colabdeploy_shared.schemas import ArtifactSpec, DesiredState

from .adapters import AdapterError, ModelAdapter, ModelMeta, available_adapters, load_with_probe, resolve_adapter
from .artifacts import ArtifactCache, ArtifactError, ArtifactResolver, ResolvedArtifact

LogFn = Callable[[str, str, dict[str, Any]], None]


@dataclass
class LoadedModel:
    """One warm model living in this VM."""

    key: str  # "{model_id}@{version}"
    meta: ModelMeta
    adapter: ModelAdapter
    artifact: ResolvedArtifact
    loaded_at: float = field(default_factory=time.time)
    warmup_ms: float = 0.0
    predictions: int = 0
    errors: int = 0
    last_used: float = field(default_factory=time.time)
    last_error: str | None = None
    deployment_id: str = ""
    endpoint_id: str = ""

    def touch(self) -> None:
        self.last_used = time.time()

    def status(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "model_id": self.meta.model_id,
            "model_version": self.meta.model_version,
            "adapter": self.adapter.kind_value,
            "deployment_id": self.deployment_id,
            "endpoint_id": self.endpoint_id,
            "state": DeploymentState.READY.value,
            "warm": self.warmup_ms > 0,
            "warmup_ms": self.warmup_ms,
            "predictions": self.predictions,
            "errors": self.errors,
            "last_error": self.last_error,
            "loaded_s": round(time.time() - self.loaded_at, 1),
            "from_cache": self.artifact.from_cache,
            "device": getattr(self.adapter, "device", None),
            "detail": self.adapter.health.detail,
        }


@dataclass
class ReconcileResult:
    deployment_id: str
    model_id: str
    model_version: int
    state: str
    warm: bool = False
    warmup_ms: float | None = None
    error: str | None = None
    adapter: str | None = None
    elapsed_ms: float = 0.0
    from_cache: bool = False


class ModelStore:
    """LRU pool of loaded adapters + the reconcile loop."""

    def __init__(
        self,
        resolver: ArtifactResolver,
        cache: ArtifactCache,
        *,
        max_loaded: int = 4,
        log: LogFn | None = None,
    ) -> None:
        self.resolver = resolver
        self.cache = cache
        self.max_loaded = max(1, max_loaded)
        self._log = log or (lambda message, level="info", **kw: None)
        self._pool: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()
        self._failed: dict[str, str] = {}
        self._last_desired_revision = 0
        self.adapters_available = available_adapters()
        #: warm-up runs are observable separately from served predictions because
        #: they bypass ``/execute`` (a warm model must not look like traffic).
        self.counters = {"warmups": 0, "warmup_failures": 0, "evictions": 0, "cache_hits": 0}

    # ------------------------------------------------------------------ #
    @property
    def keys(self) -> list[str]:
        with self._lock:
            return list(self._pool)

    def get(self, model_id: str, version: int) -> LoadedModel | None:
        key = f"{model_id}@{version}"
        with self._lock:
            loaded = self._pool.get(key)
            if loaded is not None:
                self._pool.move_to_end(key)
                loaded.touch()
            return loaded

    def find_by_endpoint(self, endpoint_id: str) -> LoadedModel | None:
        with self._lock:
            for loaded in reversed(list(self._pool.values())):
                if loaded.endpoint_id == endpoint_id:
                    loaded.touch()
                    return loaded
        return None

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [loaded.status() for loaded in self._pool.values()]

    def failures(self) -> dict[str, str]:
        with self._lock:
            return dict(self._failed)

    # ------------------------------------------------------------------ #
    def load_deployment(self, dep: dict[str, Any], *, warmup: bool = True) -> ReconcileResult:
        """Resolve → detect → load → warm. Never raises; failures become results."""
        started = time.perf_counter()
        meta = ModelMeta.from_deployment(dep)
        key = meta.key
        result = ReconcileResult(
            deployment_id=meta.deployment_id,
            model_id=meta.model_id,
            model_version=meta.model_version,
            state=DeploymentState.LOADING.value,
            adapter=meta.adapter_kind,
        )

        with self._lock:
            existing = self._pool.get(key)
        if existing is not None:
            existing.deployment_id = meta.deployment_id or existing.deployment_id
            existing.endpoint_id = meta.endpoint_id or existing.endpoint_id
            result.state = DeploymentState.READY.value
            result.warm = existing.warmup_ms > 0
            result.warmup_ms = existing.warmup_ms
            result.from_cache = existing.artifact.from_cache
            result.elapsed_ms = _ms(started)
            result.adapter = existing.adapter.kind_value
            self._failed.pop(meta.deployment_id, None)
            return result

        # 1) artifact
        try:
            artifact = self.resolver.resolve(meta.artifact)
        except (ArtifactError, OSError) as exc:
            return self._fail(result, meta, f"artifact: {exc}", started)
        result.from_cache = artifact.from_cache
        if artifact.from_cache:
            self.counters["cache_hits"] += 1

        # 2) adapter + load (probe when auto)
        adapter: ModelAdapter
        try:
            kind = meta.adapter_kind
            if kind and kind != AdapterKind.AUTO.value:
                adapter = resolve_adapter(meta.artifact, meta)
                adapter.load(artifact.path, meta)
            else:
                def on_attempt(name: str, ok: bool, error: str | None) -> None:
                    self._log(
                        f"probe {name}: {'loaded' if ok else 'failed'}" + (f" - {error}" if error else ""),
                        "debug" if ok else "warning",
                        model_id=meta.model_id,
                    )

                adapter, errors = load_with_probe(artifact.path, meta, on_attempt=on_attempt)
                if errors:
                    self._log(f"auto-detect notes for {key}: {'; '.join(errors[:3])}", "debug")
        except AdapterError as exc:
            return self._fail(result, meta, f"load: {exc}", started, artifact=artifact)
        except Exception as exc:  # noqa: BLE001 - unexpected framework blowup
            return self._fail(result, meta, f"load: {type(exc).__name__}: {exc}", started, artifact=artifact)

        # 3) warm up
        warmup_ms = 0.0
        sample = meta.warmup_sample
        if warmup and sample:
            try:
                warmup_ms = adapter.warmup(sample)
                self.counters["warmups"] += 1
            except Exception as exc:  # noqa: BLE001
                self.counters["warmup_failures"] += 1
                self._log(f"warm-up failed for {key}: {exc}", "warning")
        adapter.health.warmup_ms = warmup_ms

        loaded = LoadedModel(
            key=key,
            meta=meta,
            adapter=adapter,
            artifact=artifact,
            warmup_ms=warmup_ms,
            deployment_id=meta.deployment_id,
            endpoint_id=meta.endpoint_id,
        )
        with self._lock:
            self._pool[key] = loaded
            self._pool.move_to_end(key)
            self._failed.pop(meta.deployment_id, None)
            self._evict_locked()

        result.state = DeploymentState.READY.value
        result.warm = warmup_ms > 0
        result.warmup_ms = warmup_ms
        result.adapter = adapter.kind_value
        result.elapsed_ms = _ms(started)
        self._log(
            f"loaded {key} with {result.adapter}"
            + (f" (artifact from cache)" if artifact.from_cache else "")
            + f" in {result.elapsed_ms:.0f} ms"
            + (f", warm-up {warmup_ms:.0f} ms" if warmup_ms else ""),
            "info",
            model_id=meta.model_id,
        )
        return result

    def _fail(
        self,
        result: ReconcileResult,
        meta: ModelMeta,
        message: str,
        started: float,
        artifact: ResolvedArtifact | None = None,
    ) -> ReconcileResult:
        result.state = DeploymentState.FAILED.value
        result.error = message[:2000]
        result.elapsed_ms = _ms(started)
        result.from_cache = bool(artifact and artifact.from_cache)
        with self._lock:
            self._failed[meta.deployment_id or meta.key] = result.error
        self._log(f"FAILED to load {meta.key}: {message}", "error", model_id=meta.model_id)
        return result

    def _evict_locked(self) -> None:
        """Drop least-recently-used models once the pool exceeds ``max_loaded``."""
        while len(self._pool) > self.max_loaded:
            key, loaded = next(iter(self._pool.items()))
            self._pool.pop(key, None)
            self.counters["evictions"] += 1
            try:
                loaded.adapter.unload()
            except Exception:  # noqa: BLE001 - eviction must not raise
                pass
            self._log(f"evicted {key} from the warm pool (max_loaded={self.max_loaded})", "info")

    def unload(self, key: str) -> bool:
        with self._lock:
            loaded = self._pool.pop(key, None)
        if loaded is None:
            return False
        try:
            loaded.adapter.unload()
        except Exception:  # noqa: BLE001
            pass
        self._log(f"unloaded {key}", "info")
        return True

    def unload_all(self) -> int:
        with self._lock:
            keys = list(self._pool)
        return sum(1 for key in keys if self.unload(key))

    # ------------------------------------------------------------------ #
    def reconcile(
        self, desired: DesiredState | dict[str, Any], *, warmup_samples: list[dict[str, Any]] | None = None
    ) -> list[ReconcileResult]:
        """Converge towards the control plane's desired state."""
        if isinstance(desired, dict):
            desired = DesiredState.model_validate(desired)
        results: list[ReconcileResult] = []
        wanted_keys: set[str] = set()
        for dep in desired.deployments:
            payload = dep.model_dump(mode="json") if hasattr(dep, "model_dump") else dict(dep)
            key = f"{payload['model_id']}@{payload['model_version']}"
            wanted_keys.add(key)
            results.append(self.load_deployment(payload))
            self._last_desired_revision = desired.revision

        # Unload anything no longer wanted (frees VRAM for the new version).
        with self._lock:
            stale = [key for key in self._pool if key not in wanted_keys]
        for key in stale:
            self.unload(key)

        # Warm with hot cache keys shipped by the gateway on reconnect.
        for sample in warmup_samples or []:
            self.warm_from_sample(sample)

        self.cache.prune()
        return results

    def warm_from_sample(self, sample: dict[str, Any]) -> None:
        """Prime a model using a real cached request (see architecture.md §8)."""
        endpoint_id = str(sample.get("endpoint_id", ""))
        payload = sample.get("payload") or sample.get("inputs") or {}
        if not isinstance(payload, dict):
            return
        loaded = self.find_by_endpoint(endpoint_id) if endpoint_id else None
        if loaded is None:
            return
        try:
            started = time.perf_counter()
            loaded.adapter.predict(payload)
            elapsed = (time.perf_counter() - started) * 1000
            loaded.warmup_ms = max(loaded.warmup_ms, round(elapsed, 2))
            loaded.touch()
            self.counters["warmups"] += 1
            self._log(f"warm-up from hot cache key for {loaded.key} took {elapsed:.0f} ms", "debug")
        except Exception as exc:  # noqa: BLE001
            self.counters["warmup_failures"] += 1
            self._log(f"warm-up from cache key failed for {loaded.key}: {exc}", "warning")

    # ------------------------------------------------------------------ #
    def predict(self, model_id: str, version: int, payload: dict[str, Any]) -> Any:
        loaded = self.get(model_id, version)
        if loaded is None:
            raise AdapterError(
                f"model {model_id}@{version} is not loaded in this runtime",
                hint=f"loaded: {', '.join(self.keys) or 'none'}",
            )
        started = time.perf_counter()
        try:
            output = loaded.adapter.predict(payload)
        except Exception as exc:  # noqa: BLE001
            loaded.errors += 1
            loaded.adapter.record_prediction(False, str(exc))
            loaded.last_error = str(exc)[:500]
            message = exc.message if isinstance(exc, AdapterError) else f"{type(exc).__name__}: {exc}"
            hint = getattr(exc, "hint", None)
            raise AdapterError(message + (f" | hint: {hint}" if hint else "")) from exc
        loaded.predictions += 1
        loaded.adapter.record_prediction(True)
        loaded.touch()
        return {
            "output": output,
            "model_ms": round((time.perf_counter() - started) * 1000, 3),
            "adapter": loaded.adapter.kind_value,
            "device": getattr(loaded.adapter, "device", None),
            "warm": loaded.warmup_ms > 0,
        }

    def predict_for_endpoint(self, endpoint_id: str, payload: dict[str, Any]) -> Any:
        loaded = self.find_by_endpoint(endpoint_id)
        if loaded is None:
            raise AdapterError(
                f"no model loaded for endpoint '{endpoint_id}'",
                hint=f"loaded: {', '.join(self.keys) or 'none'}",
            )
        return self.predict(loaded.meta.model_id, loaded.meta.model_version, payload)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def build_artifact_spec(uri: str, checksum: str | None = None) -> ArtifactSpec:
    """Convenience for tests/CLI."""
    return ArtifactSpec(uri=uri, checksum=checksum, filename=Path(uri).name)
