"""Adapter registry: ``adapter: "auto"`` resolution.

The registry answers one question: given an artifact and a hint, which adapter
should load it? Two-stage, cheapest first:

1. **Detect** - ask every registered adapter's ``detect()`` (name/URI inspection
   only, no imports of heavy frameworks) and take the highest-priority match.
2. **Probe** - if nothing is confident, try ``load()`` in priority order and keep
   the first that succeeds *and* survives a warm-up call. Failures are collected
   and reported together, so the UI shows "tried sklearn: …; tried onnx: …"
   instead of a mystery error.

Heavy adapters import their framework lazily inside ``load()``, so registering all
of them costs nothing on a VM that only has scikit-learn.
"""

from __future__ import annotations

from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec

from .base import AdapterError, Health, ModelAdapter, ModelMeta, jsonable, payload_table, payload_text
from .echo import EchoAdapter
from .python_function import PythonFunctionAdapter

__all__ = [
    "AdapterError",
    "EchoAdapter",
    "Health",
    "ModelAdapter",
    "ModelMeta",
    "PythonFunctionAdapter",
    "adapter_kinds",
    "available_adapters",
    "build_adapter",
    "framework_available",
    "get_adapter_class",
    "jsonable",
    "payload_table",
    "payload_text",
    "resolve_adapter",
]

# Lazy table: adapter kind -> "module:ClassName". Importing on demand keeps the
# agent's startup under a second even with every framework installed.
_LAZY: dict[str, tuple[str, str]] = {
    AdapterKind.SKLEARN.value: ("colabdeploy.adapters.sklearn", "SklearnAdapter"),
    AdapterKind.XGBOOST.value: ("colabdeploy.adapters.xgboost", "XGBoostAdapter"),
    AdapterKind.ONNX.value: ("colabdeploy.adapters.onnx", "OnnxAdapter"),
    AdapterKind.PYTORCH.value: ("colabdeploy.adapters.pytorch", "PyTorchAdapter"),
    AdapterKind.KERAS.value: ("colabdeploy.adapters.keras", "KerasAdapter"),
    AdapterKind.HUGGINGFACE.value: ("colabdeploy.adapters.huggingface", "HuggingFaceAdapter"),
}

_EAGER: dict[str, type[ModelAdapter]] = {
    AdapterKind.ECHO.value: EchoAdapter,
    AdapterKind.PYTHON_FUNCTION.value: PythonFunctionAdapter,
}

#: Probe order for ``adapter: "auto"`` - cheap/unambiguous first, heavy last.
_AUTO_ORDER: tuple[str, ...] = (
    AdapterKind.PYTHON_FUNCTION.value,
    AdapterKind.ECHO.value,
    AdapterKind.ONNX.value,
    AdapterKind.XGBOOST.value,
    AdapterKind.SKLEARN.value,
    AdapterKind.PYTORCH.value,
    AdapterKind.KERAS.value,
    AdapterKind.HUGGINGFACE.value,
)


def get_adapter_class(kind: str) -> type[ModelAdapter] | None:
    """Resolve an adapter class by kind, importing lazily. ``None`` if unknown."""
    key = getattr(kind, "value", str(kind))
    if key in _EAGER:
        return _EAGER[key]
    target = _LAZY.get(key)
    if target is None:
        return None
    module_name, class_name = target
    try:
        import importlib

        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return getattr(module, class_name, None)


def build_adapter(kind: str) -> ModelAdapter:
    """Instantiate an adapter, raising a helpful error when unavailable."""
    cls = get_adapter_class(kind)
    if cls is None:
        raise AdapterError(
            f"adapter '{kind}' is not available in this runtime",
            hint=f"installed adapters: {', '.join(available_adapters())}",
        )
    return cls()


def adapter_kinds() -> list[str]:
    return [k.value for k in AdapterKind if k != AdapterKind.AUTO]


def framework_available(cls: type[ModelAdapter]) -> bool:
    """True when every framework module this adapter needs can be imported.

    ``"a|b"`` means either satisfies the requirement (keras ships standalone *or*
    inside tensorflow). Cached per module name - importing torch twice is cheap
    but walking ``sys.modules`` is cheaper.
    """
    import importlib
    import sys

    for requirement in getattr(cls, "framework", ()) or ():
        alternatives = [name.strip() for name in requirement.split("|") if name.strip()]
        ok = False
        for name in alternatives:
            if name in sys.modules:
                ok = True
                break
            try:
                importlib.import_module(name)
                ok = True
                break
            except Exception:  # noqa: BLE001 - missing optional dependency
                continue
        if not ok:
            return False
    return True


def available_adapters() -> list[str]:
    """Adapters that can actually serve a model **right now** (reported to the UI).

    Honest capability reporting matters: the gateway uses it to explain why a
    deploy failed ("torch is not installed in this runtime") instead of guessing.
    """
    ready: list[str] = []
    for kind in (*_EAGER, *_LAZY):
        cls = get_adapter_class(kind)
        if cls is None:
            continue
        try:
            if framework_available(cls):
                ready.append(kind)
        except Exception:  # noqa: BLE001
            continue
    return sorted(set(ready))


def resolve_adapter(artifact: ArtifactSpec, meta: ModelMeta) -> ModelAdapter:
    """Stage 1: pick an adapter class from the kind or from cheap detection."""
    kind = meta.adapter_kind
    if kind and kind != AdapterKind.AUTO.value:
        return build_adapter(kind)
    candidates: list[type[ModelAdapter]] = []
    for name in _AUTO_ORDER:
        cls = get_adapter_class(name)
        if cls is None:
            continue
        try:
            if cls.detect(artifact, meta):
                candidates.append(cls)
        except Exception:  # noqa: BLE001 - a broken detect() must not block others
            continue
    if candidates:
        candidates.sort(key=lambda c: getattr(c, "priority", 100))
        return candidates[0]()
    # Nothing confident: hand back the cheapest general adapter so the probe stage
    # in :func:`load_with_probe` can walk the list.
    return build_adapter(AdapterKind.SKLEARN.value)


def probe_order(meta: ModelMeta) -> list[str]:
    """Adapter kinds to try, in order, when auto-detection must load to decide."""
    kind = meta.adapter_kind
    if kind and kind != AdapterKind.AUTO.value:
        return [kind]
    order = list(_AUTO_ORDER)
    if meta.entrypoint:
        order = [AdapterKind.PYTHON_FUNCTION.value, *[k for k in order if k != AdapterKind.PYTHON_FUNCTION.value]]
    usable = []
    for kind in order:
        cls = get_adapter_class(kind)
        if cls is not None and framework_available(cls):
            usable.append(kind)
    return usable


def load_with_probe(
    artifact_path: Any, meta: ModelMeta, on_attempt: Any | None = None
) -> tuple[ModelAdapter, list[str]]:
    """Stage 2: try adapters in order until one loads. Returns ``(adapter, errors)``.

    ``on_attempt(kind, ok, error)`` lets the caller stream progress into the UI,
    which turns a slow first deploy into something legible instead of a spinner.
    """
    errors: list[str] = []
    order = probe_order(meta)
    for kind in order:
        cls = get_adapter_class(kind)
        if cls is None:
            errors.append(f"{kind}: adapter not installed")
            continue
        adapter = cls()
        try:
            adapter.load(artifact_path, meta)
        except AdapterError as exc:
            errors.append(f"{kind}: {exc}")
            if on_attempt:
                on_attempt(kind, False, str(exc))
            continue
        except Exception as exc:  # noqa: BLE001 - unexpected load failure
            errors.append(f"{kind}: {type(exc).__name__}: {exc}")
            if on_attempt:
                on_attempt(kind, False, str(exc))
            continue
        if on_attempt:
            on_attempt(kind, True, None)
        return adapter, errors
    raise AdapterError(
        "no adapter could load this artifact",
        hint="; ".join(errors[:6]) or f"tried {', '.join(order)}",
    )
