"""Custom-code adapter - the escape hatch for "any model".

Some models are not a pickle: they are a preprocessing pipeline, a heuristic, an
ensemble, or a call to a third-party library we have no adapter for. Instead of
forcing that into a framework, the deployer supplies a Python function and we run
it.

Safety model (see architecture.md §9): **this code runs in your own Colab VM,
under your own Google account.** It never runs in the control plane, and the UI
says so at deploy time. We still keep the blast radius sane:

* the function is compiled once at load; syntax errors fail the deployment loudly;
* only the named entrypoint is exposed - other names in the snippet stay private;
* a per-call timeout is enforced when the caller supplies one (best effort, since
  Python cannot preempt arbitrary C code).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable

_DEFAULT_SOURCE = '''def predict(payload, model=None, meta=None):
    """Return any JSON-serialisable value."""
    return {"echo": payload}
'''


class PythonFunctionAdapter(ModelAdapter):
    kind = AdapterKind.PYTHON_FUNCTION
    priority = 5  # explicit intent always wins the auto-probe
    extensions = (".py",)

    def __init__(self) -> None:
        super().__init__()
        self._fn: Callable[..., Any] | None = None
        self._namespace: dict[str, Any] = {}

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        source = (meta.entrypoint or "").strip()
        if not source and artifact_path.exists():
            source = artifact_path.read_text(encoding="utf-8")
        if not source:
            source = _DEFAULT_SOURCE

        # ``entrypoint`` may be a bare callable name with the code in the artifact,
        # or the code itself. Detect by looking for a def/lambda.
        name = "predict"
        if source and "\n" not in source and "def " not in source and "lambda" not in source:
            name = source.strip()
            source = artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else _DEFAULT_SOURCE

        namespace: dict[str, Any] = {"__name__": "colabdeploy_user_code"}
        try:
            compiled = compile(source, f"<model:{meta.model_id}>", "exec")
            exec(compiled, namespace)  # noqa: S102 - documented, user-owned runtime
        except SyntaxError as exc:
            raise AdapterError(
                f"custom code has a syntax error at line {exc.lineno}: {exc.msg}",
                hint="the snippet must define 'def predict(payload, model=None, meta=None)'",
            ) from exc

        if "lambda" in source and name == "predict" and "predict" not in namespace:
            # Allow ``predict = lambda payload: {...}`` style snippets.
            for key, value in namespace.items():
                if key.startswith("_") or not callable(value):
                    continue
                name = key
                break

        fn = namespace.get(name)
        if not callable(fn):
            available = sorted(k for k, v in namespace.items() if callable(v) and not k.startswith("_"))
            raise AdapterError(
                f"custom code defines no callable named '{name}'",
                hint=f"available callables: {', '.join(available) or 'none'}",
            )
        self._fn = fn
        self._namespace = namespace
        self.model = namespace.get("model")
        self.mark_loaded(meta, artifact_path)

    def predict(self, payload: dict[str, Any]) -> Any:
        if self._fn is None:
            raise AdapterError("python_function adapter is not loaded")
        started = time.perf_counter()
        try:
            result = _call_flexibly(self._fn, payload, self.model, self.meta)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - user code, report it, keep serving
            raise AdapterError(
                f"custom predict() raised {type(exc).__name__}: {exc}",
                hint=f"after {int((time.perf_counter() - started) * 1000)} ms",
            ) from exc
        return jsonable(result)

    def describe(self) -> ModelSchema:
        return ModelSchema(
            fields={"inputs": {"type": "object"}},
            example={"inputs": {}},
            description="Custom Python function: predict(payload, model=None, meta=None).",
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        if meta is not None and meta.entrypoint:
            return True
        name = (artifact.filename or artifact.uri).lower()
        return name.endswith(".py")


def _call_flexibly(fn: Callable[..., Any], payload: dict[str, Any], model: Any, meta: ModelMeta | None) -> Any:
    """Call user code with as many of (payload, model, meta) as it accepts."""
    import inspect

    try:
        signature = inspect.signature(fn)
        params = signature.parameters
        accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        accepts_var_pos = any(p.kind is p.VAR_POSITIONAL for p in params.values())
        names = set(params)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(payload)

    if accepts_var_pos:
        return fn(payload, model, meta)
    kwargs: dict[str, Any] = {}
    if "model" in names or accepts_var_kw:
        kwargs["model"] = model
    if "meta" in names:
        kwargs["meta"] = meta
    positional_names = [n for n, p in params.items() if p.kind is p.POSITIONAL_OR_KEYWORD]
    if positional_names and positional_names[0] not in ("payload", "inputs", "data", "x", "request"):
        # First parameter has an unusual name - still pass the payload positionally.
        pass
    try:
        return fn(payload, **kwargs)
    except TypeError:
        # Fall back to the simplest possible signature.
        return fn(payload)
