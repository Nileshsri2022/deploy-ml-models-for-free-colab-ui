"""PyTorch adapter.

Supports the three things people actually save:

* ``torch.save(model.state_dict(), ...)``  - needs the class, so we also accept a
  companion ``arch`` entrypoint that rebuilds the module before loading weights;
* ``torch.save(model, ...)``               - whole-object pickle;
* ``torch.jit.save(...)``                  - TorchScript, loads with no source at all
  (the most reliable way to ship a model to Colab).

Inputs are converted to tensors from either a flat numeric list (``{"inputs": [..]}``),
a tabular payload, or a batch dimension supplied as ``{"shape": [1, 3, 224, 224]}``.
Device placement prefers CUDA when the VM has one and falls back to CPU with a log
line - a model deployed on a T4 must still serve after Colab hands us a CPU box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable, payload_table


class PyTorchAdapter(ModelAdapter):
    kind = AdapterKind.PYTORCH
    framework: tuple[str, ...] = ("torch",)
    priority = 30
    extensions = (".pt", ".pth", ".pt2", ".ckpt", ".torchscript")

    def __init__(self) -> None:
        super().__init__()
        self.device = "cpu"
        self.is_scripted = False
        self.input_dtype: str = "float32"

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "torch is not installed in this runtime",
                hint="!pip install torch  (or use the ONNX adapter for CPU-only serving)",
            ) from exc
        if not artifact_path.exists():
            raise AdapterError(f"artifact not found: {artifact_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model: Any = None

        # 1) TorchScript: no source needed.
        try:
            model = torch.jit.load(str(artifact_path), map_location=self.device)
            self.is_scripted = True
        except (RuntimeError, ValueError, EOFError):
            model = None

        # 2) Whole-object pickle.
        if model is None:
            try:
                model = torch.load(str(artifact_path), map_location=self.device, weights_only=False)
            except Exception as first_exc:  # noqa: BLE001
                # 3) state_dict + a user-supplied architecture factory.
                state = _try_state_dict(torch, artifact_path)
                if state is None:
                    raise AdapterError(
                        f"could not load torch artifact: {first_exc}",
                        hint="save with torch.jit.save, torch.save(model), or supply an "
                        "'arch' entrypoint that returns the module",
                    ) from first_exc
                model = _build_from_entrypoint(meta, state, torch, self.device)

        if isinstance(model, dict):
            raise AdapterError(
                "artifact contains a state_dict but no architecture",
                hint="deploy with adapter='python_function' and an entrypoint that builds "
                "the module, or export to TorchScript/ONNX",
            )

        try:
            model.to(self.device)
            model.eval()
        except AttributeError as exc:  # pragma: no cover - not an nn.Module
            raise AdapterError(f"loaded object is not a torch module: {exc}") from exc
        except RuntimeError as exc:  # e.g. CUDA OOM at load time
            self.device = "cpu"
            try:
                model.to("cpu")
                model.eval()
            except Exception:  # noqa: BLE001
                raise AdapterError(f"could not place model on CPU either: {exc}") from exc

        self.model = model
        self.mark_loaded(meta, artifact_path)
        self.health.detail = f"torch model on {self.device}" + (" (TorchScript)" if self.is_scripted else "")

    def _tensor(self, payload: dict[str, Any]) -> Any:
        import torch  # type: ignore

        dtype = getattr(torch, str(payload.get("dtype") or self.input_dtype), torch.float32)
        if "tensor" in payload and isinstance(payload["tensor"], list):
            data: Any = payload["tensor"]
        elif any(k in payload for k in ("instances", "columns", "features", "rows", "X", "data")):
            _, rows = payload_table(payload)
            data = rows
        elif "inputs" in payload:
            data = payload["inputs"]
        else:
            values = [v for v in payload.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if not values:
                raise AdapterError(
                    "could not build a tensor from this payload",
                    hint='send {"inputs": [[1.0, 2.0]]} or {"instances": [{"x": 1.0}]}',
                )
            data = [values]
        try:
            tensor = torch.as_tensor(data, dtype=dtype)
        except (ValueError, TypeError, RuntimeError) as exc:
            raise AdapterError(f"could not convert payload to a {dtype} tensor: {exc}") from exc
        shape = payload.get("shape")
        if isinstance(shape, list) and shape:
            try:
                tensor = tensor.reshape(shape)
            except RuntimeError as exc:
                raise AdapterError(f"cannot reshape to {shape}: {exc}") from exc
        return tensor.to(self.device)

    def predict(self, payload: dict[str, Any]) -> Any:
        if self.model is None:
            raise AdapterError("pytorch adapter is not loaded")
        import torch  # type: ignore

        tensor = self._tensor(payload)
        no_grad = getattr(torch, "no_grad", None)
        try:
            with (no_grad() if no_grad else _nullcontext()):
                if str(payload.get("mode", "forward")) == "inference":
                    fn = getattr(self.model, "inference", None) or getattr(self.model, "predict", None)
                    output = fn(tensor) if callable(fn) else self.model(tensor)
                else:
                    output = self.model(tensor)
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                try:
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            raise AdapterError(f"forward pass failed: {exc}", hint=f"input shape {list(tensor.shape)}") from exc
        if isinstance(output, (tuple, list)):
            output = output[0] if output else None
        return {
            "output": jsonable(output),
            "device": self.device,
            "input_shape": jsonable(list(tensor.shape)),
            "scripted": self.is_scripted,
        }

    def describe(self) -> ModelSchema:
        return ModelSchema(
            fields={
                "inputs": {"type": "array", "description": "nested numeric list"},
                "shape": {"type": "array", "description": "optional reshape, e.g. [1,3,224,224]"},
                "dtype": {"type": "string", "enum": ["float32", "float64", "long", "int64"]},
            },
            required=["inputs"],
            example={"inputs": [[0.0, 0.0]]},
            description=f"PyTorch module on {self.device}" + (" (TorchScript)" if self.is_scripted else ""),
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        name = (artifact.filename or artifact.uri).lower()
        if name.endswith(cls.extensions):
            return True
        return "pytorch_model" in name or "torchscript" in name


def _try_state_dict(torch: Any, artifact_path: Path) -> Any | None:
    try:
        obj = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        try:
            obj = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            return None
    if isinstance(obj, dict) and all(isinstance(k, str) for k in list(obj)[:8]):
        return obj
    return None


def _build_from_entrypoint(meta: ModelMeta, state: Any, torch: Any, device: str) -> Any:
    """Use ``entrypoint`` source to rebuild the architecture, then load weights."""
    if not meta.entrypoint:
        return state
    namespace: dict[str, Any] = {"torch": torch, "__name__": "colabdeploy_arch"}
    try:
        exec(compile(meta.entrypoint, f"<arch:{meta.model_id}>", "exec"), namespace)  # noqa: S102
    except SyntaxError as exc:
        raise AdapterError(f"architecture snippet has a syntax error: {exc.msg}") from exc
    factory = namespace.get("build_model") or namespace.get("arch") or namespace.get("model_fn")
    if not callable(factory):
        raise AdapterError(
            "architecture snippet must define build_model()",
            hint="def build_model(): return MyNet(...)",
        )
    module = factory()
    missing = module.load_state_dict(state, strict=False)
    return module.to(device)


class _nullcontext:  # pragma: no cover - torch always has no_grad in practice
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False
