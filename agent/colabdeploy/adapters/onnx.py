"""ONNX Runtime adapter - the best CPU story and the fastest cold start.

ONNX is the recommended export format for this platform: one file, no framework
runtime, ``onnxruntime`` installs in seconds on a free Colab CPU box, and the
session tells us its own input names/shapes so we can build the request schema
instead of guessing.

Input handling:
* ``{"inputs": {"float_input": [[1.0, 2.0]]}}`` - named, used verbatim
* ``{"instances": [{"a": 1}]}`` / ``{"columns": [...], "data": [...]}`` - mapped
  onto the session's first input in declared order
* ``{"image": "<b64>"}`` - decoded to NCHW float32 when the session wants 4-D
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable, payload_image_bytes, payload_table

_NP_KINDS = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(bool)": "bool",
    "tensor(uint8)": "uint8",
}


class OnnxAdapter(ModelAdapter):
    kind = AdapterKind.ONNX
    framework: tuple[str, ...] = ("onnxruntime", "numpy")
    priority = 15  # ahead of sklearn: an .onnx file is unambiguous
    extensions = (".onnx",)

    def __init__(self) -> None:
        super().__init__()
        self.input_specs: list[dict[str, Any]] = []
        self.output_names: list[str] = []
        self.providers: list[str] = []

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        try:
            import numpy as np  # type: ignore
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                f"onnxruntime is not installed ({exc.name})",
                hint="!pip install onnxruntime  (or onnxruntime-gpu for CUDA)",
            ) from exc
        if not artifact_path.exists():
            raise AdapterError(f"artifact not found: {artifact_path}")

        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in preferred if p in available] or available or ["CPUExecutionProvider"]
        try:
            session = ort.InferenceSession(
                str(artifact_path), providers=providers, sess_options=_session_options()
            )
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises many types
            raise AdapterError(
                f"onnxruntime could not open the model: {exc}",
                hint="re-export with opset <= 17, or check the file is a complete .onnx graph",
            ) from exc

        self.model = session
        self.providers = list(session.get_providers())
        self.input_specs = [
            {"name": i.name, "shape": [d if isinstance(d, int) else str(d) for d in (i.shape or [])],
             "type": i.type}
            for i in session.get_inputs()
        ]
        self.output_names = [o.name for o in session.get_outputs()]
        self._np = np
        self.mark_loaded(meta, artifact_path)
        self.health.detail = f"onnx session, providers={self.providers}"

    # ------------------------------------------------------------------ #
    def _feed(self, payload: dict[str, Any]) -> dict[str, Any]:
        np = self._np
        named = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else None
        feed: dict[str, Any] = {}

        if named:
            for spec in self.input_specs:
                name = spec["name"]
                if name not in named:
                    continue
                feed[name] = np.asarray(named[name], dtype=_NP_KINDS.get(spec["type"], "float32"))
            if feed:
                return feed

        # Image models: single 4-D input.
        first_shape = self.input_specs[0]["shape"] if self.input_specs else []
        if any(k in payload for k in ("image", "image_b64", "img")) and len(first_shape) == 4:
            raw = payload_image_bytes(payload)
            array = _decode_image_np(np, raw, first_shape)
            return {self.input_specs[0]["name"]: array}

        columns, rows = payload_table(payload)
        array = np.asarray(rows, dtype=_NP_KINDS.get(self.input_specs[0]["type"], "float32")) \
            if self.input_specs else np.asarray(rows, dtype="float32")
        if len(self.input_specs) == 1:
            return {self.input_specs[0]["name"]: array}

        # Multi-input graph: split columns across inputs positionally by width.
        offset = 0
        for spec in self.input_specs:
            width = spec["shape"][-1] if spec["shape"] and isinstance(spec["shape"][-1], int) else 1
            feed[spec["name"]] = np.asarray(
                array[:, offset : offset + width], dtype=_NP_KINDS.get(spec["type"], "float32")
            )
            offset += width
        return feed

    def predict(self, payload: dict[str, Any]) -> Any:
        if self.model is None:
            raise AdapterError("onnx adapter is not loaded")
        try:
            feed = self._feed(payload)
            outputs = self.model.run(self.output_names or None, feed)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 - runtime shape/dtype errors
            raise AdapterError(
                f"onnx inference failed: {exc}",
                hint=f"expected inputs {jsonable(self.input_specs)}",
            ) from exc
        result: dict[str, Any] = {}
        names = self.output_names or [f"output_{i}" for i in range(len(outputs))]
        for name, value in zip(names, outputs):
            result[name] = jsonable(value)
        primary = result[names[0]] if names else None
        return {
            "output": primary,
            "outputs": result,
            "providers": self.providers,
            "input_shapes": {k: jsonable(list(getattr(v, "shape", []))) for k, v in feed.items()},
        }

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[Any]:
        """ONNX batches natively - stack rows instead of looping."""
        if not payloads or len(self.input_specs) != 1:
            return super().predict_batch(payloads)
        try:
            np = self._np
            stacked = np.concatenate([self._feed(p)[self.input_specs[0]["name"]] for p in payloads], axis=0)
            outputs = self.model.run(self.output_names or None, {self.input_specs[0]["name"]: stacked})
            values = jsonable(outputs[0])
            if isinstance(values, list) and len(values) == len(payloads):
                return [{"output": v} for v in values]
        except Exception:  # noqa: BLE001 - fall back to per-item inference
            pass
        return super().predict_batch(payloads)

    def describe(self) -> ModelSchema:
        fields: dict[str, Any] = {
            "inputs": {
                "type": "object",
                "description": "named arrays keyed by model input name",
                "properties": {s["name"]: {"type": "array", "shape": s["shape"]} for s in self.input_specs},
            },
            "instances": {"type": "array"},
        }
        example: dict[str, Any]
        if self.input_specs:
            first = self.input_specs[0]
            width = first["shape"][-1] if isinstance(first["shape"][-1], int) else 1
            example = {"inputs": {first["name"]: [[0.0] * max(1, int(width))]}}
        else:
            example = {"instances": [[0.0]]}
        return ModelSchema(
            fields=fields,
            example=example,
            description=f"ONNX model · inputs {', '.join(s['name'] for s in self.input_specs) or 'unknown'} "
            f"· outputs {', '.join(self.output_names) or 'unknown'}",
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        return (artifact.filename or artifact.uri).lower().endswith(cls.extensions)


def _session_options() -> Any:
    import onnxruntime as ort  # type: ignore

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, (ort.__dict__.get("cpu_count") or 2))
    options.log_severity_level = 3  # keep Colab output readable
    return options


def _decode_image_np(np: Any, raw: bytes, shape: list[Any]) -> Any:
    """Best-effort image decode without forcing a Pillow dependency at import time."""
    try:
        from PIL import Image  # type: ignore
        import io

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        target_h = shape[2] if isinstance(shape[2], int) else image.height
        target_w = shape[3] if isinstance(shape[3], int) else image.width
        image = image.resize((int(target_w), int(target_h)))
        array = np.asarray(image, dtype="float32") / 255.0
        return np.transpose(array, (2, 0, 1))[None, ...]  # HWC -> NCHW
    except ImportError as exc:
        raise AdapterError(
            "image input needs Pillow", hint="!pip install Pillow"
        ) from exc
