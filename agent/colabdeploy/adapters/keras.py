"""TensorFlow / Keras adapter.

Supports ``.keras`` (Keras 3), ``.h5`` (legacy) and a ``SavedModel`` directory.
Tabular payloads become numpy arrays; image payloads are decoded and resized to
the model's declared input shape, which is read from the graph rather than
hard-coded so the same adapter serves an MLP and a CNN.

TensorFlow is heavy (~600 MB) so it is imported lazily and a failure to import
produces an actionable hint instead of a traceback in the Colab output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable, payload_image_bytes, payload_table


class KerasAdapter(ModelAdapter):
    kind = AdapterKind.KERAS
    framework: tuple[str, ...] = ("keras|tensorflow", "numpy")
    priority = 40
    extensions = (".h5", ".keras", ".pb")

    def __init__(self) -> None:
        super().__init__()
        self.input_shape: list[Any] = []
        self.output_shape: list[Any] = []
        self.is_image_model = False

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        try:
            # Keras 3 ships standalone; fall back to tf.keras for older artifacts.
            try:
                import keras  # type: ignore
            except ImportError:
                from tensorflow import keras  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "neither keras nor tensorflow is installed in this runtime",
                hint="!pip install keras  (or tensorflow-cpu); or export to ONNX for a lighter runtime",
            ) from exc
        if not artifact_path.exists():
            raise AdapterError(f"artifact not found: {artifact_path}")

        try:
            if artifact_path.is_dir():
                model = keras.models.load_model(str(artifact_path))
            elif (artifact_path / "saved_model.pb").exists():
                model = keras.models.load_model(str(artifact_path))
            else:
                model = keras.models.load_model(str(artifact_path), compile=False)
        except Exception as exc:  # noqa: BLE001 - keras raises many types
            raise AdapterError(
                f"keras could not load '{artifact_path.name}': {exc}",
                hint="was it saved with model.save('m.keras') / model.save('m.h5')?",
            ) from exc

        self.model = model
        try:
            self.input_shape = [jsonable(d) for d in model.input_shape] if model.input_shape else []
            self.output_shape = [jsonable(d) for d in model.output_shape] if model.output_shape else []
        except (AttributeError, TypeError, ValueError):  # pragma: no cover - multi-input graphs
            self.input_shape, self.output_shape = [], []
        self.is_image_model = len(self.input_shape) == 4
        self.mark_loaded(meta, artifact_path)
        self.health.detail = f"keras model, input {self.input_shape}"

    def _array(self, payload: dict[str, Any]) -> Any:
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise AdapterError("numpy is required for keras inference", hint="!pip install numpy") from exc

        if self.is_image_model and any(k in payload for k in ("image", "image_b64", "img")):
            raw = payload_image_bytes(payload)
            return _decode_image_np(np, raw, self.input_shape)

        if "inputs" in payload and isinstance(payload["inputs"], list):
            data: Any = payload["inputs"]
        else:
            _, rows = payload_table(payload)
            data = rows
        try:
            array = np.asarray(data, dtype="float32")
        except (ValueError, TypeError) as exc:
            raise AdapterError(f"could not build an array from the payload: {exc}") from exc

        expected_rank = len(self.input_shape)
        if expected_rank and array.ndim != expected_rank:
            try:
                array = array.reshape((1, *array.shape)) if array.ndim == expected_rank - 1 else array
            except ValueError:  # pragma: no cover
                pass
        return array

    def predict(self, payload: dict[str, Any]) -> Any:
        if self.model is None:
            raise AdapterError("keras adapter is not loaded")
        array = self._array(payload)
        task = str(payload.get("task", "predict"))
        fn = getattr(self.model, task, None)
        if not callable(fn):
            raise AdapterError(
                f"keras model has no '{task}' method", hint="try task='predict' or 'predict_proba'"
            )
        try:
            raw = fn(array, verbose=0) if task.startswith("predict") else fn(array)
        except Exception as exc:  # noqa: BLE001 - shape mismatches are common
            raise AdapterError(
                f"keras {task} failed: {exc}",
                hint=f"model input shape {self.input_shape}, got {jsonable(list(getattr(array,'shape',[])))}",
            ) from exc
        values = jsonable(raw)
        output: dict[str, Any] = {"output": values, "input_shape": jsonable(list(array.shape))}
        if task in ("predict_proba", "predict") and isinstance(values, list) and values:
            first = values[0]
            if isinstance(first, list) and len(first) > 1:
                output["label"] = [max(range(len(r)), key=lambda i: float(r[i])) for r in values if isinstance(r, list)]
            elif isinstance(first, (int, float)) and len(first) == 1:
                output["label"] = [1 if float(r[0]) >= 0.5 else 0 for r in values if isinstance(r, list)]
        return output

    def describe(self) -> ModelSchema:
        if self.is_image_model:
            return ModelSchema(
                fields={"image": {"type": "string", "description": "base64 or data URI"}},
                required=["image"],
                example={"image": "<base64>"},
                description=f"Keras image model expecting input {self.input_shape}",
            )
        width = self.input_shape[-1] if self.input_shape and isinstance(self.input_shape[-1], int) else 1
        return ModelSchema(
            fields={
                "inputs": {"type": "array"},
                "instances": {"type": "array"},
                "task": {"type": "string", "enum": ["predict", "predict_proba"]},
            },
            example={"inputs": [[0.0] * max(1, int(width))]},
            description=f"Keras model, input {self.input_shape}, output {self.output_shape}",
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        uri = (artifact.filename or artifact.uri).lower()
        return uri.endswith(cls.extensions) or "savedmodel" in uri or "saved_model" in uri


def _decode_image_np(np: Any, raw: bytes, input_shape: list[Any]) -> Any:
    try:
        import io

        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise AdapterError("image input needs Pillow", hint="!pip install Pillow") from exc
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    height = input_shape[1] if len(input_shape) == 4 and isinstance(input_shape[1], int) else image.height
    width = input_shape[2] if len(input_shape) == 4 and isinstance(input_shape[2], int) else image.width
    image = image.resize((int(width), int(height)))
    array = np.asarray(image, dtype="float32") / 255.0
    channels_last = len(input_shape) == 4 and input_shape[-1] in (1, 3, 4)
    if channels_last:
        return array[None, ...]
    return np.transpose(array, (2, 0, 1))[None, ...]
