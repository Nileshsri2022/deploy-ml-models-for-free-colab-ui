"""scikit-learn / joblib adapter.

Handles pickled estimators and full ``Pipeline`` objects. Beyond ``predict`` it
exposes the other estimator surface people actually call in production:
``predict_proba``, ``decision_function``, ``transform`` (for feature pipelines)
and a combined ``predict`` + ``probability`` output for classifiers.

If the pickle needs a class that is not installed in the VM, loading fails with a
clear hint naming the missing module - the single most common Colab deploy error.
"""

from __future__ import annotations

import importlib
import pickle
from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable, payload_table

_PROBA_TASKS = ("predict_proba", "decision_function")


class SklearnAdapter(ModelAdapter):
    kind = AdapterKind.SKLEARN
    #: framework modules that must import for this adapter to work
    framework: tuple[str, ...] = ("sklearn", "numpy")
    priority = 20
    extensions = (".pkl", ".pickle", ".joblib", ".job", ".sklearn")

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: list[str] | None = None
        self.classes: list[Any] | None = None
        self.tasks: list[str] = []

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        if not artifact_path.exists():
            raise AdapterError(f"artifact not found: {artifact_path}")
        try:
            import joblib  # type: ignore

            loader: Any = joblib.load
        except ImportError:
            loader = _pickle_load
        try:
            model = loader(str(artifact_path))
        except ModuleNotFoundError as exc:
            raise AdapterError(
                f"loading this model needs the '{exc.name}' package",
                hint=f"run  !pip install {exc.name}  in Colab and re-deploy",
            ) from exc
        except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as exc:
            raise AdapterError(
                f"could not unpickle artifact: {exc}",
                hint="was it saved with joblib.dump / pickle.dump from scikit-learn?",
            ) from exc

        if not hasattr(model, "predict") and not hasattr(model, "transform"):
            raise AdapterError(
                f"loaded object of type {type(model).__name__} has no predict()/transform()",
                hint="sklearn adapter expects an estimator, pipeline or transformer",
            )
        self.model = model
        self.feature_names = list(getattr(model, "feature_names_in_", []) or []) or None
        raw_classes = getattr(model, "classes_", None)
        self.classes = jsonable(list(raw_classes)) if raw_classes is not None else None
        self.tasks = [t for t in _PROBA_TASKS if hasattr(model, t)]
        if hasattr(model, "predict"):
            self.tasks.insert(0, "predict")
        if not hasattr(model, "predict") and hasattr(model, "transform"):
            self.tasks = ["transform"]
        self.mark_loaded(meta, artifact_path)

    def _frame(self, payload: dict[str, Any]) -> Any:
        columns, rows = payload_table(payload)
        try:
            import pandas as pd  # type: ignore

            if self.feature_names and set(columns) != set(self.feature_names):
                # Align/reindex to the training-time column order when possible.
                frame = pd.DataFrame(rows, columns=columns)
                missing = [c for c in self.feature_names if c not in frame.columns]
                if missing:
                    raise AdapterError(
                        f"payload is missing training features: {missing[:8]}",
                        hint=f"expected columns {self.feature_names[:12]}",
                    )
                return frame[self.feature_names]
            return pd.DataFrame(rows, columns=columns)
        except ImportError:
            try:
                import numpy as np  # type: ignore

                return np.array(rows, dtype=object)
            except ImportError as exc:  # pragma: no cover
                raise AdapterError(
                    "install pandas or numpy to serve tabular models", hint="!pip install pandas"
                ) from exc

    def predict(self, payload: dict[str, Any]) -> Any:
        if self.model is None:
            raise AdapterError("sklearn adapter is not loaded")
        task = str(payload.get("task") or self.tasks[0])
        if task not in self.tasks and hasattr(self.model, task):
            self.tasks.append(task)
        if not hasattr(self.model, task):
            raise AdapterError(
                f"model does not support task '{task}'",
                hint=f"supported: {', '.join(self.tasks)}",
            )
        features = self._frame(payload)
        fn = getattr(self.model, task)
        raw = fn(features)

        if task == "predict" and self.classes and _looks_like_indices(raw):
            # Classifiers often return integer indices; give callers the labels too.
            labels = _map_labels(raw, self.classes)
            out: dict[str, Any] = {"prediction": jsonable(raw), "label": labels}
            if "predict_proba" in self.tasks:
                proba = jsonable(self.model.predict_proba(features))
                out["probability"] = proba
                out["classes"] = self.classes
                out["top"] = _top_class(proba, self.classes)
            return out
        if task in _PROBA_TASKS:
            return {"probability": jsonable(raw), "classes": self.classes}
        if task == "transform":
            return {"transformed": jsonable(raw)}
        return {"prediction": jsonable(raw)}

    def describe(self) -> ModelSchema:
        fields: dict[str, Any] = {
            "instances": {"type": "array", "description": "list of row objects"},
            "task": {"type": "string", "enum": self.tasks or ["predict"]},
        }
        example: dict[str, Any]
        if self.feature_names:
            row = dict.fromkeys(self.feature_names[:8], 0)
            example = {"instances": [row]}
        else:
            example = {"instances": [{"x": 0.0}]}
        return ModelSchema(
            fields=fields,
            required=["instances"],
            example=example,
            description=(
                f"scikit-learn model with {len(self.feature_names or [])} known features; "
                f"tasks: {', '.join(self.tasks) or 'predict'}"
            ),
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        name = (artifact.filename or artifact.uri).lower()
        return name.endswith(cls.extensions)


def _pickle_load(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _looks_like_indices(raw: Any) -> bool:
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(raw)
        return arr.dtype.kind in "iu"
    except Exception:
        return all(isinstance(v, int) and not isinstance(v, bool) for v in list(raw)[:5]) if hasattr(
            raw, "__iter__"
        ) else isinstance(raw, int)


def _map_labels(raw: Any, classes: list[Any]) -> Any:
    try:
        indices = list(raw) if hasattr(raw, "__iter__") else [raw]
        return [classes[int(i)] if 0 <= int(i) < len(classes) else None for i in indices]
    except (TypeError, ValueError):  # pragma: no cover
        return None


def _top_class(proba: Any, classes: list[Any]) -> Any:
    try:
        rows = list(proba)
        first = rows[0] if rows else []
        if not first:
            return None
        idx = max(range(len(first)), key=lambda i: float(first[i]))
        return {"class": classes[idx] if idx < len(classes) else idx, "probability": float(first[idx])}
    except (TypeError, ValueError, IndexError):  # pragma: no cover
        return None


def module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False
