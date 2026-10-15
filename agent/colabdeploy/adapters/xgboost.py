"""XGBoost adapter (also handles LightGBM artifacts saved as text/JSON).

Accepts the native ``.xgb``/``.ubj``/``.json`` booster formats plus joblib-pickled
sklearn-API estimators (``XGBClassifier``). Uses ``DMatrix`` when feature names are
known so a mismatch is caught at inference time rather than silently reordering
columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable, payload_table


class XGBoostAdapter(ModelAdapter):
    kind = AdapterKind.XGBOOST
    framework: tuple[str, ...] = ("xgboost", "numpy")
    priority = 18
    extensions = (".xgb", ".ubj", ".bst")

    def __init__(self) -> None:
        super().__init__()
        self.is_booster = False
        self.feature_names: list[str] | None = None

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        if not artifact_path.exists():
            raise AdapterError(f"artifact not found: {artifact_path}")
        name = artifact_path.name.lower()

        # Pickled sklearn-API estimator (XGBClassifier / LGBMClassifier / ...).
        if name.endswith((".pkl", ".pickle", ".joblib")):
            from .sklearn import SklearnAdapter

            delegate = SklearnAdapter()
            delegate.load(artifact_path, meta)
            self.model = delegate.model
            self.is_booster = False
            self.feature_names = delegate.feature_names
            self._delegate = delegate
            self.mark_loaded(meta, artifact_path)
            return

        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "xgboost is not installed in this runtime", hint="!pip install xgboost"
            ) from exc
        booster = xgb.Booster()
        try:
            booster.load_model(str(artifact_path))
        except Exception as exc:  # noqa: BLE001 - xgboost raises xgb.core.XGBoostError
            raise AdapterError(
                f"xgboost could not load '{artifact_path.name}': {exc}",
                hint="save with booster.save_model('m.json') or model.save_model('m.ubj')",
            ) from exc
        self.model = booster
        self.is_booster = True
        try:
            self.feature_names = list(booster.feature_names or []) or None
        except Exception:  # noqa: BLE001
            self.feature_names = None
        self.mark_loaded(meta, artifact_path)

    def predict(self, payload: dict[str, Any]) -> Any:
        if self.model is None:
            raise AdapterError("xgboost adapter is not loaded")
        columns, rows = payload_table(payload)
        if not self.is_booster:
            return self._delegate.predict(payload)  # type: ignore[attr-defined]

        import numpy as np  # type: ignore
        import xgboost as xgb  # type: ignore

        array = np.asarray(rows, dtype="float32")
        if self.feature_names and columns != self.feature_names:
            missing = [c for c in self.feature_names if c not in columns]
            if missing:
                raise AdapterError(
                    f"payload is missing booster features: {missing[:8]}",
                    hint=f"expected {self.feature_names[:12]}",
                )
            array = array[:, [columns.index(c) for c in self.feature_names]]
        try:
            matrix = xgb.DMatrix(array, feature_names=self.feature_names or columns)
            raw = self.model.predict(matrix)
        except Exception as exc:  # noqa: BLE001
            raise AdapterError(f"xgboost predict failed: {exc}") from exc

        values = jsonable(raw)
        output: dict[str, Any] = {"prediction": values}
        first = values[0] if isinstance(values, list) and values else values
        if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
            # Binary classifier margins -> a label alongside the probability.
            output["label"] = [1 if float(v) >= 0.5 else 0 for v in values]
            output["probability"] = values
        elif isinstance(first, list):
            output["label"] = [
                max(range(len(row)), key=lambda i: float(row[i])) for row in values if isinstance(row, list)
            ]
        return output

    def describe(self) -> ModelSchema:
        features = self.feature_names or []
        row = {name: 0.0 for name in features[:8]} or {"f0": 0.0}
        return ModelSchema(
            fields={"instances": {"type": "array"}, "columns": {"type": "array"}, "data": {"type": "array"}},
            required=["instances"],
            example={"instances": [row]},
            description=(
                f"XGBoost {'booster' if self.is_booster else 'estimator'}"
                + (f" with {len(features)} named features" if features else "")
            ),
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        name = (artifact.filename or artifact.uri).lower()
        return name.endswith(cls.extensions) or "xgb" in name
