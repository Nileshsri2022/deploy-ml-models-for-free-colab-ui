"""Echo adapter - the end-to-end test double and the "hello world" deploy.

It needs no ML dependency at all, which makes it the perfect first deployment:
you can prove the tunnel, the registry, the cache and the UI in about ten
seconds, before spending a minute downloading scikit-learn.

It also models a deterministic function of the payload, so cached answers are
*correct* answers - ideal for exercising the HIT/STALE/MISS paths in tests.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta


class EchoAdapter(ModelAdapter):
    kind = AdapterKind.ECHO
    priority = 10  # tried early for .json / unknown artifacts in auto mode
    extensions = (".echo.json",)

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        config: dict[str, Any] = {}
        if artifact_path.exists() and artifact_path.stat().st_size:
            try:
                config = json.loads(artifact_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                config = {}  # an empty/placeholder artifact is fine for echo
        if not isinstance(config, dict):
            raise AdapterError("echo artifact must be a JSON object")
        self.model = config
        self.mark_loaded(meta, artifact_path)

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.model is None:
            raise AdapterError("echo adapter is not loaded")
        latency_s = float(self.model.get("latency_ms", 0)) / 1000.0
        if latency_s:
            time.sleep(min(latency_s, 5.0))
        if self.model.get("fail"):
            raise AdapterError(str(self.model["fail"]))
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        features = [v for v in payload.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return {
            "echo": payload,
            "model_id": self.meta.model_id if self.meta else None,
            "model_version": self.meta.model_version if self.meta else None,
            "adapter": self.kind.value,
            "sha256": digest,
            "numeric_sum": round(sum(features), 6) if features else None,
            "feature_count": len(payload),
            "device": "cpu",
            "served_at_ms": int(time.time() * 1000),
        }

    def describe(self) -> ModelSchema:
        return ModelSchema(
            fields={"message": {"type": "string"}, "value": {"type": "number"}},
            required=[],
            example={"message": "hello", "value": 1},
            description="Echoes the payload back with a digest - no ML dependency required.",
        )

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        name = (artifact.filename or artifact.uri).lower()
        return name.endswith(cls.extensions) or "echo" in name
