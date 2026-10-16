"""HuggingFace adapter - ``transformers`` pipelines and Hub artifacts.

Two ways to use it:

* ``artifact.uri = "hf://<repo>"`` (optionally ``hf://<repo>/<file>``) with
  ``extra = {"task": "sentiment-analysis", "device": "cuda"}``; we build a
  pipeline and let HF handle the download/cache.
* A local file (``.safetensors``, ``pytorch_model.bin``, ``tokenizer.json``) plus
  the same ``task`` hint.

Output shape varies wildly by task, so the adapter normalises to
``{"output": ..., "task": ..., "labels"/"text"/...}`` and always includes the raw
pipeline result under ``output`` for callers that know what they want.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.hf import parse_hf_uri
from colabdeploy_shared.schemas import ArtifactSpec, ModelSchema

from .base import AdapterError, ModelAdapter, ModelMeta, jsonable, payload_image_bytes, payload_text


class HuggingFaceAdapter(ModelAdapter):
    kind = AdapterKind.HUGGINGFACE
    framework: tuple[str, ...] = ("transformers",)
    priority = 25
    extensions = (".safetensors",)

    def __init__(self) -> None:
        super().__init__()
        self.task: str = ""
        self.repo: str = ""
        self.device: str | None = None
        self.model_name: str = ""

    def load(self, artifact_path: Path, meta: ModelMeta) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise AdapterError(
                "transformers is not installed in this runtime",
                hint="!pip install transformers accelerate",
            ) from exc

        extra = meta.extra or {}
        self.task = str(extra.get("task") or meta.entrypoint or "text-classification")
        uri = meta.artifact.uri
        if uri.startswith("hf://"):
            self.repo = parse_hf_uri(uri).repo
        elif artifact_path.is_dir():
            self.repo = str(artifact_path)
        else:
            self.repo = str(extra.get("repo") or meta.name or "")
        if not self.repo:
            raise AdapterError(
                "huggingface adapter needs a repo id",
                hint='set artifact.uri = "hf://distilbert-base-uncased-finetuned-sst-2-english"',
            )
        self.device = str(extra.get("device") or (-1 if extra.get("device") != "cuda" else 0))

        kwargs: dict[str, Any] = {}
        for key in ("revision", "trust_remote_code", "model_kwargs", "tokenizer"):
            if key in extra:
                kwargs[key] = extra[key]
        try:
            import torch  # type: ignore

            device_value: Any = extra.get("device")
            if device_value in (None, "", "auto"):
                device_value = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(self.task, model=self.repo, device=device_value, **kwargs)
            self.device = str(device_value)
        except ImportError:
            self.pipe = pipeline(self.task, model=self.repo, **kwargs)
            self.device = "cpu"
        except Exception as exc:
            raise AdapterError(
                f"could not build pipeline for task '{self.task}' on '{self.repo}': {exc}",
                hint="check the repo id, that the task matches the model, and that the VM has "
                "network access to huggingface.co",
            ) from exc

        self.model = self.pipe
        self.model_name = getattr(getattr(self.pipe, "model", None), "name_or_path", self.repo)
        self.mark_loaded(meta, artifact_path)
        self.health.detail = f"hf pipeline task={self.task} model={self.repo} device={self.device}"

    # ------------------------------------------------------------------ #
    def _input(self, payload: dict[str, Any]) -> Any:
        task = self.task
        vision = ("image-classification", "object-detection", "image-segmentation", "depth-estimation")
        if task in vision and ("image" in payload or "image_b64" in payload):
            import io

            from PIL import Image  # type: ignore

            return Image.open(io.BytesIO(payload_image_bytes(payload)))
        if "text" in payload and isinstance(payload["text"], list):
            return payload["text"]
        if task in ("text-generation", "text2text-generation", "summarization", "translation", "question-answering"):
            if task == "question-answering":
                return {
                    "question": str(payload.get("question", "")),
                    "context": str(payload.get("context", "")),
                }
            return payload_text(payload)
        if "inputs" in payload:
            return payload["inputs"]
        return payload_text(payload)

    def predict(self, payload: dict[str, Any]) -> Any:
        if self.model is None:
            raise AdapterError("huggingface adapter is not loaded")
        data = self._input(payload)
        kwargs = {k: v for k, v in payload.items() if k in _PIPELINE_KWARGS}
        try:
            raw = self.model(data, **kwargs)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"pipeline '{self.task}' failed: {exc}",
                hint="reduce max_length/batch size, or check the input matches the task",
            ) from exc
        output = jsonable(raw)
        result: dict[str, Any] = {"output": output, "task": self.task, "model": self.model_name}
        if isinstance(output, list) and output and isinstance(output[0], dict):
            best = output[0] if len(output[0]) == 1 else max(
                output, key=lambda item: float(item.get("score", 0)) if isinstance(item, dict) else 0
            )
            if isinstance(best, dict):
                result["label"] = best.get("label")
                result["score"] = best.get("score")
                if "generated_text" in best:
                    result["text"] = best["generated_text"]
        return result

    def describe(self) -> ModelSchema:
        if "image" in self.task:
            return ModelSchema(
                fields={"image": {"type": "string"}},
                required=["image"],
                example={"image": "<base64>"},
                description=f"HF {self.task} pipeline on {self.repo}",
            )
        fields: dict[str, Any] = {"text": {"type": "string"}}
        if self.task == "question-answering":
            fields = {"question": {"type": "string"}, "context": {"type": "string"}}
        return ModelSchema(
            fields=fields,
            required=list(fields)[:1],
            example=dict.fromkeys(fields, "…"),
            description=f"HF {self.task} pipeline on {self.repo}",
        )

    def unload(self) -> None:
        super().unload()
        self.pipe = None  # type: ignore[assignment]
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def detect(cls, artifact: ArtifactSpec, meta: ModelMeta | None = None) -> bool:
        uri = (artifact.uri or "").lower()
        name = (artifact.filename or "").lower()
        return uri.startswith("hf://") or name.endswith(cls.extensions) or "pytorch_model.bin" in uri


_PIPELINE_KWARGS = {
    "max_length",
    "max_new_tokens",
    "num_return_sequences",
    "top_k",
    "top_p",
    "temperature",
    "do_sample",
    "truncation",
    "return_tensors",
    "candidate_labels",
}
