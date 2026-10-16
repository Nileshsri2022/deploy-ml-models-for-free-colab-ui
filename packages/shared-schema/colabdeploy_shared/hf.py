"""Hugging Face Hub helpers shared by the gateway, the agent and the UI.

Everything here is pure: no network, no transformers import. The same rules are
used to decide what the agent downloads and what the UI and gateway block or warn
about, so the three never disagree.

URI shapes (see docs/specs/huggingface-ui-spec.md, section 6.2):

* ``hf://<name>``                 legacy single-segment repo, e.g. ``distilgpt2``
* ``hf://<owner>/<name>``         owner-prefixed repo, e.g. ``Qwen/Qwen2.5-...``
* ``hf://<owner>/<name>/<file>``  one file from a repo (sub-folders allowed)

A Hub repo id is at most ``owner/name``, so the first two path segments are always
the repo and anything after them is a file path. This avoids the old bug where
``hf://Qwen/Qwen2.5-...`` was read as repo ``Qwen``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HF_SCHEME = "hf://"

# Tasks the huggingface adapter can feed through ``transformers.pipeline`` today
# (spec section 5, phase 1). Keep in sync with agent/colabdeploy/adapters/huggingface.py.
SUPPORTED_TASKS: tuple[str, ...] = (
    "text-classification",
    "token-classification",
    "zero-shot-classification",
    "text-generation",
    "summarization",
    "translation",
    "question-answering",
    "fill-mask",
    "feature-extraction",
    "image-classification",
)

_DIFFUSION_MARKERS = ("stable-diffusion", "sdxl", "controlnet", "diffusers")
_GPTQ_AWQ_MARKERS = ("gptq", "awq")
_CHAT_MARKERS = ("instruct", "-chat", "chat-", "assistant")
_REASONING_MARKERS = ("-r1", "deepseek-r1", "reasoning", "thinking")


@dataclass(frozen=True)
class HfRef:
    """A parsed ``hf://`` URI."""

    repo: str
    filename: str | None = None

    @property
    def is_repo_only(self) -> bool:
        """True when the whole repo is needed (the pipeline downloads it on load)."""
        return self.filename is None


def is_hf_uri(uri: str | None) -> bool:
    return bool(uri) and uri.strip().lower().startswith(HF_SCHEME)


def parse_hf_uri(uri: str) -> HfRef:
    """Split an ``hf://`` URI into repo id and optional file path.

    Raises ``ValueError`` for URIs that are not ``hf://`` or have an empty repo.
    """
    text = (uri or "").strip()
    if not is_hf_uri(text):
        raise ValueError(f"not a huggingface uri: {uri!r}")
    rest = text[len(HF_SCHEME):].strip("/")
    parts = [p for p in rest.split("/") if p]
    if not parts:
        raise ValueError("huggingface uri has no repo id")
    if len(parts) == 1:
        return HfRef(repo=parts[0])
    repo = f"{parts[0]}/{parts[1]}"
    filename = "/".join(parts[2:]) or None
    return HfRef(repo=repo, filename=filename)


@dataclass
class HfCheck:
    """Result of checking a Hub URI before deploy."""

    repo: str
    filename: str | None
    task: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blockers

    def as_dict(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "filename": self.filename,
            "task": self.task,
            "ok": self.ok,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def check_hf_uri(uri: str, task: str | None = None) -> HfCheck:
    """Apply the blocking and warning rules from the spec (sections 6.3 and 7).

    Works from the name only. It does not contact the Hub, so it cannot detect a
    missing or private repo; the agent reports those when it loads the model.
    """
    ref = parse_hf_uri(uri)
    chosen = (task or "").strip() or "text-classification"
    result = HfCheck(repo=ref.repo, filename=ref.filename, task=chosen)
    lowered = ref.repo.lower()
    file_lower = (ref.filename or "").lower()

    if file_lower.endswith(".gguf") or "gguf" in lowered:
        result.blockers.append(
            "GGUF models need a llama.cpp runtime, which this agent does not include. "
            "Choose a safetensors or PyTorch version of the model."
        )
    if any(marker in lowered for marker in _DIFFUSION_MARKERS):
        result.blockers.append(
            "Diffusion pipelines are not supported by the huggingface adapter. "
            "Use the python_function adapter with a .py file instead."
        )
    if chosen not in SUPPORTED_TASKS:
        result.blockers.append(
            f"task '{chosen}' is not supported yet. Supported tasks: {', '.join(SUPPORTED_TASKS)}."
        )

    if any(marker in lowered for marker in _GPTQ_AWQ_MARKERS):
        result.warnings.append(
            "Quantised (GPTQ/AWQ) model: the runtime needs the matching extra packages "
            "(for example optimum and auto-gptq). Install them before deploying, or the load will fail."
        )
    if any(marker in lowered for marker in _CHAT_MARKERS):
        result.warnings.append(
            "Instruction/chat model: inputs are sent as plain text, not as chat messages. "
            "Write prompts as completions, for example 'Answer briefly: ... Answer:'."
        )
    if any(marker in lowered for marker in _REASONING_MARKERS):
        result.warnings.append(
            "Reasoning model: answers include step-by-step thinking, so they are longer and slower."
        )
    return result
