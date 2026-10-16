"""Hugging Face URI handling, pre-deploy checks, and the spec's blocking rules.

Covers docs/specs/huggingface-ui-spec.md sections 6.2 (repo resolution) and 6.3 / 7
(validation). No network: the Hub is never contacted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from colabdeploy.artifacts import ArtifactCache, ArtifactError, ArtifactResolver
from colabdeploy_shared.hf import (
    SUPPORTED_TASKS,
    HfRef,
    check_hf_uri,
    is_hf_uri,
    parse_hf_uri,
)
from colabdeploy_shared.schemas import ArtifactSpec


# --------------------------------------------------------------------------- #
# parse_hf_uri: the owner/name bug fix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("hf://distilgpt2", HfRef("distilgpt2", None)),
        ("hf://distilbert-base-uncased-finetuned-sst-2-english", HfRef("distilbert-base-uncased-finetuned-sst-2-english", None)),
        ("hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", HfRef("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", None)),
        ("hf://deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", HfRef("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", None)),
        ("hf://org/repo/model.onnx", HfRef("org/repo", "model.onnx")),
        ("hf://org/repo/onnx/model.onnx", HfRef("org/repo", "onnx/model.onnx")),
        ("  hf://org/repo/  ", HfRef("org/repo", None)),
    ],
)
def test_parse_keeps_owner_and_name_together(uri: str, expected: HfRef) -> None:
    assert parse_hf_uri(uri) == expected


def test_parse_rejects_non_hf_and_empty_uris() -> None:
    with pytest.raises(ValueError, match="not a huggingface uri"):
        parse_hf_uri("file:///content/m.pkl")
    with pytest.raises(ValueError, match="no repo id"):
        parse_hf_uri("hf://")
    with pytest.raises(ValueError, match="no repo id"):
        parse_hf_uri("hf:///")


def test_is_hf_uri_is_case_insensitive() -> None:
    assert is_hf_uri("HF://org/repo")
    assert not is_hf_uri("https://huggingface.co/org/repo")
    assert not is_hf_uri(None)


# --------------------------------------------------------------------------- #
# check_hf_uri: blocking and warning rules (spec section 7)
# --------------------------------------------------------------------------- #
def test_supported_task_list_is_the_phase_one_set() -> None:
    assert "text-generation" in SUPPORTED_TASKS
    assert "image-classification" in SUPPORTED_TASKS
    assert "automatic-speech-recognition" not in SUPPORTED_TASKS  # phase 2


def test_plain_text_model_passes_without_notes() -> None:
    check = check_hf_uri("hf://distilgpt2", "text-generation")
    assert check.ok
    assert check.blockers == [] and check.warnings == []


def test_gguf_is_blocked_by_name_or_file() -> None:
    assert not check_hf_uri("hf://TheBloke/Some-Model-GGUF").ok
    blocked = check_hf_uri("hf://TheBloke/Some-Model-GGUF/model.Q4_K_M.gguf")
    assert any("llama.cpp" in message for message in blocked.blockers)


def test_diffusion_pipelines_are_blocked() -> None:
    check = check_hf_uri("hf://stabilityai/stable-diffusion-xl-base-1.0")
    assert not check.ok
    assert any("python_function" in message for message in check.blockers)


def test_unsupported_task_is_blocked_and_lists_alternatives() -> None:
    check = check_hf_uri("hf://distilgpt2", "automatic-speech-recognition")
    assert not check.ok
    assert "automatic-speech-recognition" in check.blockers[0]
    assert "text-generation" in check.blockers[0]


def test_quantised_model_warns_about_packages() -> None:
    check = check_hf_uri("hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "text-generation")
    assert check.ok
    assert any("GPTQ" in warning and "auto-gptq" in warning for warning in check.warnings)


def test_chat_and_reasoning_models_warn_but_do_not_block() -> None:
    chat = check_hf_uri("hf://Qwen/Qwen2.5-0.5B-Instruct", "text-generation")
    assert chat.ok and any("plain text" in w for w in chat.warnings)
    reasoning = check_hf_uri("hf://deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "text-generation")
    assert reasoning.ok and any("Reasoning" in w for w in reasoning.warnings)


def test_default_task_is_text_classification() -> None:
    assert check_hf_uri("hf://distilbert-base-uncased-finetuned-sst-2-english").task == "text-classification"


# --------------------------------------------------------------------------- #
# Agent side: a repo-only owner/name URI is a pipeline download, not a file fetch
# --------------------------------------------------------------------------- #
@pytest.fixture()
def resolver(tmp_path: Path) -> Any:
    cache = ArtifactCache(tmp_path / "artifact-cache")
    yield ArtifactResolver(cache, gateway_url="http://gateway.test", runtime_token="CDP_RT_test")
    cache.close()


def test_owner_name_repo_is_a_placeholder_not_a_file_download(resolver: ArtifactResolver) -> None:
    resolved = resolver.resolve(ArtifactSpec(uri="hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"))
    assert resolved.source == "hf"
    assert resolved.is_dir
    assert resolved.size_bytes == 0


def test_owner_name_with_file_downloads_that_one_file(
    resolver: ArtifactResolver, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file path means a real single-file download. The Hub call is stubbed here."""
    import sys
    import types

    calls: list[dict[str, str]] = []
    source = tmp_path / "downloaded.onnx"
    source.write_bytes(b"onnx-bytes")

    def fake_download(repo_id: str, filename: str) -> str:
        calls.append({"repo_id": repo_id, "filename": filename})
        return str(source)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    resolved = resolver.resolve(ArtifactSpec(uri="hf://org/repo/model.onnx"))
    assert calls == [{"repo_id": "org/repo", "filename": "model.onnx"}]
    assert resolved.source == "hf"
    assert resolved.path.read_bytes() == b"onnx-bytes"


def test_failed_file_download_reports_the_repo_and_file(
    resolver: ArtifactResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    def boom(repo_id: str, filename: str) -> str:
        raise RuntimeError("404 not found")

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(ArtifactError, match=r"org/repo/model\.onnx"):
        resolver.resolve(ArtifactSpec(uri="hf://org/repo/model.onnx"))


# --------------------------------------------------------------------------- #
# Gateway: check endpoint and registration blocking
# --------------------------------------------------------------------------- #
def test_hf_check_endpoint_reports_warnings(api: Any) -> None:
    response = api.post(
        "/api/models/hf-check",
        json={"uri": "hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "task": "text-generation"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["repo"] == "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"
    assert body["warnings"]


def test_hf_check_endpoint_rejects_non_hf_uris(api: Any) -> None:
    response = api.post("/api/models/hf-check", json={"uri": "file:///m.pkl"})
    assert response.status_code == 400


def test_registering_a_gguf_model_is_blocked(api: Any) -> None:
    response = api.post(
        "/api/models",
        json={"name": "gguf-model", "adapter": "huggingface",
              "artifact": {"uri": "hf://TheBloke/Some-Model-GGUF/m.gguf"}},
    )
    assert response.status_code == 400
    assert "llama.cpp" in response.json()["error"]["message"]


def test_registering_an_owner_prefixed_model_succeeds(api: Any) -> None:
    response = api.post(
        "/api/models",
        json={"name": "qwen-gptq", "adapter": "huggingface", "entrypoint": "text-generation",
              "artifact": {"uri": "hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"}},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["adapter"] == "huggingface"
    assert body["artifact"]["uri"] == "hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"


def test_registering_with_an_unsupported_task_is_blocked(api: Any) -> None:
    response = api.post(
        "/api/models",
        json={"name": "asr", "adapter": "huggingface", "entrypoint": "automatic-speech-recognition",
              "artifact": {"uri": "hf://openai/whisper-tiny"}},
    )
    assert response.status_code == 400
    assert "not supported yet" in response.json()["error"]["message"]


def test_non_hf_registrations_are_not_affected_by_hf_rules(api: Any) -> None:
    response = api.post(
        "/api/models",
        json={"name": "pickle-model", "artifact": {"uri": "file:///content/m.pkl"}},
    )
    assert response.status_code == 201
    assert response.json()["adapter"] == "sklearn"


# --------------------------------------------------------------------------- #
# Adapter: the pipeline receives the full owner/name repo (spec R6)
# --------------------------------------------------------------------------- #
def _fake_transformers(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stand in for transformers.pipeline and record what the adapter asked for."""
    import sys
    import types

    calls: list[dict[str, Any]] = []

    def pipeline(task: str, model: str, **kwargs: Any) -> Any:
        calls.append({"task": task, "model": model, **kwargs})
        return lambda data, **_: [{"label": "POSITIVE", "score": 0.99}]

    fake = types.ModuleType("transformers")
    fake.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return calls


@pytest.mark.parametrize(
    ("uri", "expected_repo"),
    [
        ("hf://distilgpt2", "distilgpt2"),
        ("hf://Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4"),
    ],
)
def test_adapter_loads_the_full_repo_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, uri: str, expected_repo: str
) -> None:
    from colabdeploy.adapters.base import ModelMeta
    from colabdeploy.adapters.huggingface import HuggingFaceAdapter

    calls = _fake_transformers(monkeypatch)
    meta = ModelMeta(
        model_id="mdl_test", model_version=1, name="m",
        artifact=ArtifactSpec(uri=uri), entrypoint="text-generation",
    )
    adapter = HuggingFaceAdapter()
    adapter.load(tmp_path, meta)
    assert calls[0]["model"] == expected_repo
    assert calls[0]["task"] == "text-generation"
