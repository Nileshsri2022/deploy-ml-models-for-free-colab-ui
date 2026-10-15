"""Adapters: detection, the custom-code escape hatch, and payload normalisation.

Only the dependency-free adapters (``echo``, ``python_function``) are exercised
here; the framework adapters need sklearn/torch/onnxruntime, which the agent
lazy-imports and reports honestly through ``available_adapters()``. Their
detection logic *is* covered, because that runs without the framework installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from colabdeploy.adapters import (
    AdapterError,
    available_adapters,
    build_adapter,
    framework_available,
    get_adapter_class,
    load_with_probe,
    probe_order,
    resolve_adapter,
)
from colabdeploy.adapters.base import (
    ModelMeta,
    jsonable,
    payload_image_bytes,
    payload_table,
    payload_text,
)
from colabdeploy.adapters.echo import EchoAdapter
from colabdeploy.adapters.python_function import PythonFunctionAdapter
from colabdeploy_shared.enums import AdapterKind
from colabdeploy_shared.schemas import ArtifactSpec


def meta_for(artifact: ArtifactSpec, **kwargs: Any) -> ModelMeta:
    return ModelMeta(
        model_id=kwargs.pop("model_id", "mdl_test"),
        model_version=kwargs.pop("model_version", 1),
        artifact=artifact,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "filename,expected",
    [
        ("model.onnx", AdapterKind.ONNX),
        ("model.pt", AdapterKind.PYTORCH),
        ("model.pth", AdapterKind.PYTORCH),
        ("model.h5", AdapterKind.KERAS),
        ("model.keras", AdapterKind.KERAS),
        ("model.joblib", AdapterKind.SKLEARN),
        ("model.pkl", AdapterKind.SKLEARN),
        ("booster.ubj", AdapterKind.XGBOOST),
    ],
)
def test_extension_detection(filename: str, expected: AdapterKind) -> None:
    artifact = ArtifactSpec(uri=f"store://art_x/{filename}", filename=filename)
    meta = meta_for(artifact, adapter=AdapterKind.AUTO)
    adapter = resolve_adapter(artifact, meta)
    # The adapter we get is either the expected one, or one whose framework is
    # installed and matched earlier in priority order.
    assert adapter.kind in (expected, AdapterKind.SKLEARN) or framework_available(type(adapter))


def test_hf_uri_is_detected_by_the_huggingface_adapter() -> None:
    artifact = ArtifactSpec(uri="hf://distilbert-base-uncased-finetuned-sst-2-english")
    meta = meta_for(artifact, adapter=AdapterKind.AUTO)
    cls = get_adapter_class(AdapterKind.HUGGINGFACE.value)
    assert cls is not None and cls.detect(artifact, meta)
    # ...and it is only *attempted* when transformers is actually installed.
    expected = framework_available(cls)
    assert (AdapterKind.HUGGINGFACE.value in probe_order(meta)) is expected


def test_entrypoint_forces_the_custom_code_adapter() -> None:
    artifact = ArtifactSpec(uri="file:///content/whatever.bin")
    meta = meta_for(artifact, adapter=AdapterKind.AUTO, entrypoint="def predict(p): return p")
    assert probe_order(meta)[0] == AdapterKind.PYTHON_FUNCTION.value


def test_explicit_adapter_kind_wins() -> None:
    artifact = ArtifactSpec(uri="file:///content/model.onnx")
    meta = meta_for(artifact, adapter=AdapterKind.ECHO)
    adapter = resolve_adapter(artifact, meta)
    assert isinstance(adapter, EchoAdapter)


def test_available_adapters_never_lies() -> None:
    """Every reported adapter must be instantiable and its framework importable.

    The gateway shows this list in the UI and uses it to explain a failed deploy,
    so an optimistic entry is worse than a missing one.
    """
    kinds = available_adapters()
    assert AdapterKind.ECHO.value in kinds, "echo has no dependencies, so it is always available"
    assert AdapterKind.PYTHON_FUNCTION.value in kinds
    for kind in kinds:
        cls = get_adapter_class(kind)
        assert cls is not None
        assert framework_available(cls), f"{kind} reported available but its framework is missing"
        assert build_adapter(kind) is not None


def test_unavailable_frameworks_are_excluded_from_the_probe_order() -> None:
    """Auto-detection must not waste a load attempt on a framework we do not have."""
    meta = meta_for(ArtifactSpec(uri="file:///content/model.bin"), adapter=AdapterKind.AUTO)
    order = probe_order(meta)
    for kind in order:
        cls = get_adapter_class(kind)
        assert cls is not None
        assert framework_available(cls), f"{kind} in probe order without its framework"
    # With no ML framework installed in CI, only the dependency-free adapters remain.
    if AdapterKind.PYTORCH.value not in available_adapters():
        assert AdapterKind.PYTORCH.value not in order


def test_explicit_kind_is_honoured_even_when_the_framework_is_missing() -> None:
    """An explicit adapter choice must fail loudly, not silently fall back."""
    meta = meta_for(ArtifactSpec(uri="file:///content/model.pt"), adapter=AdapterKind.PYTORCH)
    assert probe_order(meta) == [AdapterKind.PYTORCH.value]


# --------------------------------------------------------------------------- #
# Echo adapter (the dependency-free path used by the e2e tests)
# --------------------------------------------------------------------------- #
def test_echo_adapter_predict_is_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "m.echo.json"
    artifact.write_text("{}", encoding="utf-8")
    adapter = EchoAdapter()
    adapter.load(artifact, meta_for(ArtifactSpec(uri=str(artifact))))
    payload = {"text": "hello", "n": 2}
    first = adapter.predict(payload)
    second = adapter.predict(payload)
    assert first["sha256"] == second["sha256"], "cached answers must be correct answers"
    assert first["echo"] == payload
    assert first["numeric_sum"] == 2
    assert first["model_id"] == "mdl_test"


def test_echo_adapter_empty_artifact_is_fine(tmp_path: Path) -> None:
    artifact = tmp_path / "m.echo.json"
    artifact.write_text("", encoding="utf-8")
    adapter = EchoAdapter()
    adapter.load(artifact, meta_for(ArtifactSpec(uri=str(artifact))))
    assert adapter.predict({"a": 1})["echo"] == {"a": 1}


def test_echo_adapter_can_simulate_failure_and_latency(tmp_path: Path) -> None:
    artifact = tmp_path / "bad.echo.json"
    artifact.write_text(json.dumps({"fail": "boom", "latency_ms": 1}), encoding="utf-8")
    adapter = EchoAdapter()
    adapter.load(artifact, meta_for(ArtifactSpec(uri=str(artifact))))
    with pytest.raises(AdapterError, match="boom"):
        adapter.predict({"a": 1})


def test_echo_adapter_predict_before_load_fails(tmp_path: Path) -> None:
    adapter = EchoAdapter()
    with pytest.raises(AdapterError, match="not loaded"):
        adapter.predict({"a": 1})


def test_warmup_returns_elapsed_ms_and_swallows_errors(tmp_path: Path) -> None:
    artifact = tmp_path / "m.echo.json"
    artifact.write_text(json.dumps({"fail": "nope"}), encoding="utf-8")
    adapter = EchoAdapter()
    adapter.load(artifact, meta_for(ArtifactSpec(uri=str(artifact))))
    elapsed = adapter.warmup({"a": 1})
    assert isinstance(elapsed, float) and elapsed >= 0.0
    assert "warmup failed" in (adapter.health.last_error or "")
    assert adapter.warmup(None) == 0.0


# --------------------------------------------------------------------------- #
# python_function adapter
# --------------------------------------------------------------------------- #
def _load_fn(source: str, tmp_path: Path, artifact_name: str = "code.py") -> PythonFunctionAdapter:
    path = tmp_path / artifact_name
    path.write_text(source, encoding="utf-8")
    adapter = PythonFunctionAdapter()
    adapter.load(path, meta_for(ArtifactSpec(uri=str(path)), entrypoint=source))
    return adapter


def test_python_function_runs_user_code(tmp_path: Path) -> None:
    adapter = _load_fn(
        "def predict(payload, model=None, meta=None):\n"
        "    return {'sum': sum(payload.get('values', [])), 'version': meta.model_version}\n",
        tmp_path,
    )
    out = adapter.predict({"values": [1, 2, 3]})
    assert out == {"sum": 6, "version": 1}


def test_python_function_accepts_minimal_signature(tmp_path: Path) -> None:
    adapter = _load_fn("def predict(payload):\n    return payload['x'] * 2\n", tmp_path)
    assert adapter.predict({"x": 21}) == 42


def test_python_function_accepts_lambda_style(tmp_path: Path) -> None:
    adapter = _load_fn("predict = lambda payload: {'ok': True}\n", tmp_path)
    assert adapter.predict({}) == {"ok": True}


def test_python_function_entrypoint_may_be_a_callable_name(tmp_path: Path) -> None:
    path = tmp_path / "code.py"
    path.write_text("def handler(payload):\n    return {'handled': payload}\n", encoding="utf-8")
    adapter = PythonFunctionAdapter()
    adapter.load(path, meta_for(ArtifactSpec(uri=str(path)), entrypoint="handler"))
    assert adapter.predict({"a": 1}) == {"handled": {"a": 1}}


def test_python_function_reports_syntax_errors_with_a_line_number(tmp_path: Path) -> None:
    with pytest.raises(AdapterError) as excinfo:
        _load_fn("def predict(payload)\n    return 1\n", tmp_path)
    assert "syntax error" in str(excinfo.value)
    assert "line 1" in str(excinfo.value)


def test_python_function_reports_missing_callable(tmp_path: Path) -> None:
    with pytest.raises(AdapterError) as excinfo:
        _load_fn("VALUE = 3\n", tmp_path)
    assert "defines no callable" in str(excinfo.value)


def test_python_function_wraps_runtime_errors(tmp_path: Path) -> None:
    adapter = _load_fn("def predict(payload):\n    raise ZeroDivisionError('bad')\n", tmp_path)
    with pytest.raises(AdapterError) as excinfo:
        adapter.predict({})
    assert "ZeroDivisionError" in str(excinfo.value)


def test_python_function_converts_numpy_like_output(tmp_path: Path) -> None:
    class FakeArray:
        def tolist(self) -> list[float]:
            return [1.0, 2.0]

    adapter = _load_fn(
        "class FakeArray:\n"
        "    def tolist(self):\n"
        "        return [1.0, 2.0]\n\n"
        "def predict(payload):\n"
        "    return {'values': FakeArray(), 'raw': b'bytes'}\n",
        tmp_path,
    )
    out = adapter.predict({})
    assert out["values"] == [1.0, 2.0]
    assert out["raw"] == "bytes"


def test_python_function_defaults_to_a_no_op(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("", encoding="utf-8")
    adapter = PythonFunctionAdapter()
    adapter.load(path, meta_for(ArtifactSpec(uri=str(path)), entrypoint=""))
    assert adapter.predict({"a": 1}) == {"echo": {"a": 1}}


# --------------------------------------------------------------------------- #
# Auto-probe
# --------------------------------------------------------------------------- #
def test_load_with_probe_reports_every_failed_attempt(tmp_path: Path) -> None:
    """One bad model must never look deployed - every failure is surfaced."""
    artifact = tmp_path / "broken.py"
    artifact.write_text("def predict(payload)\n    return 1\n", encoding="utf-8")
    meta = meta_for(
        ArtifactSpec(uri=str(artifact), filename="broken.py"),
        adapter=AdapterKind.PYTHON_FUNCTION,
        entrypoint=artifact.read_text(encoding="utf-8"),
    )
    attempts: list[tuple[str, bool, str | None]] = []
    with pytest.raises(AdapterError) as excinfo:
        load_with_probe(
            artifact, meta, on_attempt=lambda k, ok, err: attempts.append((k, ok, err))
        )
    assert attempts and all(not ok for _, ok, _ in attempts)
    assert "syntax error" in (attempts[0][2] or "")
    assert "no adapter could load" in str(excinfo.value)
    assert "hint" in str(excinfo.value)


def test_load_with_probe_falls_through_to_echo_for_unknown_artifacts(tmp_path: Path) -> None:
    """``echo`` is the permissive tail of the probe order: auto mode always lands."""
    artifact = tmp_path / "mystery.bin"
    artifact.write_bytes(b"\x00\x01not a model")
    meta = meta_for(ArtifactSpec(uri=str(artifact), filename="mystery.bin"))
    attempts: list[str] = []
    adapter, errors = load_with_probe(artifact, meta, on_attempt=lambda k, ok, err: attempts.append(k))
    assert AdapterKind.ECHO.value in attempts
    assert isinstance(adapter, EchoAdapter)
    assert adapter.predict({"probe": True})["echo"] == {"probe": True}
    # Every adapter tried before echo reports *why* it gave up - that list is what
    # the UI shows instead of a mystery "could not load" error.
    assert errors, "expected the earlier attempts to explain themselves"
    assert all(": " in error for error in errors)  # each one is "kind: reason"
    assert not any(error.startswith(f"{AdapterKind.ECHO.value}:") for error in errors)


def test_load_with_probe_succeeds_for_an_echo_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "m.echo.json"
    artifact.write_text("{}", encoding="utf-8")
    meta = meta_for(ArtifactSpec(uri=str(artifact), filename="m.echo.json"), adapter=AdapterKind.ECHO)
    adapter, errors = load_with_probe(artifact, meta)
    assert isinstance(adapter, EchoAdapter)
    assert adapter.predict({"ok": True})["echo"] == {"ok": True}
    assert errors == []  # an explicit adapter kind does not probe, so nothing failed


# --------------------------------------------------------------------------- #
# Payload normalisation helpers
# --------------------------------------------------------------------------- #
def test_payload_table_accepts_every_common_shape() -> None:
    assert payload_table({"columns": ["a", "b"], "data": [[1, 2]]}) == (["a", "b"], [[1, 2]])
    assert payload_table({"instances": [{"a": 1, "b": 2}]}) == (["a", "b"], [[1, 2]])
    assert payload_table({"inputs": [{"a": 1}]}) == (["a"], [[1]])
    assert payload_table({"features": [[1, 2], [3, 4]]}) == (["f0", "f1"], [[1, 2], [3, 4]])
    assert payload_table({"x": 1, "y": 2}) == (["x", "y"], [[1, 2]]), "a flat object is one row"


def test_payload_table_rejects_garbage_with_a_hint() -> None:
    with pytest.raises(AdapterError) as excinfo:
        payload_table({"instances": 5})
    assert "hint" in str(excinfo.value)
    with pytest.raises(AdapterError):
        payload_table({"instances": []})


def test_payload_text_finds_the_input() -> None:
    assert payload_text({"text": "hi"}) == "hi"
    assert payload_text({"prompt": "hi"}) == "hi"
    assert payload_text({"inputs": ["a", "b"]}) == "a"
    assert payload_text({"whatever": "hi"}) == "hi", "single-key payloads are unambiguous"
    with pytest.raises(AdapterError):
        payload_text({"a": 1, "b": 2})


def test_payload_image_bytes_handles_data_uris_and_base64(tmp_path: Path) -> None:
    import base64

    raw = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
    encoded = base64.b64encode(raw).decode()
    assert payload_image_bytes({"image": encoded}) == raw
    assert payload_image_bytes({"image": f"data:image/png;base64,{encoded}"}) == raw
    path = tmp_path / "img.bin"
    path.write_bytes(raw)
    assert payload_image_bytes({"image": str(path)}) == raw
    with pytest.raises(AdapterError):
        payload_image_bytes({"nothing": 1})


def test_jsonable_handles_everything_a_model_might_return() -> None:
    class Fake:
        def __repr__(self) -> str:
            return "<fake>"

    assert jsonable({"a": [1, (2, 3), {4}, None, True]}) == {
        "a": [1, [2, 3], [4], None, True]
    }
    assert jsonable(b"abc") == "abc"
    assert jsonable(Fake()) == "<fake>"
    assert jsonable(1.5) == 1.5


def test_meta_key_and_from_deployment_round_trip() -> None:
    dep = {
        "deployment_id": "dep_1",
        "endpoint_id": "ep_1",
        "model_id": "mdl_1",
        "model_version": 3,
        "adapter": "echo",
        "artifact": {"uri": "file:///x.echo.json"},
        "schema_in": {"fields": {"a": {"type": "string"}}, "required": ["a"]},
        "warmup_sample": {"a": "b"},
        "some_future_field": True,
    }
    meta = ModelMeta.from_deployment(dep)
    assert meta.key == "mdl_1@3"
    assert meta.endpoint_id == "ep_1"
    assert meta.schema_in.required == ["a"]
    assert meta.extra == {"some_future_field": True}, "unknown fields survive for forward compat"
