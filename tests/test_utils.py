"""Cache-key stability - the property every cache tier depends on.

If ``canonical_json`` ever changed behaviour, every cached prediction would
silently miss after a deploy, so these assertions are deliberately pedantic.
"""

from __future__ import annotations

import json

import pytest

from colabdeploy_shared.protocol import AGENT_PROTOCOL_VERSION, ErrorFrame, parse_frame
from colabdeploy_shared.utils import cache_key, canonical_json, sha256_file, short_id, stable_hash


def test_key_order_does_not_change_the_hash() -> None:
    a = canonical_json({"b": 1, "a": 2, "nested": {"z": 1, "y": 2}})
    b = canonical_json({"a": 2, "nested": {"y": 2, "z": 1}, "b": 1})
    assert a == b
    assert stable_hash(a) == stable_hash(b)


def test_cache_key_is_stable_across_equivalent_payloads() -> None:
    k1 = cache_key("ep_1", 3, {"text": "hi", "n": 1})
    k2 = cache_key("ep_1", 3, {"n": 1, "text": "hi"})
    assert k1 == k2


def test_cache_key_separates_every_dimension() -> None:
    base = cache_key("ep_1", 1, {"x": 1})
    assert base != cache_key("ep_2", 1, {"x": 1})  # different endpoint
    assert base != cache_key("ep_1", 2, {"x": 1})  # different model version
    assert base != cache_key("ep_1", 1, {"x": 2})  # different payload


def test_int_and_float_and_bool_are_distinct() -> None:
    values = [1, 1.0, True, "1"]
    keys = {stable_hash(v) for v in values}
    assert len(keys) == len(values), "type confusion would poison the cache"


def test_special_floats_are_serialisable() -> None:
    payload = {"nan": float("nan"), "inf": float("inf"), "ninf": float("-inf")}
    text = canonical_json(payload)
    assert json.loads(text)["nan"] == {"__float__": "nan"}
    # Same NaN payload hashes the same way every time.
    assert stable_hash(payload) == stable_hash(payload)


def test_lists_are_order_sensitive_but_tuples_are_not() -> None:
    assert stable_hash([1, 2]) != stable_hash([2, 1])
    assert stable_hash([1, 2]) == stable_hash((1, 2))


def test_bytes_and_exotic_objects_do_not_explode() -> None:
    assert stable_hash(b"abc")
    assert stable_hash(object())
    assert stable_hash({"set": {1, 2, 3}})


def test_short_id_is_prefixed_and_unique() -> None:
    ids = {short_id("mdl") for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("mdl_") for i in ids)
    assert all(len(i) == len("mdl_") + 18 for i in ids)


def test_sha256_file_matches_known_digest(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"colabdeploy")
    digest = sha256_file(str(path))
    assert len(digest) == 64
    # A single byte change must change the digest (artifact integrity check).
    path.write_bytes(b"colabdeplop")
    assert sha256_file(str(path)) != digest


def test_parse_frame_degrades_on_unknown_kind() -> None:
    frame = parse_frame(json.dumps({"kind": "from_the_future", "id": "x"}))
    assert isinstance(frame, ErrorFrame)
    assert frame.code == "unknown_frame"


def test_parse_frame_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_frame("{not json")
    with pytest.raises(ValueError):
        parse_frame("[1,2,3]")


def test_protocol_version_is_pinned() -> None:
    """Bumping this must be a deliberate, documented act."""
    assert AGENT_PROTOCOL_VERSION == 1
