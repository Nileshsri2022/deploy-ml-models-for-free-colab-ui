"""Canonical serialisation helpers.

Cache keys must be *stable* across processes, restarts and Python versions, so we
never hash `repr()` or a dict's insertion order. `canonical_json` sorts keys,
normalises floats and separators; `stable_hash` pins the digest.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

# Sentinel used for floats that JSON cannot represent faithfully.
_FLOAT_NORM = "__float__"


def _normalise(value: Any) -> Any:
    """Recursively convert a payload into a form with a single canonical spelling."""
    if isinstance(value, dict):
        return {str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {_FLOAT_NORM: "nan"}
        if math.isinf(value):
            return {_FLOAT_NORM: "inf" if value > 0 else "-inf"}
        # repr() round-trips exactly for floats and avoids 0.1 -> 0.10000000000000001 noise.
        return {_FLOAT_NORM: repr(value)}
    if isinstance(value, int):
        return value
    if isinstance(value, (bytes, bytearray)):
        return {_FLOAT_NORM: "b64", "v": hashlib.sha256(bytes(value)).hexdigest()}
    # Anything exotic: hash its repr rather than exploding.
    return {_FLOAT_NORM: "repr", "v": repr(value)}


def canonical_json(payload: Any) -> str:
    """Serialise `payload` to a deterministic JSON string."""
    return json.dumps(_normalise(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(*parts: Any) -> str:
    """SHA-256 over the canonical form of each part, joined by ':'."""
    h = hashlib.sha256()
    for i, part in enumerate(parts):
        if i:
            h.update(b":")
        if isinstance(part, (bytes, bytearray)):
            h.update(bytes(part))
        elif isinstance(part, str):
            h.update(part.encode("utf-8"))
        else:
            h.update(canonical_json(part).encode("utf-8"))
    return h.hexdigest()


def cache_key(endpoint_id: str, model_version: int, payload: Any) -> str:
    """The single cache-key formula used by every tier (see architecture.md §8)."""
    return stable_hash(f"{endpoint_id}:{model_version}", payload)


def short_id(prefix: str, n_bytes: int = 9) -> str:
    """Human-friendly, URL-safe id such as ``mdl_3f9a1c2b4d5e6f70``."""
    import secrets

    return f"{prefix}_{secrets.token_hex(n_bytes)}"


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
