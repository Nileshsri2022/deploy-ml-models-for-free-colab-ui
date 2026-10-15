"""Artifact resolution and the on-VM blob cache.

The agent downloads model bytes itself, so this layer is where a flaky network,
a huge checkpoint and a bad checksum all have to behave predictably:

* content-addressed keys, so the same bytes are never fetched twice;
* ``.part`` + ``os.replace``, so a half-written blob is never served;
* a checksum mismatch is a hard failure that also removes the bad file;
* LRU pruning keeps a long-lived Colab session from filling the disk.

No network is touched: ``http(s)://`` cases assert the guard rails instead.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from colabdeploy.artifacts import (
    ArtifactCache,
    ArtifactError,
    ArtifactResolver,
    uri_scheme,
)
from colabdeploy_shared.schemas import ArtifactSpec


@pytest.fixture()
def cache(tmp_path: Path) -> ArtifactCache:
    store = ArtifactCache(tmp_path / "artifact-cache")
    yield store
    store.close()


@pytest.fixture()
def resolver(cache: ArtifactCache) -> ArtifactResolver:
    return ArtifactResolver(cache, gateway_url="http://gateway.test", runtime_token="CDP_RT_test")


# --------------------------------------------------------------------------- #
# Scheme parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("file:///content/model.pkl", "file"),
        ("https://example.com/m.onnx", "https"),
        ("http://example.com/m.onnx", "http"),
        ("gdrive://1AbCdEf", "gdrive"),
        ("hf://org/repo/model.onnx", "hf"),
        ("store://art_0123456789abcdef", "store"),
        # The documented inline spelling has no "//", which is exactly what used
        # to make it look like a filename.
        ('inline:{"latency_ms": 0}', "inline"),
        ("inline:,{\"latency_ms\": 0}", "inline"),
        # Bare paths are local artifacts, whatever they look like.
        ("/content/model.pkl", ""),
        ("model.pkl", ""),
        ("./weights/m.pt", ""),
        ("C:\\models\\m.pkl", ""),
        ("", ""),
        # An unknown scheme is reported as a scheme, not treated as a filename,
        # so resolve() can refuse it with a clear message.
        ("ftp://example.com/m.bin", "ftp"),
        ("s3://bucket/key", "s3"),
    ],
)
def test_uri_scheme(uri: str, expected: str) -> None:
    assert uri_scheme(uri) == expected


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #
def test_key_is_content_addressed_when_a_checksum_is_known(cache: ArtifactCache) -> None:
    spec = ArtifactSpec(uri="https://example.com/a.pkl", checksum="deadbeef" * 8)
    assert cache.key_for(spec) == f"sha256-{('deadbeef' * 8)[:32]}"


def test_key_falls_back_to_the_uri(cache: ArtifactCache) -> None:
    spec = ArtifactSpec(uri="gdrive://1AbC")
    expected = f"uri-{hashlib.sha256(b'gdrive://1AbC').hexdigest()[:32]}"
    assert cache.key_for(spec) == expected


def test_identical_bytes_under_different_uris_share_a_slot(cache: ArtifactCache) -> None:
    """The point of content addressing: a Drive copy and an HF mirror collide on purpose."""
    digest = hashlib.sha256(b"weights").hexdigest()
    a = ArtifactSpec(uri="gdrive://one", checksum=digest)
    b = ArtifactSpec(uri="hf://org/repo/weights", checksum=digest)
    assert cache.key_for(a) == cache.key_for(b)


def test_lookup_miss_returns_none(cache: ArtifactCache) -> None:
    assert cache.lookup("uri-does-not-exist") is None


def test_record_then_lookup_counts_hits(cache: ArtifactCache, tmp_path: Path) -> None:
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"weights")
    spec = ArtifactSpec(uri="file:///blob.bin")
    key = cache.key_for(spec)
    cache.record(key, spec, blob, hashlib.sha256(b"weights").hexdigest(), blob.stat().st_size)

    assert cache.lookup(key) == blob
    assert cache.lookup(key) == blob
    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["bytes"] == len(b"weights")
    # record() counts as the first hit; two lookups add two more.
    assert stats["hits"] == 3


def test_lookup_self_heals_when_the_blob_is_deleted(cache: ArtifactCache, tmp_path: Path) -> None:
    """A stale index row must not be served as a cache hit."""
    blob = tmp_path / "gone.bin"
    blob.write_bytes(b"x")
    spec = ArtifactSpec(uri="file:///gone.bin")
    key = cache.key_for(spec)
    cache.record(key, spec, blob, None, 1)

    blob.unlink()
    assert cache.lookup(key) is None
    assert cache.stats()["entries"] == 0


def test_prune_evicts_least_recently_used(cache: ArtifactCache, tmp_path: Path) -> None:
    recorded: list[tuple[str, Path]] = []
    for index in range(3):
        blob = tmp_path / f"m{index}.bin"
        blob.write_bytes(b"x" * 100)
        spec = ArtifactSpec(uri=f"file:///m{index}.bin")
        key = cache.key_for(spec)
        cache.record(key, spec, blob, None, 100)
        recorded.append((key, blob))
        time.sleep(0.01)

    # Touch the oldest entry so it becomes the most recently used one.
    cache.lookup(recorded[0][0])

    removed = cache.prune(keep_bytes=150)
    assert removed == 2
    assert cache.stats()["bytes"] == 100
    assert cache.lookup(recorded[0][0]) == recorded[0][1]
    assert cache.lookup(recorded[1][0]) is None
    assert not recorded[1][1].exists()  # the blob goes with its index row


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def test_resolve_inline_json(resolver: ArtifactResolver) -> None:
    payload = {"latency_ms": 5, "greeting": "hello"}
    resolved = resolver.resolve(ArtifactSpec(uri=f"inline:{json.dumps(payload)}", filename="m.json"))

    assert resolved.from_cache is False
    assert resolved.source == "inline"
    assert json.loads(resolved.path.read_text()) == payload


def test_resolve_inline_data_uri_spelling(resolver: ArtifactResolver) -> None:
    resolved = resolver.resolve(ArtifactSpec(uri='inline:,{"ok": true}', filename="m.json"))
    assert json.loads(resolved.path.read_text()) == {"ok": True}


def test_resolve_inline_python_source(resolver: ArtifactResolver) -> None:
    """The python_function adapter ships its code as an inline artifact.

    ``ArtifactSpec`` strips surrounding whitespace from ``uri``, so an inline
    payload cannot rely on a trailing newline - the compiled source is identical
    either way.
    """
    source = "def predict(payload, model=None, meta=None):\n    return {'n': len(payload)}"
    resolved = resolver.resolve(ArtifactSpec(uri=f"inline:{source}", filename="model.py"))
    assert resolved.path.read_text() == source


def test_second_resolve_is_a_cache_hit(resolver: ArtifactResolver) -> None:
    spec = ArtifactSpec(uri='inline:{"a": 1}', filename="m.json")
    first = resolver.resolve(spec)
    second = resolver.resolve(spec)

    assert first.from_cache is False
    assert second.from_cache is True
    assert second.path == first.path
    assert resolver.cache.stats()["entries"] == 1


def test_resolve_local_file(resolver: ArtifactResolver, tmp_path: Path) -> None:
    source = tmp_path / "model.pkl"
    source.write_bytes(b"pickled-weights")

    resolved = resolver.resolve(ArtifactSpec(uri=f"file://{source}", filename="model.pkl"))
    assert resolved.path.read_bytes() == b"pickled-weights"
    assert resolved.size_bytes == len(b"pickled-weights")
    assert resolved.source == "local"
    assert resolved.path != source  # a copy inside the cache, not the original


def test_resolve_bare_path(resolver: ArtifactResolver, tmp_path: Path) -> None:
    source = tmp_path / "bare.bin"
    source.write_bytes(b"bare")
    resolved = resolver.resolve(ArtifactSpec(uri=str(source), filename="bare.bin"))
    assert resolved.path.read_bytes() == b"bare"


def test_missing_local_artifact_fails_loudly(resolver: ArtifactResolver, tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "absent.pkl"
    with pytest.raises(ArtifactError, match="local artifact not found"):
        resolver.resolve(ArtifactSpec(uri=str(missing)))


def test_checksum_mismatch_deletes_the_blob(resolver: ArtifactResolver) -> None:
    body = json.dumps({"latency_ms": 0})
    wrong = hashlib.sha256(b"something else entirely").hexdigest()
    spec = ArtifactSpec(uri=f"inline:{body}", checksum=wrong, filename="m.json")

    with pytest.raises(ArtifactError, match="checksum mismatch"):
        resolver.resolve(spec)

    # The bad bytes must not linger where a later lookup could find them.
    assert resolver.cache.lookup(resolver.cache.key_for(spec)) is None
    assert resolver.cache.stats()["entries"] == 0


def test_matching_checksum_records_the_digest(resolver: ArtifactResolver) -> None:
    body = json.dumps({"ok": 1})
    digest = hashlib.sha256(body.encode()).hexdigest()
    resolved = resolver.resolve(ArtifactSpec(uri=f"inline:{body}", checksum=digest))
    assert resolved.checksum == digest


def test_network_schemes_are_refused_when_network_is_disabled(
    resolver: ArtifactResolver,
) -> None:
    for uri in ("https://example.com/m.onnx", "gdrive://1AbC", "store://art_1234567890abcdef"):
        with pytest.raises(ArtifactError, match="network fetch is disabled"):
            resolver.resolve(ArtifactSpec(uri=uri), allow_network=False)


def test_store_scheme_needs_gateway_credentials(cache: ArtifactCache) -> None:
    """A store:// artifact is served by the gateway and authenticated by runtime token."""
    anonymous = ArtifactResolver(cache)
    with pytest.raises(ArtifactError, match="gateway_url is not configured"):
        anonymous.resolve(ArtifactSpec(uri="store://art_1234567890abcdef"))

    no_token = ArtifactResolver(cache, gateway_url="http://gateway.test")
    with pytest.raises(ArtifactError, match="runtime token is missing"):
        no_token.resolve(ArtifactSpec(uri="store://art_1234567890abcdef"))


def test_unsupported_scheme_is_reported(resolver: ArtifactResolver) -> None:
    with pytest.raises(ArtifactError, match="unsupported artifact scheme"):
        resolver.resolve(ArtifactSpec(uri="ftp://example.com/m.bin"))


def test_huggingface_repo_without_a_filename_is_a_placeholder(resolver: ArtifactResolver) -> None:
    """Pipeline-style repos (no filename) are downloaded by the adapter itself."""
    resolved = resolver.resolve(ArtifactSpec(uri="hf://org-repo"))
    assert resolved.source == "hf"
    assert resolved.is_dir
    assert resolved.size_bytes == 0


def test_directory_artifacts_are_copied_whole(resolver: ArtifactResolver, tmp_path: Path) -> None:
    source = tmp_path / "pipeline"
    (source / "sub").mkdir(parents=True)
    (source / "model.json").write_text("{}")
    (source / "sub" / "weights.bin").write_bytes(b"abc")

    resolved = resolver.resolve(ArtifactSpec(uri=f"file://{source}"))
    assert resolved.is_dir
    assert (resolved.path / "model.json").exists()
    assert (resolved.path / "sub" / "weights.bin").read_bytes() == b"abc"


def test_elapsed_time_is_reported(resolver: ArtifactResolver) -> None:
    resolved: Any = resolver.resolve(ArtifactSpec(uri='inline:{"a": 1}'))
    assert resolved.elapsed_ms >= 0.0
