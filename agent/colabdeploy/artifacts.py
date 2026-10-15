"""Agent-side artifact resolution with a durable on-disk cache.

The single biggest cost of a Colab restart is re-downloading weights. This module
makes that rare:

* artifacts are cached under ``{workdir}/artifacts`` keyed by their **checksum**
  (or a hash of the URI when no checksum is known), so a session *resume* or a
  redeploy of the same version loads from local disk in milliseconds;
* an SQLite index records what we have, so we can answer "is it already here?"
  without stat-ing the world;
* supported schemes cover every free storage option a Colab user has: local path,
  HTTP(S), the gateway's own artifact store (UI uploads), Google Drive via
  ``gdown``, and HuggingFace Hub.

``hf://<repo>`` with no filename is special: the transformers pipeline does its
own download, so we hand back the repo id and let the adapter deal with it.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from colabdeploy_shared.schemas import ArtifactSpec

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


class ArtifactError(RuntimeError):
    """Artifact could not be obtained; the message is safe to surface in the UI."""


@dataclass
class ResolvedArtifact:
    path: Path
    uri: str
    checksum: str | None
    size_bytes: int
    from_cache: bool
    source: str  # local | http | store | gdrive | hf | inline
    elapsed_ms: float = 0.0

    @property
    def is_dir(self) -> bool:
        return self.path.is_dir()


class ArtifactCache:
    """SQLite index + blob directory for downloaded artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.root / "index.sqlite3"), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                cache_key TEXT PRIMARY KEY,
                uri TEXT NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                hits INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------ #
    @staticmethod
    def key_for(artifact: ArtifactSpec) -> str:
        """Checksum when known (content-addressed), else a hash of the URI."""
        if artifact.checksum:
            return f"sha256-{artifact.checksum[:32]}"
        digest = hashlib.sha256(artifact.uri.encode("utf-8")).hexdigest()
        return f"uri-{digest[:32]}"

    def lookup(self, key: str) -> Path | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT path FROM artifacts WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        path = Path(row[0])
        if not path.exists():
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM artifacts WHERE cache_key = ?", (key,))
            return None
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE artifacts SET hits = hits + 1, last_used = ? WHERE cache_key = ?",
                (time.time(), key),
            )
        return path

    def record(
        self, key: str, artifact: ArtifactSpec, path: Path, checksum: str | None, size: int
    ) -> None:
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO artifacts (cache_key, uri, path, checksum, size_bytes, hits, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    path=excluded.path, checksum=excluded.checksum,
                    size_bytes=excluded.size_bytes, last_used=excluded.last_used
                """,
                (key, artifact.uri, str(path), checksum, size, now, now),
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes),0), COALESCE(SUM(hits),0) FROM artifacts"
            ).fetchone()
        return {"entries": int(row[0]), "bytes": int(row[1]), "hits": int(row[2]), "root": str(self.root)}

    def prune(self, keep_bytes: int = 8 * 1024 * 1024 * 1024) -> int:
        """LRU-prune the blob dir so a long-lived session cannot fill the disk."""
        with self._lock:
            total = self._conn.execute("SELECT COALESCE(SUM(size_bytes),0) FROM artifacts").fetchone()[0]
            if int(total) <= keep_bytes:
                return 0
            rows = self._conn.execute(
                "SELECT cache_key, path, size_bytes FROM artifacts ORDER BY last_used ASC"
            ).fetchall()
            removed = 0
            for key, path, size in rows:
                if int(total) <= keep_bytes:
                    break
                blob = Path(path)
                if blob.is_dir():
                    shutil.rmtree(blob, ignore_errors=True)
                elif blob.exists():
                    blob.unlink(missing_ok=True)
                self._conn.execute("DELETE FROM artifacts WHERE cache_key = ?", (key,))
                total = int(total) - int(size)
                removed += 1
            self._conn.commit()
            return removed

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()


#: Schemes the resolver understands. Kept explicit because ``urlparse`` only
#: reports a scheme when it is followed by ``//``, which would misread the
#: documented ``inline:<json>`` spelling as a filename.
SUPPORTED_SCHEMES = ("file", "http", "https", "store", "gdrive", "hf", "inline")


def uri_scheme(uri: str) -> str:
    r"""Return the scheme of an artifact URI, or ``""`` for a bare local path.

    Three cases matter:

    * ``https://…``, ``inline:{…}``      -> the scheme, whether or not ``//`` follows
    * ``/content/m.pkl``, ``m.pkl``      -> ``""`` (a local path)
    * ``s3://bucket/key``                -> ``"s3"``, so ``resolve`` can refuse it
      with a clear message instead of hunting for a file named ``s3://…``

    A Windows-style ``C:\models\m.pkl`` stays a local path: ``c`` is not a
    supported scheme and the remainder does not start with ``//``.
    """
    head, separator, rest = (uri or "").partition(":")
    if not separator:
        return ""
    head = head.strip().lower()
    if head in SUPPORTED_SCHEMES:
        return head
    return head if rest.startswith("//") else ""


class ArtifactResolver:
    """Turn an :class:`ArtifactSpec` into bytes on local disk."""

    def __init__(
        self,
        cache: ArtifactCache,
        *,
        gateway_url: str = "",
        runtime_token: str = "",
        timeout_s: float = 300.0,
        log: Any = None,
    ) -> None:
        self.cache = cache
        self.gateway_url = gateway_url.rstrip("/")
        self.runtime_token = runtime_token
        self.timeout_s = timeout_s
        self._log = log or (lambda message, **kw: None)

    # ------------------------------------------------------------------ #
    def resolve(self, artifact: ArtifactSpec, *, allow_network: bool = True) -> ResolvedArtifact:
        started = time.perf_counter()
        uri = artifact.uri.strip()
        scheme = uri_scheme(uri)

        # HuggingFace repo without a filename: the pipeline handles the download.
        if uri.startswith("hf://") and "/" not in uri.split("://", 1)[1]:
            repo = uri.split("://", 1)[1]
            placeholder = self.cache.blobs / f"hf-{hashlib.sha256(repo.encode()).hexdigest()[:16]}"
            placeholder.mkdir(parents=True, exist_ok=True)
            return ResolvedArtifact(
                path=placeholder, uri=uri, checksum=None, size_bytes=0,
                from_cache=True, source="hf", elapsed_ms=_ms(started),
            )

        key = self.cache.key_for(artifact)
        cached = self.cache.lookup(key)
        if cached is not None:
            self._log(f"artifact cache hit: {cached.name}")
            return ResolvedArtifact(
                path=cached, uri=uri, checksum=artifact.checksum,
                size_bytes=_size(cached), from_cache=True, source=_source_of(uri),
                elapsed_ms=_ms(started),
            )

        target = self.cache.blobs / _safe_filename(key, artifact)
        if not allow_network and scheme in ("http", "https", "store", "gdrive"):
            raise ArtifactError(
                f"artifact '{uri}' is not cached locally and network fetch is disabled"
            )

        if not uri or scheme == "file" or (not scheme and not uri.startswith(("hf://",))):
            self._fetch_local(uri, target)
        elif scheme in ("http", "https"):
            self._fetch_http(uri, target)
        elif scheme == "store":
            self._fetch_store(uri.split("://", 1)[1], target)
        elif scheme == "gdrive":
            self._fetch_gdrive(uri.split("://", 1)[1], target)
        elif scheme == "hf":
            self._fetch_hf(uri, target)
        elif scheme == "inline":
            # ``inline:<json>`` (documented) and the data-URI spelling
            # ``inline:,<json>`` both mean "the bytes are already in the URI".
            body = uri.partition(":")[2]
            if body.startswith(","):
                body = body[1:]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        else:
            raise ArtifactError(f"unsupported artifact scheme '{scheme or 'file'}' in uri '{uri}'")

        checksum = _sha256(target) if target.is_file() else None
        if artifact.checksum and checksum and artifact.checksum != checksum:
            target.unlink(missing_ok=True)
            raise ArtifactError(
                f"checksum mismatch for {uri}: expected {artifact.checksum[:16]}…, got {checksum[:16]}…"
            )
        self.cache.record(key, artifact, target, checksum, _size(target))
        self._log(f"artifact ready: {target.name} ({_size(target) / 1e6:.1f} MB)")
        return ResolvedArtifact(
            path=target, uri=uri, checksum=checksum or artifact.checksum,
            size_bytes=_size(target), from_cache=False, source=_source_of(uri),
            elapsed_ms=_ms(started),
        )

    # ------------------------------------------------------------------ #
    def _fetch_local(self, uri: str, target: Path) -> None:
        path = Path(uri.split("://", 1)[1] if uri.startswith("file://") else uri).expanduser()
        # Colab mounts Drive at /content/drive - a very common deploy source.
        if not path.exists():
            for candidate in (Path("/content") / path.name, Path.cwd() / path.name):
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise ArtifactError(f"local artifact not found: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(path, target, symlinks=True)
        else:
            shutil.copyfile(path, target)

    def _fetch_http(self, url: str, target: Path) -> None:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ArtifactError("httpx is required to download artifacts") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with (
                httpx.Client(follow_redirects=True, timeout=self.timeout_s) as client,
                client.stream("GET", url) as response,
            ):
                if response.status_code >= 400:
                    raise ArtifactError(f"download failed with HTTP {response.status_code}: {url}")
                with tmp.open("wb") as fh:
                    for chunk in response.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
            os.replace(tmp, target)
        except ArtifactError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(f"download failed: {exc}") from exc

    def _fetch_store(self, artifact_id: str, target: Path) -> None:
        """Pull a UI-uploaded artifact back down from the control plane."""
        if not self.gateway_url:
            raise ArtifactError("gateway_url is not configured; cannot fetch store:// artifacts")
        if not self.runtime_token:
            raise ArtifactError("runtime token is missing; cannot fetch store:// artifacts")
        url = f"{self.gateway_url}/internal/artifacts/{artifact_id}"
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ArtifactError("httpx is required to fetch store artifacts") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            with (
                httpx.Client(timeout=self.timeout_s, follow_redirects=True) as client,
                client.stream("GET", url, headers={"X-Runtime-Token": self.runtime_token}) as response,
            ):
                    if response.status_code >= 400:
                        detail = response.read()[:200].decode("utf-8", "replace")
                        raise ArtifactError(
                            f"gateway returned HTTP {response.status_code} for {artifact_id}: {detail}"
                        )
                    with tmp.open("wb") as fh:
                        for chunk in response.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
            os.replace(tmp, target)
        except ArtifactError:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise ArtifactError(f"store fetch failed: {exc}") from exc

    def _fetch_gdrive(self, file_id: str, target: Path) -> None:
        # A Drive file already mounted in the VM beats a download.
        mounted = Path(f"/content/drive/MyDrive/{file_id}")
        if mounted.exists():
            self._fetch_local(str(mounted), target)
            return
        try:
            import gdown  # type: ignore
        except ImportError as exc:
            raise ArtifactError(
                "gdown is required for gdrive:// artifacts; run !pip install gdown, "
                "or mount Drive and use a file:// path"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(id=file_id.split("/")[-1], output=str(target), quiet=True, fuzzy=True)
        except Exception as exc:
            raise ArtifactError(f"Google Drive download failed: {exc}") from exc
        if not target.exists() or target.stat().st_size == 0:
            raise ArtifactError(
                f"Google Drive returned nothing for id '{file_id}'",
            )

    def _fetch_hf(self, uri: str, target: Path) -> None:
        rest = uri.split("://", 1)[1]
        repo, _, filename = rest.partition("/")
        if not filename:
            raise ArtifactError(
                f"cannot resolve '{uri}' - use hf://<repo>/<filename> or let the pipeline download it"
            )
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:
            raise ArtifactError(
                "huggingface_hub is required for hf:// artifacts; run !pip install huggingface_hub"
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            local = hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:
            raise ArtifactError(f"HuggingFace download failed for {repo}/{filename}: {exc}") from exc
        shutil.copyfile(local, target)


# --------------------------------------------------------------------------- #
def _safe_filename(key: str, artifact: ArtifactSpec) -> str:
    name = artifact.filename or Path(urlparse(artifact.uri).path).name or "artifact.bin"
    name = _SAFE_NAME.sub("_", name)[:80]
    suffix = Path(name).suffix or ".bin"
    stem = Path(name).stem
    if not stem.endswith(suffix):
        return f"{key}-{stem}{suffix}"
    return f"{key}-{name}"


def _size(path: Path) -> int:
    try:
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return path.stat().st_size
    except OSError:  # pragma: no cover
        return 0


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_of(uri: str) -> str:
    scheme = uri_scheme(uri) or "file"
    return {"file": "local", "http": "http", "https": "http", "store": "store",
            "gdrive": "gdrive", "hf": "hf", "inline": "inline"}.get(scheme, scheme or "local")


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
