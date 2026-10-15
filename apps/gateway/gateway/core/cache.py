"""The resilience cache (architecture.md §8, Tier 2).

This is not a performance optimisation - it is the substrate that lets a published
endpoint keep answering while the Colab VM is dead. Three behaviours matter:

* **TTL freshness** - a young entry is a ``HIT``.
* **stale-while-revalidate** - an expired-but-not-ancient entry is a ``STALE`` hit.
  When a runtime is offline, ``STALE`` becomes the *primary* answer.
* **pinning + warm-up** - hot keys survive eviction, and on reconnect the gateway
  hands the top-K keys to the agent so the first real request is already warm.

Backends: ``MemoryCache`` (LRU, optionally persisted to disk on shutdown/startup)
and ``SqliteCache`` (durable, survives process restarts without a snapshot). Both
implement :class:`CacheBackend`, so adding Redis later is one class.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colabdeploy_shared.schemas import CacheEntryInfo, CacheStats

CACHE_HIT = "HIT"
CACHE_STALE = "STALE"
CACHE_MISS = "MISS"


@dataclass
class CacheLookup:
    """Result of a cache probe."""

    status: str = CACHE_MISS
    value: Any = None
    key: str | None = None
    age_s: float = 0.0
    stale_for_s: float = 0.0
    created_at: float = 0.0
    endpoint_id: str | None = None
    model_version: int = 0

    @property
    def usable(self) -> bool:
        return self.status in (CACHE_HIT, CACHE_STALE)


@dataclass
class _Entry:
    key: str
    value: Any
    endpoint_id: str
    model_version: int
    created_at: float
    ttl_s: float
    swr_s: float
    pinned: bool = False
    hits: int = 0
    size_bytes: int = 0
    raw: bytes = field(default=b"", repr=False)

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_s

    @property
    def stale_at(self) -> float:
        return self.created_at + self.ttl_s + self.swr_s

    def status(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        if now < self.expires_at:
            return CACHE_HIT
        if now < self.stale_at:
            return CACHE_STALE
        return CACHE_MISS

    def to_info(self, preview: bool = True) -> CacheEntryInfo:
        return CacheEntryInfo(
            key=self.key,
            endpoint_id=self.endpoint_id,
            model_version=self.model_version,
            created_at=self.created_at,  # type: ignore[arg-type]
            expires_at=self.expires_at,  # type: ignore[arg-type]
            stale_at=self.stale_at,  # type: ignore[arg-type]
            pinned=self.pinned,
            hits=self.hits,
            size_bytes=self.size_bytes,
            preview=_preview(self.value) if preview else None,
        )


def _preview(value: Any, limit: int = 400) -> dict[str, Any] | None:
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover
        return None
    if len(text) > limit:
        text = text[:limit] + "…"
    return {"json": text}


def _size_of(value: Any) -> tuple[int, bytes]:
    try:
        raw = json.dumps(value, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError):  # pragma: no cover
        raw = repr(value).encode("utf-8")
    return len(raw), raw


class CacheBackend(ABC):
    """Interface every cache tier implements."""

    name = "abstract"

    @abstractmethod
    def get(self, key: str) -> CacheLookup: ...

    @abstractmethod
    def peek(self, key: str) -> CacheLookup:
        """Like :meth:`get` but does not count a hit or touch LRU order."""

    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        *,
        endpoint_id: str,
        model_version: int,
        ttl_s: float,
        swr_s: float = 0.0,
        pinned: bool = False,
    ) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def pin(self, key: str, pinned: bool = True) -> bool: ...

    @abstractmethod
    def invalidate_endpoint(self, endpoint_id: str) -> int: ...

    @abstractmethod
    def entries(self, limit: int = 100, endpoint_id: str | None = None) -> list[CacheEntryInfo]: ...

    @abstractmethod
    def stats(self) -> CacheStats: ...

    @abstractmethod
    def hot_keys(self, endpoint_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        return None


class MemoryCache(CacheBackend):
    """LRU cache with TTL + stale-while-revalidate, optionally snapshotted to disk.

    Bounded by both entry count and total bytes; pinned entries are exempt from
    eviction (they are the reason an offline endpoint can still answer).
    """

    name = "memory"

    def __init__(
        self,
        max_entries: int = 20000,
        max_bytes: int = 256 * 1024 * 1024,
        persist_path: str | Path | None = None,
        autoload: bool = True,
    ) -> None:
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.persist_path = Path(persist_path) if persist_path else None
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._stale_hits = 0
        self._evictions = 0
        self._bytes = 0
        if autoload and self.persist_path and self.persist_path.exists():
            self.load()

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> CacheLookup:
        with self._lock:
            entry = self._data.get(key)
            now = time.time()
            if entry is None:
                self._misses += 1
                return CacheLookup(status=CACHE_MISS, key=key)
            status = entry.status(now)
            if status == CACHE_MISS:
                # Past even the stale window: drop it and count a miss.
                self._remove(entry)
                self._misses += 1
                return CacheLookup(status=CACHE_MISS, key=key)
            entry.hits += 1
            self._data.move_to_end(key)
            if status == CACHE_HIT:
                self._hits += 1
            else:
                self._stale_hits += 1
            return CacheLookup(
                status=status,
                value=entry.value,
                key=key,
                age_s=max(0.0, now - entry.created_at),
                stale_for_s=max(0.0, now - entry.expires_at) if status == CACHE_STALE else 0.0,
                created_at=entry.created_at,
                endpoint_id=entry.endpoint_id,
                model_version=entry.model_version,
            )

    def peek(self, key: str) -> CacheLookup:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return CacheLookup(status=CACHE_MISS, key=key)
            now = time.time()
            status = entry.status(now)
            return CacheLookup(
                status=status,
                value=entry.value if status != CACHE_MISS else None,
                key=key,
                age_s=max(0.0, now - entry.created_at),
                created_at=entry.created_at,
                endpoint_id=entry.endpoint_id,
                model_version=entry.model_version,
            )

    def set(
        self,
        key: str,
        value: Any,
        *,
        endpoint_id: str,
        model_version: int,
        ttl_s: float,
        swr_s: float = 0.0,
        pinned: bool = False,
    ) -> None:
        size, raw = _size_of(value)
        with self._lock:
            old = self._data.get(key)
            if old is not None:
                pinned = pinned or old.pinned
                self._remove(old)
            entry = _Entry(
                key=key,
                value=value,
                endpoint_id=endpoint_id,
                model_version=model_version,
                created_at=time.time(),
                ttl_s=float(ttl_s),
                swr_s=float(swr_s),
                pinned=pinned,
                size_bytes=size,
                raw=raw,
            )
            self._data[key] = entry
            self._bytes += size
            self._evict()

    def delete(self, key: str) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            self._remove(entry)
            return True

    def pin(self, key: str, pinned: bool = True) -> bool:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return False
            entry.pinned = pinned
            return True

    def invalidate_endpoint(self, endpoint_id: str) -> int:
        with self._lock:
            doomed = [e for e in self._data.values() if e.endpoint_id == endpoint_id and not e.pinned]
            for entry in doomed:
                self._remove(entry)
            return len(doomed)

    def entries(self, limit: int = 100, endpoint_id: str | None = None) -> list[CacheEntryInfo]:
        with self._lock:
            items: Iterable[_Entry] = list(self._data.values())
            if endpoint_id:
                items = [e for e in items if e.endpoint_id == endpoint_id]
            ordered = sorted(items, key=lambda e: (e.hits, e.created_at), reverse=True)
            return [e.to_info() for e in ordered[:limit]]

    def stats(self) -> CacheStats:
        with self._lock:
            total = self._hits + self._misses + self._stale_hits
            return CacheStats(
                backend=self.name,
                entries=len(self._data),
                hits=self._hits,
                misses=self._misses,
                stale_hits=self._stale_hits,
                evictions=self._evictions,
                hit_ratio=round((self._hits + self._stale_hits) / total, 4) if total else 0.0,
                size_bytes=self._bytes,
                max_entries=self.max_entries,
                pinned=sum(1 for e in self._data.values() if e.pinned),
            )

    def hot_keys(self, endpoint_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Top-K keys by hit count - sent to the agent as warm-up samples on reconnect."""
        with self._lock:
            items = list(self._data.values())
            if endpoint_id:
                items = [e for e in items if e.endpoint_id == endpoint_id]
            items.sort(key=lambda e: (e.pinned, e.hits, e.created_at), reverse=True)
            return [
                {
                    "key": e.key,
                    "endpoint_id": e.endpoint_id,
                    "model_version": e.model_version,
                    "hits": e.hits,
                    "sample": e.value.get("__request__") if isinstance(e.value, dict) else None,
                }
                for e in items[:limit]
            ]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    # ------------------------------------------------------------------ #
    def _remove(self, entry: _Entry) -> None:
        self._data.pop(entry.key, None)
        self._bytes = max(0, self._bytes - entry.size_bytes)

    def _evict(self) -> None:
        """LRU eviction, skipping pinned entries."""
        if len(self._data) <= self.max_entries and self._bytes <= self.max_bytes:
            return
        for key in list(self._data.keys()):
            if len(self._data) <= self.max_entries and self._bytes <= self.max_bytes:
                break
            entry = self._data[key]
            if entry.pinned:
                continue
            self._remove(entry)
            self._evictions += 1

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "saved_at": time.time(),
                "counters": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "stale_hits": self._stale_hits,
                    "evictions": self._evictions,
                },
                "entries": [
                    {
                        "key": e.key,
                        "value": e.value,
                        "endpoint_id": e.endpoint_id,
                        "model_version": e.model_version,
                        "created_at": e.created_at,
                        "ttl_s": e.ttl_s,
                        "swr_s": e.swr_s,
                        "pinned": e.pinned,
                        "hits": e.hits,
                    }
                    for e in self._data.values()
                ],
            }

    def save(self) -> bool:
        """Persist to disk so a gateway restart does not lose the resilience buffer."""
        if self.persist_path is None:
            return False
        with self._lock:
            snap = self.snapshot()
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persist_path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(snap, fh, default=str, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.persist_path)
            return True

    def load(self) -> int:
        if self.persist_path is None or not self.persist_path.exists():
            return 0
        try:
            with self.persist_path.open("r", encoding="utf-8") as fh:
                snap = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return 0
        now = time.time()
        loaded = 0
        counters = snap.get("counters", {})
        self._hits = int(counters.get("hits", 0))
        self._misses = int(counters.get("misses", 0))
        self._stale_hits = int(counters.get("stale_hits", 0))
        self._evictions = int(counters.get("evictions", 0))
        for row in snap.get("entries", []):
            # Drop entries that are past even their stale window while we were down.
            if float(row.get("created_at", 0)) + float(row.get("ttl_s", 0)) + float(
                row.get("swr_s", 0)
            ) < now:
                continue
            size, raw = _size_of(row.get("value"))
            entry = _Entry(
                key=row["key"],
                value=row.get("value"),
                endpoint_id=row.get("endpoint_id", ""),
                model_version=int(row.get("model_version", 0)),
                created_at=float(row.get("created_at", now)),
                ttl_s=float(row.get("ttl_s", 0)),
                swr_s=float(row.get("swr_s", 0)),
                pinned=bool(row.get("pinned", False)),
                hits=int(row.get("hits", 0)),
                size_bytes=size,
                raw=raw,
            )
            self._data[entry.key] = entry
            self._bytes += size
            loaded += 1
        self._evict()
        return loaded

    def close(self) -> None:
        self.save()


class SqliteCache(CacheBackend):
    """Durable cache backend - survives restarts with no snapshot step.

    Same semantics as :class:`MemoryCache`, backed by a single SQLite file. Used
    when ``CDP_CACHE_BACKEND=sqlite``; a good default for larger caches.
    """

    name = "sqlite"

    def __init__(self, path: str | Path, max_entries: int = 100000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                endpoint_id TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                value TEXT NOT NULL,
                created_at REAL NOT NULL,
                ttl_s REAL NOT NULL,
                swr_s REAL NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                hits INTEGER NOT NULL DEFAULT 0,
                size_bytes INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS cache_ep ON cache(endpoint_id)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL)"
        )
        self._conn.commit()

    def _row_to_entry(self, row: tuple) -> _Entry:
        value = json.loads(row[3])
        return _Entry(
            key=row[0],
            value=value,
            endpoint_id=row[1],
            model_version=int(row[2]),
            created_at=float(row[4]),
            ttl_s=float(row[5]),
            swr_s=float(row[6]),
            pinned=bool(row[7]),
            hits=int(row[8]),
            size_bytes=int(row[9]),
        )

    _SELECT = (
        "SELECT key, endpoint_id, model_version, value, created_at, ttl_s, swr_s, pinned, hits, "
        "size_bytes FROM cache WHERE key = ?"
    )

    def get(self, key: str) -> CacheLookup:
        with self._lock, self._conn:
            cur = self._conn.execute(self._SELECT, (key,))
            row = cur.fetchone()
            if row is None:
                self._bump("misses")
                return CacheLookup(status=CACHE_MISS, key=key)
            entry = self._row_to_entry(row)
            now = time.time()
            status = entry.status(now)
            if status == CACHE_MISS:
                self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                self._bump("misses")
                return CacheLookup(status=CACHE_MISS, key=key)
            self._conn.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            self._bump("hits" if status == CACHE_HIT else "stale_hits")
            return CacheLookup(
                status=status,
                value=entry.value,
                key=key,
                age_s=max(0.0, now - entry.created_at),
                stale_for_s=max(0.0, now - entry.expires_at) if status == CACHE_STALE else 0.0,
                created_at=entry.created_at,
                endpoint_id=entry.endpoint_id,
                model_version=entry.model_version,
            )

    def peek(self, key: str) -> CacheLookup:
        with self._lock:
            cur = self._conn.execute(self._SELECT, (key,))
            row = cur.fetchone()
            if row is None:
                return CacheLookup(status=CACHE_MISS, key=key)
            entry = self._row_to_entry(row)
            status = entry.status()
            return CacheLookup(
                status=status,
                value=entry.value if status != CACHE_MISS else None,
                key=key,
                created_at=entry.created_at,
                endpoint_id=entry.endpoint_id,
                model_version=entry.model_version,
            )

    def set(
        self,
        key: str,
        value: Any,
        *,
        endpoint_id: str,
        model_version: int,
        ttl_s: float,
        swr_s: float = 0.0,
        pinned: bool = False,
    ) -> None:
        raw = json.dumps(value, default=str, separators=(",", ":"))
        with self._lock, self._conn:
            existing = self._conn.execute("SELECT pinned FROM cache WHERE key = ?", (key,)).fetchone()
            if existing:
                pinned = pinned or bool(existing[0])
            self._conn.execute(
                """
                INSERT INTO cache (key, endpoint_id, model_version, value, created_at, ttl_s,
                                   swr_s, pinned, hits, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value, created_at=excluded.created_at, ttl_s=excluded.ttl_s,
                    swr_s=excluded.swr_s, model_version=excluded.model_version,
                    pinned=excluded.pinned, size_bytes=excluded.size_bytes, hits=0
                """,
                (
                    key,
                    endpoint_id,
                    model_version,
                    raw,
                    time.time(),
                    float(ttl_s),
                    float(swr_s),
                    int(pinned),
                    len(raw.encode("utf-8")),
                ),
            )
            self._trim()

    def _trim(self) -> None:
        count = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        if count <= self.max_entries:
            return
        excess = count - self.max_entries
        self._conn.execute(
            """
            DELETE FROM cache WHERE key IN (
                SELECT key FROM cache WHERE pinned = 0 ORDER BY hits ASC, created_at ASC LIMIT ?
            )
            """,
            (excess,),
        )
        self._bump("evictions", excess)

    def delete(self, key: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            return cur.rowcount > 0

    def pin(self, key: str, pinned: bool = True) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("UPDATE cache SET pinned = ? WHERE key = ?", (int(pinned), key))
            return cur.rowcount > 0

    def invalidate_endpoint(self, endpoint_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM cache WHERE endpoint_id = ? AND pinned = 0", (endpoint_id,)
            )
            return cur.rowcount

    def entries(self, limit: int = 100, endpoint_id: str | None = None) -> list[CacheEntryInfo]:
        query = (
            "SELECT key, endpoint_id, model_version, value, created_at, ttl_s, swr_s, pinned, "
            "hits, size_bytes FROM cache"
        )
        params: tuple[Any, ...] = ()
        if endpoint_id:
            query += " WHERE endpoint_id = ?"
            params = (endpoint_id,)
        query += " ORDER BY hits DESC, created_at DESC LIMIT ?"
        params = (*params, limit)
        with self._lock:
            return [self._row_to_entry(row).to_info() for row in self._conn.execute(query, params)]

    def stats(self) -> CacheStats:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes),0), COALESCE(SUM(pinned),0) FROM cache"
            ).fetchone()
            counters = dict(self._conn.execute("SELECT name, value FROM counters").fetchall())
            hits = int(counters.get("hits", 0))
            misses = int(counters.get("misses", 0))
            stale = int(counters.get("stale_hits", 0))
            total = hits + misses + stale
            return CacheStats(
                backend=self.name,
                entries=int(row[0]),
                hits=hits,
                misses=misses,
                stale_hits=stale,
                evictions=int(counters.get("evictions", 0)),
                hit_ratio=round((hits + stale) / total, 4) if total else 0.0,
                size_bytes=int(row[1]),
                max_entries=self.max_entries,
                pinned=int(row[2]),
            )

    def hot_keys(self, endpoint_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT key, endpoint_id, model_version, value, hits FROM cache"
        params: tuple[Any, ...] = ()
        if endpoint_id:
            query += " WHERE endpoint_id = ?"
            params = (endpoint_id,)
        query += " ORDER BY pinned DESC, hits DESC LIMIT ?"
        params = (*params, limit)
        out = []
        with self._lock:
            for key, ep, version, value, hits in self._conn.execute(query, params):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:  # pragma: no cover
                    parsed = None
                sample = parsed.get("__request__") if isinstance(parsed, dict) else None
                out.append(
                    {
                        "key": key,
                        "endpoint_id": ep,
                        "model_version": int(version),
                        "hits": int(hits),
                        "sample": sample,
                    }
                )
        return out

    def _bump(self, name: str, amount: int = 1) -> None:
        self._conn.execute(
            """
            INSERT INTO counters (name, value) VALUES (?, ?)
            ON CONFLICT(name) DO UPDATE SET value = value + excluded.value
            """,
            (name, amount),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()


def build_cache(
    backend: str = "memory",
    *,
    data_dir: str | Path = ".data",
    max_entries: int = 20000,
    max_bytes: int = 256 * 1024 * 1024,
    persist: bool = True,
) -> CacheBackend:
    """Factory used by the app lifespan and by tests.

    Bounds are honoured exactly as given: the caller (config or a test) decides how
    much memory the resilience buffer may occupy, and silently raising a cap here
    would make that limit meaningless.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if backend == "sqlite":
        return SqliteCache(data_dir / "cache.sqlite3", max_entries=max(1, max_entries))
    return MemoryCache(
        max_entries=max(1, max_entries),
        max_bytes=max(1024, max_bytes),
        persist_path=(data_dir / "cache.json") if persist else None,
    )
