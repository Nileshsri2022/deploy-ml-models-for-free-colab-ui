"""A Redis-backed mirror of :class:`~gateway.core.store.JsonStore`.

Why: free-tier hosts (Render's free web service above all) recycle the
container and its disk without warning. The registry is tiny (a few MB of
JSON), so mirroring every record to a free Redis (Upstash's free tier is
256 MB / 500K commands per month) and restoring it on boot makes the
registry outlive any spin-down, recycle or redeploy - at $0.

Design, in order of importance:

* **Redis is the mirror, the disk is still the write path.** Every mutation
  writes the local JSON file atomically first (exactly as ``JsonStore`` does),
  then marks one key dirty. A background thread pushes dirty keys to Redis at
  most every ``sync_interval_s``. If Redis is unreachable the gateway keeps
  working exactly like the plain file store; it retries the connection in the
  background and catches up on the next successful sync.
* **One Redis key per record**, not one blob per collection. The largest
  collection (audit, capped at 2000 entries) would be a ~0.5 MB payload
  rewritten on every mutation, which burns the free tier's command and
  bandwidth budget for nothing. Per-record keys mean a heartbeat syncs ~1 KB,
  not the whole collection.
* **Coalesced writes** (``sync_interval_s``, default 30): rapid mutations of
  the same key collapse into one ``SET``. A 15 s heartbeat therefore costs
  ~86K commands a month against Upstash's 500K budget, and a hard kill loses
  at most the last sync window of *mirror* data (the disk copy is immediate;
  a graceful shutdown flushes synchronously via ``close()``).
* **Restore on boot**: on first access to a collection the store reads its
  keys from Redis (``SCAN`` + ``MGET``), rebuilds the in-memory cache and
  re-materialises the local JSON file, so the disk copy keeps parity. If
  Redis has nothing, the local file is used as-is - meaning the fallback path
  is byte-for-byte today's behaviour.

Not mirrored on purpose: the inference cache (a lossy resilience buffer,
capped at ``CDP_CACHE_MAX_BYTES`` = 256 MB by default - it would eat the
entire free tier) and uploaded artifacts (binary, and they belong on the
agent's VM). This store only carries the registry metadata.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gateway.core.store import JsonStore

logger = logging.getLogger("gateway.registry")

_META_SUFFIX = ":__meta__"


class RedisSnapshotStore(JsonStore):
    """``JsonStore`` with a per-record Redis mirror and boot-time restore."""

    def __init__(
        self,
        root: str | Path,
        *,
        redis_url: str = "",
        namespace: str = "cdp:registry:v1",
        sync_interval_s: float = 30.0,
        keepalive_s: float = 86400.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(root)
        self.namespace = namespace.rstrip(":")
        self.sync_interval_s = max(0.0, float(sync_interval_s))
        self.keepalive_s = max(0.0, float(keepalive_s))
        self._url = redis_url
        self._injected = client is not None
        self._client: Any | None = client
        self._last_connect_attempt = -60.0  # allow the first attempt immediately
        self._connect_warned = False

        # (collection, key) -> monotonic sequence number, for "set" or "del".
        self._dirty: dict[tuple[str, str], str] = {}
        self._dirty_seq: dict[tuple[str, str], int] = {}
        self._seq = 0
        self._dirty_guard = threading.Lock()
        self._wakeup = threading.Event()
        self._closing = False
        self._last_keepalive = time.monotonic()

        if self.sync_interval_s > 0:
            self._flusher = threading.Thread(
                target=self._flusher_loop, name="cdp-redis-sync", daemon=True
            )
            self._flusher.start()

    # ------------------------------------------------------------------ #
    # Redis client management
    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> Any | None:
        """Return a connected client, (re)connecting at most every 30 s."""
        if self._client is not None:
            return self._client
        if not self._url:
            return None
        now = time.monotonic()
        if not self._injected and now - self._last_connect_attempt < 30.0:
            return None
        self._last_connect_attempt = now
        try:
            import redis  # optional dependency: pip install 'colabdeploy-gateway[redis]'

            client = redis.Redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            client.ping()
        except Exception as exc:  # any failure degrades to file-only
            if not self._connect_warned:
                logger.warning(
                    "registry mirror: Redis unreachable (%s); continuing on the "
                    "instance disk and retrying in the background",
                    exc,
                )
                self._connect_warned = True
            return None
        logger.info("registry mirror: connected to Redis")
        return client

    def _drop_client(self, exc: Exception) -> None:
        if not self._injected:
            self._client = None
        if not self._connect_warned:
            logger.warning(
                "registry mirror: Redis write failed (%s); will reconnect and retry",
                exc,
            )
            self._connect_warned = True

    # ------------------------------------------------------------------ #
    # Key helpers
    # ------------------------------------------------------------------ #
    def _key(self, collection: str, key: str) -> str:
        return f"{self.namespace}:{collection}:{key}"

    def _meta_key(self, collection: str) -> str:
        return f"{self.namespace}:{collection}{_META_SUFFIX}"

    # ------------------------------------------------------------------ #
    # Mutations: local write first (super), then mark one key dirty
    # ------------------------------------------------------------------ #
    def put(self, collection: str, key: str, value: dict[str, Any]) -> dict[str, Any]:
        super().put(collection, key, value)
        self._mark(collection, key, "set")
        return value

    def update(
        self, collection: str, key: str, patch: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any] | None:
        merged = super().update(collection, key, patch)
        if merged is not None:
            self._mark(collection, key, "set")
        return merged

    def delete(self, collection: str, key: str) -> bool:
        removed = super().delete(collection, key)
        if removed:
            self._mark(collection, key, "del")
        return removed

    def clear(self, collection: str | None = None) -> None:
        collections = [collection] if collection else list(self._cache)
        stale = {
            (name, key): "del"
            for name in collections
            for key in list(self._cache.get(name, {}).get("items", {}))
        }
        super().clear(collection)
        if stale:
            with self._dirty_guard:
                self._dirty.update(stale)
                for mark in stale:
                    self._seq += 1
                    self._dirty_seq[mark] = self._seq
                self._wakeup.set()
            if self.sync_interval_s <= 0:
                self._sync_once()

    def _mark(self, collection: str, key: str, op: str) -> None:
        with self._dirty_guard:
            self._dirty[(collection, key)] = op
            self._seq += 1
            self._dirty_seq[(collection, key)] = self._seq
            self._wakeup.set()
        if self.sync_interval_s <= 0:
            self._sync_once()

    # ------------------------------------------------------------------ #
    # Restore on boot
    # ------------------------------------------------------------------ #
    def _load(self, collection: str) -> dict[str, Any]:
        cached = self._cache.get(collection)
        if cached is not None:
            return cached
        restored = self._restore(collection)
        if restored is not None:
            self._cache[collection] = restored
            return restored
        return super()._load(collection)

    def warm(self, collections: list[str]) -> int:
        """Eagerly pull every known collection from Redis (call once at boot)."""
        restored = 0
        for name in collections:
            if name in self._cache:
                continue
            data = self._restore(name)
            if data is not None:
                self._cache[name] = data
                restored += 1
        return restored

    def _restore(self, collection: str) -> dict[str, Any] | None:
        """Rebuild one collection from Redis; ``None`` means "nothing there"."""
        client = self._ensure_client()
        if client is None:
            return None
        try:
            pattern = f"{self.namespace}:{collection}:*"
            keys: list[str] = []
            cursor: int | str = 0
            while True:
                cursor, batch = client.scan(cursor=cursor, match=pattern, count=500)
                keys.extend(k for k in batch if not k.endswith(_META_SUFFIX))
                if int(cursor) == 0:
                    break
            if not keys:
                return None
            prefix = len(self.namespace) + len(collection) + 2
            items: dict[str, Any] = {}
            for key, raw in zip(keys, client.mget(keys), strict=False):
                if raw is None:
                    continue
                try:
                    obj = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict):
                    items[key[prefix:]] = obj
            revision = 0
            meta_raw = client.get(self._meta_key(collection))
            if meta_raw:
                try:
                    revision = int(json.loads(meta_raw).get("revision", 0))
                except (ValueError, TypeError):
                    revision = 0
            data = {"items": items, "revision": revision}
            self._materialize(collection, data)
            return data
        except Exception as exc:  # degrade to the local file
            self._drop_client(exc)
            return None

    def _materialize(self, collection: str, data: dict[str, Any]) -> None:
        """Write a restored collection to disk with the same atomic swap."""
        path = self._path(collection)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, default=str, separators=(",", ":"))
                fh.flush()
            tmp.replace(path)
        except OSError:
            pass  # disk is only a mirror here; Redis stays authoritative

    # ------------------------------------------------------------------ #
    # Background sync
    # ------------------------------------------------------------------ #
    def _flusher_loop(self) -> None:
        """Sync purely on the clock, so bursts inside one window cost one pipeline.

        The wakeup event shortens the sleep only for shutdown; mutation bursts
        never trigger an early sync (the disk copy is the immediate one).
        """
        next_sync = time.monotonic() + self.sync_interval_s
        while not self._closing:
            remaining = max(0.0, next_sync - time.monotonic())
            self._wakeup.wait(remaining)
            self._wakeup.clear()
            if self._closing:
                break
            if time.monotonic() < next_sync:
                continue  # woken early: stay on schedule
            self._sync_once()
            self._maybe_keepalive()
            next_sync = time.monotonic() + self.sync_interval_s

    def _maybe_keepalive(self) -> None:
        """Keep free tiers that archive idle databases (Upstash) awake."""
        if self.keepalive_s <= 0 or self._client is None:
            return
        if time.monotonic() - self._last_keepalive < self.keepalive_s:
            return
        self._last_keepalive = time.monotonic()
        try:
            self._client.set(
                f"{self.namespace}:__keepalive__", str(int(time.time()))
            )
        except Exception as exc:  # retry on the next cycle
            self._drop_client(exc)

    def _sync_once(self) -> None:
        client = self._ensure_client()
        if client is None:
            return
        with self._dirty_guard:
            batch = dict(self._dirty)
            seqs = dict(self._dirty_seq)
        if not batch:
            return
        touched: dict[str, list[tuple[str, str, str]]] = {}
        for (collection, key), op in batch.items():
            touched.setdefault(collection, []).append((key, op, self._key(collection, key)))
        try:
            pipe = client.pipeline(transaction=False)
            for collection, entries in touched.items():
                for key, op, full_key in entries:
                    if op == "del":
                        pipe.delete(full_key)
                    else:
                        blob = self._serialize_one(collection, key)
                        if blob is None:
                            pipe.delete(full_key)  # vanished locally: mirror the truth
                        else:
                            pipe.set(full_key, blob)
                with self._lock(collection):
                    revision = int(self._cache.get(collection, {}).get("revision", 0))
                pipe.set(
                    self._meta_key(collection),
                    json.dumps({"revision": revision}, separators=(",", ":")),
                )
            pipe.execute()
        except Exception as exc:  # retry on the next cycle
            self._drop_client(exc)
            return
        with self._dirty_guard:
            for mark, seq in seqs.items():
                # Only clear if the key was not mutated again while we synced.
                if self._dirty_seq.get(mark) == seq:
                    self._dirty.pop(mark, None)
                    self._dirty_seq.pop(mark, None)
        self._last_keepalive = time.monotonic()

    def _serialize_one(self, collection: str, key: str) -> str | None:
        with self._lock(collection):
            item = self._cache.get(collection, {}).get("items", {}).get(key)
            return None if item is None else json.dumps(item, default=str, separators=(",", ":"))

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Flush pending mirror writes and stop the sync thread."""
        self._closing = True
        self._wakeup.set()
        thread = self.__dict__.get("_flusher")
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        self._sync_once()

    def sync_stats(self) -> dict[str, int]:
        """Diagnostics: how many records are queued for the mirror."""
        with self._dirty_guard:
            return {"pending": len(self._dirty), "collections": len(self._cache)}


def build_registry_store(settings: Any) -> JsonStore:
    """Factory used by the app container: file store, or Redis-mirrored store.

    Never raises: if the ``redis`` package is missing or the URL is empty the
    gateway falls back to the plain file store with a warning, which is the
    pre-Redis behaviour.
    """
    backend = (getattr(settings, "registry_backend", "file") or "file").strip().lower()
    url = (getattr(settings, "redis_url", "") or "").strip()
    if backend != "redis" or not url:
        if backend == "redis":
            logger.warning(
                "CDP_REGISTRY_BACKEND=redis but CDP_REDIS_URL is empty; "
                "the registry stays on the instance disk"
            )
        return JsonStore(settings.data_dir / "store")
    try:
        import redis  # noqa: F401 - optional dependency, probed up front
    except ImportError:
        logger.warning(
            "CDP_REGISTRY_BACKEND=redis but the redis client is not installed; "
            "run pip install 'colabdeploy-gateway[redis]'. "
            "Continuing with disk-only persistence."
        )
        return JsonStore(settings.data_dir / "store")
    try:
        return RedisSnapshotStore(
            settings.data_dir / "store",
            redis_url=url,
            sync_interval_s=float(getattr(settings, "registry_sync_s", 30.0)),
            keepalive_s=float(getattr(settings, "registry_keepalive_s", 86400.0)),
        )
    except Exception as exc:  # a mirror must never block boot
        logger.warning(
            "registry mirror disabled (%s); install it with "
            "pip install 'colabdeploy-gateway[redis]'",
            exc,
        )
        return JsonStore(settings.data_dir / "store")
