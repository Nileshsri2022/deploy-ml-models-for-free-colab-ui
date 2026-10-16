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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gateway.core.sentry import capture_message, scrub_sensitive_text
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
        event_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(root)
        self.namespace = namespace.rstrip(":")
        self.sync_interval_s = max(0.0, float(sync_interval_s))
        self.keepalive_s = max(0.0, float(keepalive_s))
        self._url = redis_url
        self._injected = client is not None
        self._client: Any | None = client
        self._event_callback = event_callback
        self._last_connect_attempt = -60.0  # allow the first attempt immediately
        self._connect_warned = False
        self._ever_connected = False
        self._connected = False
        self._last_error: str | None = None
        self._last_error_at: str | None = None
        self._last_success_at: str | None = None

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
            self._record_failure("connect", exc)
            return None

        self._client = client
        self._record_success()
        self._ever_connected = True
        return client

    def _drop_client(self, exc: Exception, *, operation: str = "sync") -> None:
        if not self._injected:
            self._client = None
        self._record_failure(operation, exc)

    def _record_failure(self, operation: str, exc: Exception) -> None:
        reason = scrub_sensitive_text(f"{type(exc).__name__}: {exc}")[:500]
        self._connected = False
        self._last_error = reason
        self._last_error_at = datetime.now(timezone.utc).isoformat()
        if self._connect_warned:
            return  # one warning/event per outage, not one per retry
        self._connect_warned = True
        message = "Upstash Redis operation failed; continuing with the local registry"
        logger.warning("registry mirror: %s (%s)", message, reason)
        self._emit_event(
            message,
            "warning",
            {"component": "upstash", "operation": operation, "reason": reason},
        )
        capture_message(
            f"{message}: {reason}",
            level="warning",
            tags={"component": "upstash", "operation": operation},
        )

    def _record_success(self) -> None:
        recovered = self._connect_warned
        first_connection = not self._ever_connected
        self._connected = True
        self._last_success_at = datetime.now(timezone.utc).isoformat()
        self._last_error = None
        self._last_error_at = None
        self._connect_warned = False
        if recovered or first_connection:
            message = (
                "Upstash Redis connection recovered" if recovered else "Upstash Redis connected"
            )
            logger.info("registry mirror: %s", message)
            self._emit_event(message, "info", {"component": "upstash", "status": "connected"})

    def _emit_event(self, message: str, level: str, data: dict[str, Any]) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback(message, level, data)
        except Exception:
            logger.exception("registry mirror: unable to publish health event")

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
        self,
        collection: str,
        key: str,
        patch: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]],
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
            self._record_success()
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
            self._drop_client(exc, operation="restore")
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
            self._client.set(f"{self.namespace}:__keepalive__", str(int(time.time())))
        except Exception as exc:  # retry on the next cycle
            self._drop_client(exc, operation="keepalive")

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
            self._drop_client(exc, operation="sync")
            return
        self._record_success()
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

    def _health_status(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "status": "connected" if self._connected else "disconnected",
            "last_error": self._last_error,
            "last_error_at": self._last_error_at,
            "last_success_at": self._last_success_at,
        }

    def sync_stats(self) -> dict[str, Any]:
        """Diagnostics for the mirror queue and last connection/sync state."""
        with self._dirty_guard:
            counts = {"pending": len(self._dirty), "collections": len(self._cache)}
        return {**counts, **self._health_status()}

    # ------------------------------------------------------------------ #
    # Upstash inspector (what the UI Cache page shows)
    # ------------------------------------------------------------------ #
    def inspect(self, limit: int = 200, collection: str | None = None) -> dict[str, Any]:
        """List live Redis keys. This is the Upstash view — not the inference cache."""
        client = self._ensure_client()
        empty: dict[str, Any] = {
            "backend": "upstash",
            "namespace": self.namespace,
            "entries": 0,
            "size_bytes": 0,
            "keys": [],
            "collections": {},
            **self.sync_stats(),
        }
        if client is None:
            return empty
        try:
            pattern = f"{self.namespace}:{collection}:*" if collection else f"{self.namespace}:*"
            redis_keys: list[str] = []
            cursor: int | str = 0
            while True:
                cursor, batch = client.scan(cursor=cursor, match=pattern, count=500)
                redis_keys.extend(batch)
                if int(cursor) == 0:
                    break
            redis_keys.sort()
            sliced = redis_keys[: max(1, limit)]
            values = client.mget(sliced) if sliced else []
            keys_out: list[dict[str, Any]] = []
            collections: dict[str, int] = {}
            total_bytes = 0
            for full_key, raw in zip(sliced, values, strict=False):
                rest = (
                    full_key[len(self.namespace) + 1 :]
                    if full_key.startswith(self.namespace + ":")
                    else full_key
                )
                if rest.endswith(_META_SUFFIX):
                    coll = rest[: -len(_META_SUFFIX)]
                    record_id = "__meta__"
                    kind = "meta"
                elif rest == "__keepalive__":
                    coll = "_system"
                    record_id = "__keepalive__"
                    kind = "keepalive"
                else:
                    coll, _, record_id = rest.partition(":")
                    kind = "record"
                collections[coll] = collections.get(coll, 0) + 1
                blob = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
                size = len(blob.encode("utf-8"))
                total_bytes += size
                preview = blob[:400] + ("…" if len(blob) > 400 else "")
                parsed: Any = None
                if blob:
                    try:
                        parsed = json.loads(blob)
                    except (ValueError, TypeError):
                        parsed = None
                keys_out.append(
                    {
                        "key": full_key,
                        "collection": coll,
                        "id": record_id,
                        "kind": kind,
                        "size_bytes": size,
                        "preview": preview,
                        "value": parsed if parsed is not None else blob,
                    }
                )
            self._record_success()
            return {
                "backend": "upstash",
                "namespace": self.namespace,
                "entries": len(redis_keys),
                "shown": len(keys_out),
                "size_bytes": total_bytes,
                "keys": keys_out,
                "collections": collections,
                **self.sync_stats(),
            }
        except Exception as exc:
            self._drop_client(exc, operation="inspect")
            empty.update(self._health_status())
            empty["error"] = self._last_error
            return empty

    def inspect_key(self, full_key: str) -> dict[str, Any] | None:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            raw = client.get(full_key)
        except Exception as exc:
            self._drop_client(exc, operation="inspect_key")
            return None
        self._record_success()
        if raw is None:
            return None
        blob = raw if isinstance(raw, str) else str(raw)
        parsed: Any = None
        try:
            parsed = json.loads(blob)
        except (ValueError, TypeError):
            parsed = blob
        return {"key": full_key, "size_bytes": len(blob.encode("utf-8")), "value": parsed}

    def drop_key(self, full_key: str) -> bool:
        """Delete one Redis key. Local JSON is left as-is (mirror only)."""
        client = self._ensure_client()
        if client is None:
            return False
        try:
            deleted = bool(client.delete(full_key))
            self._record_success()
            return deleted
        except Exception as exc:
            self._drop_client(exc, operation="delete_key")
            return False


def _report_factory_issue(
    message: str,
    reason: str,
    *,
    event_callback: Callable[[str, str, dict[str, Any]], None] | None,
    operation: str,
) -> None:
    safe_reason = scrub_sensitive_text(reason)[:500]
    logger.warning("%s (%s)", message, safe_reason)
    if event_callback is not None:
        try:
            event_callback(
                message,
                "warning",
                {"component": "upstash", "operation": operation, "reason": safe_reason},
            )
        except Exception:
            logger.exception("registry mirror: unable to publish setup warning")
    capture_message(
        f"{message}: {safe_reason}",
        level="warning",
        tags={"component": "upstash", "operation": operation},
    )


def build_registry_store(
    settings: Any,
    *,
    event_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> JsonStore:
    """Factory used by the app container: file store, or Redis-mirrored store.

    Never raises: if the ``redis`` package is missing or the URL is empty the
    gateway falls back to the plain file store with a warning, which is the
    pre-Redis behaviour.
    """
    backend = (getattr(settings, "registry_backend", "file") or "file").strip().lower()
    url = (getattr(settings, "redis_url", "") or "").strip()
    if backend != "redis" or not url:
        if backend == "redis":
            _report_factory_issue(
                "CDP_REGISTRY_BACKEND=redis but CDP_REDIS_URL is empty; using local JSON",
                "Redis URL is not configured",
                event_callback=event_callback,
                operation="configuration",
            )
        return JsonStore(settings.data_dir / "store")
    try:
        import redis  # noqa: F401 - optional dependency, probed up front
    except ImportError:
        _report_factory_issue(
            "CDP_REGISTRY_BACKEND=redis but the redis client is not installed; using local JSON",
            "Install the gateway redis extra",
            event_callback=event_callback,
            operation="configuration",
        )
        return JsonStore(settings.data_dir / "store")
    try:
        return RedisSnapshotStore(
            settings.data_dir / "store",
            redis_url=url,
            sync_interval_s=float(getattr(settings, "registry_sync_s", 30.0)),
            keepalive_s=float(getattr(settings, "registry_keepalive_s", 86400.0)),
            event_callback=event_callback,
        )
    except Exception as exc:  # a mirror must never block boot
        _report_factory_issue(
            "registry mirror could not be initialized; using local JSON",
            f"{type(exc).__name__}: {exc}",
            event_callback=event_callback,
            operation="initialize",
        )
        return JsonStore(settings.data_dir / "store")
