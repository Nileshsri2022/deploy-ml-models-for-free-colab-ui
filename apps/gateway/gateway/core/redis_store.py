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

import contextlib
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

#: Reconnect at most this often when Redis is *unreachable* (seconds). The
#: backoff deliberately applies to failures only: a healthy client is cached
#: and reused, never re-negotiated.
CONNECT_BACKOFF_S = 30.0

#: Schemes ``redis.Redis.from_url`` understands (without the ``://``). Anything
#: else — notably Upstash's *REST* endpoint, ``https://…upstash.io`` — can never
#: connect, and saying so up front beats a generic "not connected".
_REDIS_SCHEMES = ("redis", "rediss", "unix")


def describe_url(url: str) -> dict[str, Any]:
    """Sanity-check a ``CDP_REDIS_URL`` without touching the network.

    Returns ``{"scheme_ok": bool, "host": str, "problem": str}``. ``problem`` is
    empty when the URL is at least plausible, so the UI can name the actual
    mistake (a pasted REST URL, an untrimmed quote, a copy of the Read-only
    replica) instead of blaming env vars that are set correctly.
    """
    raw = (url or "").strip()
    out: dict[str, Any] = {"scheme_ok": False, "host": "", "problem": ""}
    if not raw:
        out["problem"] = "CDP_REDIS_URL is empty in this process's environment"
        return out
    if raw[0] in "\"'" or raw[-1] in "\"'":
        out["problem"] = "CDP_REDIS_URL is wrapped in quote characters"
        return out
    scheme, sep, rest = raw.partition("://")
    if not sep:
        out["problem"] = "CDP_REDIS_URL is not a URL (expected rediss://user:pass@host:port)"
        return out
    out["scheme_ok"] = scheme.strip().lower() in _REDIS_SCHEMES
    out["host"] = rest.rpartition("@")[2].split("/")[0]
    if not out["scheme_ok"]:
        hint = " — Upstash's REST/HTTPS endpoint is not a Redis URL" if scheme.lower().startswith("http") else ""
        out["problem"] = f"unsupported scheme '{scheme}:'{hint}"
    elif "upstash" in out["host"] and scheme.strip().lower() == "redis":
        out["problem"] = "Upstash requires TLS: use the rediss:// URL, not redis://"
    return out


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
        connect_backoff_s: float = CONNECT_BACKOFF_S,
        client: Any | None = None,
    ) -> None:
        super().__init__(root)
        self.namespace = namespace.rstrip(":")
        self.sync_interval_s = max(0.0, float(sync_interval_s))
        self.keepalive_s = max(0.0, float(keepalive_s))
        self.connect_backoff_s = max(0.0, float(connect_backoff_s))
        self._url = (redis_url or "").strip().strip("\"'")
        self._injected = client is not None
        self._client: Any | None = client
        self._last_connect_attempt = -60.0  # allow the first attempt immediately
        self._connect_warned = False
        # Why Redis is not answering, in the words the failing call produced.
        # Surfaced by status()/the /api/upstash endpoints so the UI can explain
        # "configured but unreachable" instead of "not configured".
        self._last_error = ""
        self._connect_guard = threading.Lock()
        self.restored_at_boot = 0
        url_state = describe_url(self._url)
        self._url_problem = url_state["problem"]
        self.url_ok = bool(self._url) and url_state["scheme_ok"] and not self._url_problem
        if self._url_problem:
            self._last_error = self._url_problem
            logger.warning("registry mirror disabled: %s", self._last_error)

        # (collection, key) -> monotonic sequence number, for "set" or "del".
        self._dirty: dict[tuple[str, str], str] = {}
        self._dirty_seq: dict[tuple[str, str], int] = {}
        self._seq = 0
        self._dirty_guard = threading.Lock()
        self._wakeup = threading.Event()
        self._closing = False
        self._last_keepalive = time.monotonic()

        # Always started: in coalescing mode it is the syncer, in write-through
        # mode it is the catch-up retry for a flush that failed inline.
        self._flusher = threading.Thread(
            target=self._flusher_loop, name="cdp-redis-sync", daemon=True
        )
        self._flusher.start()

    # ------------------------------------------------------------------ #
    # Redis client management
    # ------------------------------------------------------------------ #
    def _ensure_client(self) -> Any | None:
        """Return the shared Redis client, connecting (or retrying) as needed.

        Two rules make the difference between a working mirror and a silent
        "not connected":

        * the client is **cached** on ``self._client``. Opening a fresh socket
          per call leaked connections until Upstash refused more (its free tier
          caps concurrent clients), and it left ``self._client`` ``None``
          forever, which disabled the keepalive write;
        * the backoff applies to **failed** attempts only. Throttling every
          attempt made the 30 s background syncer eat the connect window, so
          the boot restore only rebuilt the first collection and the UI
          inspector reported "not connected" for the rest.
        """
        client = self._client
        if client is not None:
            return client
        if not self._url:
            return None
        with self._connect_guard:
            if self._client is not None:  # another thread connected while we waited
                return self._client
            now = time.monotonic()
            if not self._injected and now - self._last_connect_attempt < self.connect_backoff_s:
                return None  # unreachable a moment ago: retry on the next window
            self._last_connect_attempt = now
            try:
                import redis  # optional dependency: pip install 'colabdeploy-gateway[redis]'

                client = redis.Redis.from_url(
                    self._url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    # Reap a connection Render's NAT quietly killed, so the first
                    # write after an idle period is not the one that fails.
                    health_check_interval=30,
                )
                client.ping()
            except Exception as exc:  # any failure degrades to file-only
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._warn_connect()
                return None
            self._client = client
            self._last_error = ""
            self._connect_warned = False
            logger.info("registry mirror: connected to Redis (%s)", describe_url(self._url)["host"] or "redis")
            return client

    def _warn_connect(self) -> None:
        """Log the reason once per outage, never per call."""
        if self._connect_warned:
            return
        logger.warning(
            "registry mirror: Redis unreachable (%s); continuing on the "
            "instance disk and retrying in the background",
            self._last_error,
        )
        self._connect_warned = True

    def _drop_client(self, exc: Exception) -> None:
        """Retire a client that just failed, so the next call reconnects."""
        self._last_error = f"{type(exc).__name__}: {exc}"
        stale = self._client
        if not self._injected:
            self._client = None
            if stale is not None:
                with contextlib.suppress(Exception):  # a broken socket is expected here
                    stale.close()
            # Back off from now on, so a dead Redis is not re-probed per request.
            self._last_connect_attempt = time.monotonic()
        self._warn_once_write_failure()

    def _warn_once_write_failure(self) -> None:
        if self._connect_warned:
            return
        logger.warning(
            "registry mirror: Redis write failed (%s); will reconnect and retry",
            self._last_error,
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
        self.restored_at_boot = restored
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

        In write-through mode (``sync_interval_s == 0``) every mutation flushes
        inline instead, and this thread has one job left: catch up on keys a
        failed inline flush queued, which nothing else would ever retry.
        """
        write_through = self.sync_interval_s <= 0
        period = self.connect_backoff_s if write_through else self.sync_interval_s
        next_sync = time.monotonic() + period
        while not self._closing:
            remaining = max(0.0, next_sync - time.monotonic())
            self._wakeup.wait(remaining)
            self._wakeup.clear()
            if self._closing:
                break
            if not write_through:
                if time.monotonic() < next_sync:
                    # Woken by a mutation. `next_sync` is deliberately left
                    # alone: rescheduling on every heartbeat would starve the
                    # sync forever under a steady write stream.
                    continue
            elif not self._has_pending():
                # In write-through mode a mutation flushes inline, so an early
                # wakeup with an empty queue is the normal case: re-arm.
                next_sync = time.monotonic() + period
                continue
            self._sync_once()
            self._maybe_keepalive()
            next_sync = time.monotonic() + period

    def _has_pending(self) -> bool:
        with self._dirty_guard:
            return bool(self._dirty)

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
        """Flush pending mirror writes, stop the sync thread, release the socket."""
        self._closing = True
        self._wakeup.set()
        thread = self.__dict__.get("_flusher")
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)
        self._sync_once()
        if not self._injected:
            client, self._client = self._client, None
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()

    def sync_stats(self) -> dict[str, int]:
        """Diagnostics: how many records are queued for the mirror."""
        with self._dirty_guard:
            return {"pending": len(self._dirty), "collections": len(self._cache)}

    def reconnect(self) -> dict[str, Any]:
        """Drop the backoff and try to connect now (the UI's retry button).

        Without this a user who just woke an archived Upstash database has to
        wait out the 30 s window before the panel admits Redis is reachable.
        """
        self._last_connect_attempt = -60.0
        self._connect_warned = False
        client = self._ensure_client()
        if client is not None:
            self._sync_once()
        return self.status()

    def flush(self) -> dict[str, Any]:
        """Push the queued mirror writes now, and say honestly whether they landed.

        A sync that no-ops because Redis is unreachable must not be reported as
        "synced" — that is the difference between a UI that lies and one that
        points at the failing hop.
        """
        before = self.sync_stats()["pending"]
        client = self._ensure_client()
        if client is None:
            status = self.status()
            status["message"], status["hint"] = mirror_advice(status)
            return {
                "synced": False,
                "reason": status.get("reason") or status.get("state") or "unreachable",
                "message": status.get("message") or "",
                "hint": status.get("hint") or "",
                "stats": {"pending": before, "collections": len(self._cache)},
            }
        self._sync_once()
        after = self.sync_stats()["pending"]
        return {
            "synced": after == 0,
            "pushed": max(0, before - after),
            "stats": {"pending": after, "collections": len(self._cache)},
            "view": self.inspect(limit=1),
        }

    def status(self) -> dict[str, Any]:
        """Mirror health, without a Redis round-trip (safe to call per request)."""
        connected = self._client is not None
        retry_in = 0.0
        if not connected and self._url:
            wait = self.connect_backoff_s - (time.monotonic() - self._last_connect_attempt)
            retry_in = round(max(0.0, wait), 1)
        state = "connected" if connected else ("backoff" if retry_in > 0 else "retry")
        if not self._url:
            state = "disabled"
        elif self._url_problem:
            state = "invalid_url"
        elif self._last_error and not connected:
            state = "unreachable"
        return {
            "backend": "upstash",
            "enabled": True,
            "configured": bool(self._url),
            "connected": connected,
            "state": state,
            "reason": "bad_url" if self._url_problem else ("" if connected else classify_error(self._last_error)),
            "url_ok": self.url_ok,
            "namespace": self.namespace,
            "host": describe_url(self._url)["host"] if self._url else "",
            "sync_interval_s": self.sync_interval_s,
            "keepalive_s": self.keepalive_s,
            "pending": self.sync_stats()["pending"],
            "restored": int(getattr(self, "restored_at_boot", 0)),
            "last_error": self._last_error,
            "retry_in_s": retry_in,
        }

    # ------------------------------------------------------------------ #
    # Upstash inspector (what the UI Cache page shows)
    # ------------------------------------------------------------------ #
    def inspect(self, limit: int = 200, collection: str | None = None) -> dict[str, Any]:
        """List live Redis keys. This is the Upstash view — not the inference cache."""
        client = self._ensure_client()
        empty: dict[str, Any] = {
            "connected": False,
            "backend": "upstash",
            "namespace": self.namespace,
            "entries": 0,
            "size_bytes": 0,
            "pending": self.sync_stats()["pending"],
            "keys": [],
            "collections": {},
        }
        if client is None:
            # "not connected" is only useful with a reason attached: the URL may
            # be fine and the network/TLS/Upstash-restore the actual problem.
            status = self.status()
            empty.update(status)
            empty["connected"] = False  # a cached client must never claim "up" here
            empty["message"], empty["hint"] = mirror_advice(status)
            return empty
        try:
            pattern = (
                f"{self.namespace}:{collection}:*"
                if collection
                else f"{self.namespace}:*"
            )
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
                rest = full_key[len(self.namespace) + 1 :] if full_key.startswith(self.namespace + ":") else full_key
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
            return {
                **self.status(),
                "connected": True,
                "message": "",
                "hint": "",
                "entries": len(redis_keys),
                "shown": len(keys_out),
                "size_bytes": total_bytes,
                "keys": keys_out,
                "collections": collections,
            }
        except Exception as exc:
            # A Redis error mid-scan is a connection-state event, so report it
            # as one: drop the client, refresh the diagnostics, keep the shape.
            self._drop_client(exc)
            status = self.status()
            empty.update(status)
            empty["connected"] = False
            empty["message"], empty["hint"] = mirror_advice(status)
            empty["error"] = str(exc)
            return empty

    def inspect_key(self, full_key: str) -> dict[str, Any] | None:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            raw = client.get(full_key)
        except Exception as exc:
            self._drop_client(exc)
            return None
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
            return bool(client.delete(full_key))
        except Exception as exc:
            self._drop_client(exc)
            return False


#: Actionable text per failure mode, so the Cache page can say what to do
#: instead of "set the variables you already set". ``reason`` is the stable id
#: (tests and docs reference it); ``message`` is the one-liner; ``hint`` the fix.
_MIRROR_ADVICE: dict[str, tuple[str, str]] = {
    "no_url": (
        "This gateway process has no CDP_REDIS_URL in its environment.",
        "On Render a secret added after the first deploy is only injected on the *next* "
        "deploy: Environment → Add Secret Value, then Manual Deploy → 'Clear cache & "
        "deploy'. render.yaml declares CDP_REDIS_URL with sync:false, so it is never "
        "pre-filled.",
    ),
    "bad_url": (
        "CDP_REDIS_URL is not a Redis connection URL.",
        "Use the 'Redis URL' from Upstash (rediss://default:<token>@host.upstash.io:6379). "
        "The 'REST URL' (https://…) is a different API and the read-only replica cannot be "
        "written to. Strip any quote characters Render may have kept around the value.",
    ),
    "library_missing": (
        "The gateway image has no 'redis' Python client, so the mirror cannot start.",
        "The Dockerfile installs './apps/gateway[redis]'. A custom Build/Start command that "
        "runs plain 'pip install -e ./apps/gateway' or a bare requirements.txt drops the "
        "extra — add redis>=5.0 there.",
    ),
    "backend_file": (
        "CDP_REGISTRY_BACKEND is not 'redis' (the registry lives on this container's disk only).",
        "Set CDP_REGISTRY_BACKEND=redis. Note it is read once at process start: restart the "
        "service after changing any CDP_* variable.",
    ),
    "tls": (
        "Redis answered, but the TLS handshake failed.",
        "Upstash only accepts rediss:// on port 6379. If your egress path terminates TLS, "
        "append ?ssl_cert_reqs=none to the URL.",
    ),
    "dns": (
        "The gateway cannot resolve the Redis host.",
        "Check the hostname and that the service has outbound DNS: Upstash hosts live in "
        "the region you picked (…upstash.io).",
    ),
    "timeout": (
        "The TCP connect to Redis timed out.",
        "Free-tier hosts can block outbound 6379. Test from the same region, or fall back to "
        "disk-only persistence (leave CDP_REDIS_URL empty) and re-attach the runtime after a recycle.",
    ),
    "auth": (
        "Redis rejected the credentials.",
        "Upstash expects the user 'default' and the token after the ':' as the password. "
        "Regenerate the token in the Upstash console if it was rotated.",
    ),
    "max_clients": (
        "Upstash refused the connection: too many clients.",
        "The free tier caps concurrent connections. Restart the gateway (or delete idle "
        "connections in the Upstash console) — the mirror now reuses a single pooled client.",
    ),
    "restored": (
        "Upstash is bringing the database back online.",
        "Idle free databases are archived and take ~1 min to restore. Open the Upstash "
        "console once, then press 'Reconnect now'.",
    ),
    "backoff": (
        "Redis was unreachable a moment ago; the gateway retries in the background.",
        "Press 'Reconnect now' after you fixed the URL or woke the database, instead of "
        "waiting out the 30 s backoff.",
    ),
    "init_failed": (
        "The mirror could not be built in this gateway process.",
        "The gateway log has the exception. The usual cause is a mismatched 'redis' client "
        "version — pin redis>=5.0 and redeploy.",
    ),
    "unknown": (
        "Redis is configured but not answering.",
        "Look for a 'registry mirror: Redis unreachable' line in the gateway logs "
        "(Logs page, or Render → Events) — it carries the exact client error.",
    ),
}

#: Substrings that identify a client error without importing the redis package.
_ERROR_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("max_clients", ("max number of clients", "too many connections", "mcrouter", "maximum number of clients")),
    ("restored", ("restored", "archived", "is not running", "loading dataset")),
    ("auth", ("authentication", "invalid password", "noauth", "wrongpass", "invalid username-password")),
    ("tls", ("ssl", "tls", "certificate", "handshake", "unknown scheme")),
    ("dns", ("name or service not known", "getaddrinfo", "nodename nor servname", "temporary failure in name resolution")),
    ("timeout", ("timed out", "timeout", "connection refused", "unreachable", "network is unreachable")),
)


def classify_error(last_error: str) -> str:
    """Map a redis-py exception string onto one of ``_MIRROR_ADVICE``."""
    blob = (last_error or "").lower()
    if not blob:
        return "unknown"
    for reason, needles in _ERROR_SIGNATURES:
        if any(needle in blob for needle in needles):
            return reason
    return "unknown"


def mirror_advice(status: dict[str, Any]) -> tuple[str, str]:
    """``(message, hint)`` for the Cache page banner, from a status payload."""
    reason = status.get("reason") or ""
    if not reason:
        if not status.get("configured"):
            reason = "no_url"
        elif not status.get("url_ok", True):
            reason = "bad_url"
        elif status.get("state") == "backoff":
            reason = "backoff"
        else:
            reason = classify_error(status.get("last_error", ""))
    return _MIRROR_ADVICE.get(reason, _MIRROR_ADVICE["unknown"])


def registry_status(store: JsonStore, settings: Any) -> dict[str, Any]:
    """Health of the registry mirror, for /api/upstash/* and /healthz.

    Works for both stores: a :class:`RedisSnapshotStore` reports its live
    connection state, a fallback ``JsonStore`` reports *why* the mirror was
    disabled at boot (``reason`` is recorded by :func:`build_registry_store`).
    """
    if isinstance(store, RedisSnapshotStore):
        out = store.status()
        out["message"], out["hint"] = mirror_advice(out) if not out["connected"] else ("", "")
        return out
    backend = (getattr(settings, "registry_backend", "file") or "file").strip().lower()
    url = (getattr(settings, "redis_url", "") or "").strip()
    reason = getattr(store, "registry_reason", "") or ("backend_file" if backend != "redis" else "no_url")
    out = {
        "backend": "upstash" if backend == "redis" else "file",
        "enabled": backend == "redis",
        "configured": bool(url),
        "connected": False,
        "state": "disabled",
        "reason": reason,
        "namespace": "cdp:registry:v1",
        "host": describe_url(url)["host"],
        "url_ok": describe_url(url)["scheme_ok"] if url else False,
        "pending": 0,
        "last_error": getattr(store, "registry_note", ""),
        "retry_in_s": 0.0,
    }
    out["message"], out["hint"] = mirror_advice(out)
    return out


def _fallback_store(settings: Any, reason: str, note: str) -> JsonStore:
    """A plain file store that remembers why the mirror was skipped."""
    store = JsonStore(settings.data_dir / "store")
    store.registry_reason = reason  # type: ignore[attr-defined]
    store.registry_note = note  # type: ignore[attr-defined]
    return store


def build_registry_store(settings: Any) -> JsonStore:
    """Factory used by the app container: file store, or Redis-mirrored store.

    Never raises: if the ``redis`` package is missing or the URL is empty the
    gateway falls back to the plain file store with a warning, which is the
    pre-Redis behaviour. Each fallback records a ``reason`` so the Cache page
    can explain itself instead of blaming variables that are already set.
    """
    backend = (getattr(settings, "registry_backend", "file") or "file").strip().lower()
    url = (getattr(settings, "redis_url", "") or "").strip()
    if backend != "redis" or not url:
        if backend == "redis":
            logger.warning(
                "CDP_REGISTRY_BACKEND=redis but CDP_REDIS_URL is empty; "
                "the registry stays on the instance disk"
            )
            return _fallback_store(settings, "no_url", "CDP_REDIS_URL is empty")
        return _fallback_store(settings, "backend_file", "CDP_REGISTRY_BACKEND=file")
    try:
        import redis  # noqa: F401 - optional dependency, probed up front
    except ImportError:
        logger.warning(
            "CDP_REGISTRY_BACKEND=redis but the redis client is not installed; "
            "run pip install 'colabdeploy-gateway[redis]'. "
            "Continuing with disk-only persistence."
        )
        return _fallback_store(settings, "library_missing", "the 'redis' python package is not installed")
    problem = describe_url(url)["problem"]
    if problem:
        logger.warning("registry mirror disabled: %s", problem)
        return _fallback_store(settings, "bad_url", problem)
    try:
        return RedisSnapshotStore(
            settings.data_dir / "store",
            redis_url=url,
            sync_interval_s=float(getattr(settings, "registry_sync_s", 30.0)),
            keepalive_s=float(getattr(settings, "registry_keepalive_s", 86400.0)),
            connect_backoff_s=float(getattr(settings, "registry_connect_backoff_s", CONNECT_BACKOFF_S)),
        )
    except Exception as exc:  # a mirror must never block boot
        logger.warning(
            "registry mirror disabled (%s); install it with "
            "pip install 'colabdeploy-gateway[redis]'",
            exc,
        )
        return _fallback_store(settings, "init_failed", str(exc))
