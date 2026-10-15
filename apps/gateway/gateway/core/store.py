"""A tiny JSON-document store with atomic writes and per-collection locking.

Why not SQLite for v1? The control plane has to run on the smallest free tier with
zero setup, and the entire dataset (models, endpoints, runtimes, jobs) is measured
in hundreds of records, not millions. The repository interface in
``gateway.core.repository`` is what the rest of the app codes against, so swapping
in Postgres later touches one module.

Durability rules:
  * writes go to ``*.tmp`` then ``os.replace`` - a crash mid-write cannot corrupt;
  * every mutation bumps ``revision`` so the agent's desired-state poll is cheap;
  * reads never block writers for long (short critical sections, whole-file swap).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")


class JsonStore:
    """Thread-safe, file-backed collection store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._revision = 0

    # ------------------------------------------------------------------ #
    def _lock(self, name: str) -> threading.RLock:
        with self._guard:
            if name not in self._locks:
                self._locks[name] = threading.RLock()
            return self._locks[name]

    def _path(self, collection: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in collection)
        return self.root / f"{safe}.json"

    def _load(self, collection: str) -> dict[str, Any]:
        cached = self._cache.get(collection)
        if cached is not None:
            return cached
        path = self._path(collection)
        data: dict[str, Any] = {"items": {}, "revision": 0}
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict) and isinstance(loaded.get("items"), dict):
                    data = loaded
            except (json.JSONDecodeError, OSError):
                # Corrupt file: quarantine it instead of losing the rest of the system.
                try:
                    path.rename(path.with_suffix(f".corrupt-{int(time.time())}.json"))
                except OSError:  # pragma: no cover
                    pass
        self._cache[collection] = data
        return data

    def _flush(self, collection: str) -> None:
        data = self._cache[collection]
        path = self._path(collection)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, default=str, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    # ------------------------------------------------------------------ #
    @property
    def revision(self) -> int:
        return self._revision

    def all(self, collection: str) -> list[dict[str, Any]]:
        with self._lock(collection):
            return list(self._load(collection)["items"].values())

    def get(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._lock(collection):
            return self._load(collection)["items"].get(key)

    def put(self, collection: str, key: str, value: dict[str, Any]) -> dict[str, Any]:
        with self._lock(collection):
            data = self._load(collection)
            data["items"][key] = value
            data["revision"] = int(data.get("revision", 0)) + 1
            self._revision += 1
            self._flush(collection)
            return value

    def update(
        self, collection: str, key: str, patch: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Read-modify-write under the collection lock (no lost updates)."""
        with self._lock(collection):
            data = self._load(collection)
            current = data["items"].get(key)
            if current is None:
                return None
            merged = patch(current) if callable(patch) else {**current, **patch}
            data["items"][key] = merged
            data["revision"] = int(data.get("revision", 0)) + 1
            self._revision += 1
            self._flush(collection)
            return merged

    def delete(self, collection: str, key: str) -> bool:
        with self._lock(collection):
            data = self._load(collection)
            if key not in data["items"]:
                return False
            del data["items"][key]
            data["revision"] = int(data.get("revision", 0)) + 1
            self._revision += 1
            self._flush(collection)
            return True

    def find(self, collection: str, **equals: Any) -> list[dict[str, Any]]:
        return [r for r in self.all(collection) if all(r.get(k) == v for k, v in equals.items())]

    def find_one(self, collection: str, **equals: Any) -> dict[str, Any] | None:
        for record in self.all(collection):
            if all(record.get(k) == v for k, v in equals.items()):
                return record
        return None

    def count(self, collection: str) -> int:
        with self._lock(collection):
            return len(self._load(collection)["items"])

    def clear(self, collection: str | None = None) -> None:
        collections: Iterable[str] = [collection] if collection else list(self._cache)
        for name in collections:
            with self._lock(name):
                self._cache[name] = {"items": {}, "revision": 0}
                self._flush(name)

    def collections(self) -> list[str]:
        return sorted({p.stem for p in self.root.glob("*.json")})
