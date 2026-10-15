"""Persistent, per-API-key daily inference request quotas."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from gateway.core.store import JsonStore


class DailyInferenceQuota:
    def __init__(self, store: JsonStore, limit: int) -> None:
        self.store = store
        self.limit = limit
        self._lock = threading.Lock()

    def consume(self, key_id: str) -> dict[str, Any]:
        """Consume one HTTP inference request; a non-positive limit disables quotas."""
        if self.limit <= 0:
            return {"limited": False}

        now = datetime.now(timezone.utc)
        day = now.date().isoformat()
        record_id = key_id
        reset_at = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
        retry_after = max(1, int((reset_at - now).total_seconds()))

        # JsonStore is a single-process store; this lock makes read/check/write atomic
        # for concurrent requests handled by this gateway process.
        with self._lock:
            record = self.store.get("inference_usage", record_id)
            used = int((record or {}).get("requests", 0)) if (record or {}).get("day") == day else 0
            if used >= self.limit:
                return {
                    "limited": True,
                    "limit": self.limit,
                    "remaining": 0,
                    "retry_after": retry_after,
                }
            usage = {"key_id": key_id, "day": day, "requests": used + 1}
            if record is None:
                self.store.put("inference_usage", record_id, usage)
            else:
                self.store.update("inference_usage", record_id, usage)
            return {
                "limited": False,
                "limit": self.limit,
                "remaining": self.limit - used - 1,
                "retry_after": retry_after,
            }
