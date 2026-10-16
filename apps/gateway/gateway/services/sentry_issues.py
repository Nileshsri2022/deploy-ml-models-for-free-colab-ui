"""Read-only, allowlisted issue summaries from Sentry for the Logs UI."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from gateway.core.sentry import scrub_sensitive_text

logger = logging.getLogger("gateway.sentry_issues")


class SentryUpstreamError(RuntimeError):
    """Sentry could not return the configured project's issue list."""


class SentryIssueService:
    """Fetch and briefly cache a limited, safe subset of Sentry issue fields."""

    def __init__(
        self,
        *,
        token: str = "",
        org_slug: str = "",
        project_slug: str = "",
        api_base_url: str = "https://sentry.io",
        cache_ttl_s: float = 30.0,
        timeout_s: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.token = token.strip()
        self.org_slug = org_slug.strip()
        self.project_slug = project_slug.strip()
        self.api_base_url = api_base_url.strip().rstrip("/")
        try:
            parsed_base = urlsplit(self.api_base_url)
            valid_base = (
                parsed_base.scheme == "https"
                and parsed_base.port != 0
                and bool(parsed_base.hostname)
                and not parsed_base.username
                and not parsed_base.password
                and not parsed_base.query
                and not parsed_base.fragment
            )
        except (ValueError, TypeError):
            valid_base = False
        if not valid_base:
            self.api_base_url = ""
        self.cache_ttl_s = max(0.0, float(cache_ttl_s))
        self.timeout_s = max(0.1, float(timeout_s))
        self._client = client
        self._owns_client = client is None
        self._cache: dict[int, tuple[float, dict[str, Any]]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.token and self.org_slug and self.project_slug and self.api_base_url)

    async def list_issues(self, limit: int = 50) -> dict[str, Any]:
        limit = min(100, max(1, int(limit)))
        if not self.configured:
            return {"configured": False, "issues": [], "fetched_at": None, "cached": False}

        now = time.monotonic()
        cached = self._cache.get(limit)
        if cached and now - cached[0] < self.cache_ttl_s:
            result = dict(cached[1])
            result["cached"] = True
            return result

        url = (
            f"{self.api_base_url}/api/0/projects/"
            f"{quote(self.org_slug, safe='')}/{quote(self.project_slug, safe='')}/issues/"
        )
        try:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=self.timeout_s)
            response = await self._client.get(
                url,
                headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
                params={"limit": limit, "statsPeriod": "24h"},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            raw_issues = response.json()
            if not isinstance(raw_issues, list):
                raise SentryUpstreamError("Sentry returned an unexpected issue-list response")
            issues = [issue for raw in raw_issues if (issue := self._safe_issue(raw)) is not None]
            result = {
                "configured": True,
                "issues": issues,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "cached": False,
            }
            self._cache[limit] = (time.monotonic(), result)
            return result
        except SentryUpstreamError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Do not expose upstream response bodies or request headers, which may
            # contain details/operators don't need in the browser.
            logger.warning("Sentry issue-list request failed: %s", type(exc).__name__)
            raise SentryUpstreamError("Could not fetch issues from Sentry") from exc

    def _safe_issue(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        issue_id = raw.get("id")
        if not isinstance(issue_id, (str, int)):
            return None
        return {
            "id": str(issue_id),
            "short_id": _safe_text(raw.get("shortId"), 80),
            "title": _safe_text(raw.get("title"), 500) or "Untitled issue",
            "culprit": _safe_text(raw.get("culprit"), 300),
            "level": _safe_text(raw.get("level"), 32) or "error",
            "status": _safe_text(raw.get("status"), 32) or "unresolved",
            "count": _safe_count(raw.get("count")),
            "first_seen": _safe_text(raw.get("firstSeen"), 64),
            "last_seen": _safe_text(raw.get("lastSeen"), 64),
            "permalink": self._safe_permalink(raw.get("permalink")),
        }

    def _safe_permalink(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        try:
            link = urlsplit(value)
            if link.port == 0:
                return None
            base = urlsplit(self.api_base_url)
            if link.username or link.password:
                return None
            same_host = link.netloc == base.netloc
            base_is_sentry_cloud = (base.hostname or "").endswith(".sentry.io")
            link_is_sentry_cloud = (link.hostname or "") == "sentry.io" or (
                link.hostname or ""
            ).endswith(".sentry.io")
            sentry_cloud_host = base_is_sentry_cloud and link_is_sentry_cloud
            if link.scheme != "https" or not (same_host or sentry_cloud_host) or not link.path:
                return None
            return f"{link.scheme}://{link.netloc}{link.path}"
        except (ValueError, TypeError):
            return None

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()


def _safe_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    return scrub_sensitive_text(value[:limit])


def _safe_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
