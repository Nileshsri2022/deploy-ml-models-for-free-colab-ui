"""Sentry initialization and defensive event scrubbing for the gateway."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger("gateway.sentry")
_initialized_dsn: str | None = None

_SECRET_KEY = re.compile(
    r"(?:authorization|cookie|password|passwd|secret|token|api[_-]?key|body|payload|inputs?|outputs?)",
    re.IGNORECASE,
)
_INLINE_SECRET = re.compile(
    r"(?i)\b(authorization|cookie|x-runtime-token|api_key|token|password|secret)\s*[:=]\s*([^\s&,;]+)"
)
_REDIS_CREDENTIALS = re.compile(r"(?i)(rediss?://)[^/@\s]+@")
_CDP_TOKEN = re.compile(r"\bCDP_(?:API|RT)_[A-Za-z0-9_-]+\b")


def scrub_sensitive_text(value: str) -> str:
    """Remove common credentials from exception messages and log text."""
    value = _REDIS_CREDENTIALS.sub(r"\1[redacted]@", value)
    value = _CDP_TOKEN.sub("[redacted-token]", value)
    return _INLINE_SECRET.sub(r"\1=[redacted]", value)


def _scrub_url(value: str) -> str:
    """Keep only the URL origin and path; query strings can contain API keys."""
    try:
        parts = urlsplit(value)
        if not parts.scheme and not parts.netloc:
            return value.split("?", 1)[0].split("#", 1)[0]
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except (ValueError, TypeError):
        return value.split("?", 1)[0].split("#", 1)[0]


def _scrub_value(value: Any, *, key: str = "") -> Any:
    lowered_key = key.lower()
    if _SECRET_KEY.search(key) or "query" in lowered_key or lowered_key in {"vars", "variables"}:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): _scrub_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(item) for item in value)
    if isinstance(value, str):
        if lowered_key in {"url", "http.url", "url.full", "http.target"}:
            return _scrub_url(value)
        return scrub_sensitive_text(value)
    return value


def scrub_sentry_event(event: dict[str, Any], hint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strip credentials, request bodies, and user identifiers from an event.

    Sentry is used for stack traces and safe diagnostics, not for collecting raw
    requests. In particular, `/api/logs/stream` currently puts `api_key` in its
    query string because EventSource cannot send an Authorization header.
    """
    del hint  # the scrubber deliberately does not inspect exception locals

    request = event.get("request")
    if isinstance(request, dict):
        if isinstance(request.get("url"), str):
            request["url"] = _scrub_url(request["url"])
        # Header values, query strings, cookies, and body data are unnecessary
        # for the diagnostics we want and are safest omitted entirely.
        for key in ("headers", "cookies", "data", "query_string", "env"):
            request.pop(key, None)

    # Do not attach user identity/IP data to error reports.
    event.pop("user", None)

    if "breadcrumbs" in event:
        event["breadcrumbs"] = _scrub_value(event["breadcrumbs"])
    if "extra" in event:
        event["extra"] = _scrub_value(event["extra"])
    if "contexts" in event:
        event["contexts"] = _scrub_value(event["contexts"])
    if "tags" in event:
        event["tags"] = _scrub_value(event["tags"])
    if "spans" in event:
        event["spans"] = _scrub_value(event["spans"])

    exception = event.get("exception")
    values = exception.get("values", []) if isinstance(exception, dict) else []
    has_exception = isinstance(values, list) and bool(values)
    if isinstance(values, list):
        for item in values:
            if isinstance(item, dict):
                if isinstance(item.get("value"), str):
                    # Exception messages may embed an input/output value. Keep
                    # the exception type and stack frames, but not free text.
                    item["value"] = "[exception details redacted]"
                stacktrace = item.get("stacktrace")
                frames = stacktrace.get("frames", []) if isinstance(stacktrace, dict) else []
                for frame in frames:
                    if isinstance(frame, dict):
                        frame.pop("vars", None)

    for key in ("message", "transaction"):
        if isinstance(event.get(key), str):
            event[key] = (
                "[exception details redacted]"
                if key == "message" and has_exception
                else scrub_sensitive_text(event[key])
            )

    logentry = event.get("logentry")
    if isinstance(logentry, dict):
        if isinstance(logentry.get("message"), str):
            logentry["message"] = (
                "[exception details redacted]"
                if has_exception
                else scrub_sensitive_text(logentry["message"])
            )
        if isinstance(logentry.get("params"), (dict, list, tuple)):
            logentry["params"] = _scrub_value(logentry["params"])
    return event


def initialize_sentry(settings: Any) -> bool:
    """Initialize Sentry only when configured; never make startup depend on it."""
    global _initialized_dsn
    dsn = str(getattr(settings, "sentry_dsn", "") or "").strip()
    if not dsn:
        return False
    if _initialized_dsn == dsn:
        return True
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=(getattr(settings, "sentry_environment", "") or "").strip()
            or getattr(settings, "env", "production"),
            release=f"colabdeploy@{getattr(settings, 'version', 'unknown')}",
            integrations=[FastApiIntegration()],
            disabled_integrations=[LoggingIntegration],
            data_collection={
                "user_info": False,
                "cookies": {"mode": "off"},
                "http_headers": {"request": {"mode": "off"}},
                "http_bodies": [],
                "url_query_params": {"mode": "off"},
                "graphql": {"document": False, "variables": False},
                "gen_ai": {"inputs": False, "outputs": False},
                "database_query_data": False,
                "queues": False,
                "stack_frame_variables": False,
                "frame_context_lines": 0,
            },
            max_request_body_size="never",
            include_local_variables=False,
            include_source_context=False,
            traces_sample_rate=_sample_rate(getattr(settings, "sentry_traces_sample_rate", 0.0)),
            trace_propagation_targets=[],
            before_send=scrub_sentry_event,
            before_send_transaction=scrub_sentry_event,
        )
        _initialized_dsn = dsn
    except ImportError:
        logger.warning("Sentry DSN is configured but sentry-sdk is not installed")
        return False
    except Exception:
        logger.exception("Sentry initialization failed; continuing without Sentry")
        return False
    return True


def capture_exception(
    error: BaseException,
    *,
    tags: dict[str, str] | None = None,
) -> None:
    """Capture an exception without allowing Sentry failures to affect requests."""
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in (tags or {}).items():
                scope.set_tag(key, value)
            sentry_sdk.capture_exception(error)
    except Exception:
        logger.debug("Unable to report exception to Sentry", exc_info=True)


def capture_message(
    message: str,
    *,
    level: str = "warning",
    tags: dict[str, str] | None = None,
) -> None:
    """Capture a sanitized operational message, e.g. one Redis outage event."""
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in (tags or {}).items():
                scope.set_tag(key, value)
            sentry_sdk.capture_message(scrub_sensitive_text(message), level=level)
    except Exception:
        logger.debug("Unable to report operational message to Sentry", exc_info=True)


def _sample_rate(value: Any) -> float:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, rate))
