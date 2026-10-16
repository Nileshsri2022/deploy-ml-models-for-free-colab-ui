"""Sentry privacy, gateway API, and issue-list behavior."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import gateway.core.sentry as sentry_module
from gateway.core.sentry import scrub_sentry_event
from gateway.services.sentry_issues import SentryIssueService


def test_sentry_scrubber_removes_request_secrets_and_bodies() -> None:
    event: dict[str, Any] = {
        "request": {
            "url": "https://example.test/api/logs/stream?api_key=CDP_API_private-secret",
            "headers": {"Authorization": "Bearer top-secret"},
            "cookies": {"session": "cookie-secret"},
            "data": {"inputs": {"text": "private model input"}},
            "query_string": "api_key=CDP_API_private-secret",
        },
        "user": {"id": "private-user", "ip_address": "192.0.2.10"},
        "exception": {
            "values": [
                {
                    "value": "failed with password=hunter2; model input=private model input",
                    "stacktrace": {"frames": [{"vars": {"text": "private local model data"}}]},
                }
            ]
        },
        "spans": [
            {
                "description": "POST /v1/models/predict",
                "data": {
                    "http.request.body": "private model output",
                    "http.url": "https://example.test/v1/models/predict?api_key=CDP_API_span-secret",
                    "url.query": "api_key=CDP_API_span-secret",
                },
            }
        ],
        "breadcrumbs": [
            {
                "message": "GET /api/logs/stream",
                "data": {"url": "/api/logs/stream?api_key=CDP_API_private-secret"},
            }
        ],
        "extra": {"runtime_token": "CDP_RT_private-token", "safe": "ok"},
    }

    cleaned = scrub_sentry_event(event)
    encoded = json.dumps(cleaned)

    assert "CDP_API_private-secret" not in encoded
    assert "CDP_RT_private-token" not in encoded
    assert "top-secret" not in encoded
    assert "cookie-secret" not in encoded
    assert "private model input" not in encoded
    assert "hunter2" not in encoded
    assert "private model output" not in encoded
    assert "private local model data" not in encoded
    assert "CDP_API_span-secret" not in encoded
    assert "private-user" not in encoded
    assert cleaned["exception"]["values"][0]["value"] == "[exception details redacted]"
    assert cleaned["request"]["url"] == "https://example.test/api/logs/stream"
    assert "headers" not in cleaned["request"]
    assert "data" not in cleaned["request"]
    assert "user" not in cleaned


def test_sentry_initialization_disables_sensitive_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration

    monkeypatch.setattr(sentry_module, "_initialized_dsn", None)
    captured: dict[str, Any] = {}

    def fake_init(**options: Any) -> None:
        captured.update(options)

    monkeypatch.setattr(sentry_sdk, "init", fake_init)
    settings = SimpleNamespace(
        sentry_dsn="https://public@example.test/1",
        sentry_environment="test",
        env="test",
        version="0.1.0",
        sentry_traces_sample_rate=0,
    )

    assert sentry_module.initialize_sentry(settings) is True
    assert captured["data_collection"]["http_bodies"] == []
    assert captured["data_collection"]["stack_frame_variables"] is False
    assert captured["data_collection"]["gen_ai"] == {"inputs": False, "outputs": False}
    assert captured["max_request_body_size"] == "never"
    assert captured["include_local_variables"] is False
    assert captured["include_source_context"] is False
    assert captured["trace_propagation_targets"] == []
    assert LoggingIntegration in captured["disabled_integrations"]
    assert "send_default_pii" not in captured


def test_unhandled_gateway_exception_is_reported_without_changing_response_contract(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    import gateway.main as gateway_main

    captured: list[tuple[BaseException, dict[str, str]]] = []
    monkeypatch.setattr(
        gateway_main,
        "capture_exception",
        lambda error, *, tags=None: captured.append((error, tags or {})),
    )

    async def fail_for_test() -> None:
        raise RuntimeError("unexpected test failure")

    app = gateway_main.create_app(state)
    app.add_api_route("/_test/unhandled", fail_for_test)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/_test/unhandled", headers={"X-Request-Id": "test-request-42"})

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL"
    assert response.json()["error"]["message"] == "unexpected test failure"
    assert len(captured) == 1
    assert isinstance(captured[0][0], RuntimeError)
    assert captured[0][1]["request_id"] == "test-request-42"
    assert captured[0][1]["http.method"] == "GET"
    assert captured[0][1]["http.route"] == "/_test/unhandled"


def test_sentry_issue_service_returns_only_allowlisted_summary_and_caches() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json=[
                {
                    "id": "42",
                    "shortId": "CDP-42",
                    "title": "Model adapter raised an error",
                    "culprit": "gateway.services.inference",
                    "level": "error",
                    "status": "unresolved",
                    "count": "7",
                    "firstSeen": "2026-10-15T10:00:00Z",
                    "lastSeen": "2026-10-16T10:00:00Z",
                    "permalink": "https://sentry.io/organizations/acme/issues/42/",
                    "metadata": {"secret": "not returned"},
                    "request": {"data": "not returned"},
                }
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    service = SentryIssueService(
        token="read-token",
        org_slug="acme",
        project_slug="control-plane",
        client=client,
    )

    async def run() -> tuple[dict[str, Any], dict[str, Any]]:
        first = await service.list_issues(limit=500)
        second = await service.list_issues(limit=500)
        await client.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert first["configured"] is True
    assert len(first["issues"]) == 1
    assert first["issues"][0] == {
        "id": "42",
        "short_id": "CDP-42",
        "title": "Model adapter raised an error",
        "culprit": "gateway.services.inference",
        "level": "error",
        "status": "unresolved",
        "count": 7,
        "first_seen": "2026-10-15T10:00:00Z",
        "last_seen": "2026-10-16T10:00:00Z",
        "permalink": "https://sentry.io/organizations/acme/issues/42/",
    }
    assert first["cached"] is False
    assert second["cached"] is True
    assert len(calls) == 1
    assert calls[0].url.params["limit"] == "100"
    assert calls[0].headers["authorization"] == "Bearer read-token"
    assert "not returned" not in json.dumps(first)
    assert "read-token" not in json.dumps(first)


def test_sentry_issue_service_rejects_unsafe_api_base_urls() -> None:
    for api_base_url in (
        "http://sentry.example.test",
        "https://user:password@sentry.example.test",
        "https://sentry.example.test/?token=private",
        "https://sentry.example.test/#private",
        "https://sentry.example.test:not-a-port",
        "https://[broken-ipv6",
    ):
        service = SentryIssueService(
            token="read-token",
            org_slug="acme",
            project_slug="control-plane",
            api_base_url=api_base_url,
        )
        assert service.configured is False
        assert asyncio.run(service.list_issues())["configured"] is False
        assert service._client is None


def test_unconfigured_sentry_issue_service_does_not_make_a_request() -> None:
    service = SentryIssueService()
    result = asyncio.run(service.list_issues())
    assert result == {"configured": False, "issues": [], "fetched_at": None, "cached": False}
    assert service._client is None


def test_sentry_public_config_is_public_but_issue_api_requires_manage(
    client: Any, api: Any
) -> None:
    settings = client.app.state.cdp.settings
    settings.sentry_dsn = "https://public@example.test/1"
    settings.sentry_read_token = "server-side-read-secret"

    public = client.get("/api/public-config")
    assert public.status_code == 200
    assert public.json()["sentry_dsn"] == "https://public@example.test/1"
    assert "sentry_read_token" not in public.json()
    assert "server-side-read-secret" not in public.text

    assert client.get("/api/sentry/issues").status_code == 401
    response = api.get("/api/sentry/issues")
    assert response.status_code == 200
    assert response.json()["configured"] is False
