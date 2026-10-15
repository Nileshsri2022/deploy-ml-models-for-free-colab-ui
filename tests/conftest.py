"""Shared fixtures.

Everything runs against a throwaway data directory, with a fixed bootstrap admin
key so tests can authenticate deterministically. The path bootstrap at the top
means ``pytest`` works on a fresh clone with no editable installs:

    pip install fastapi uvicorn httpx websockets pytest pytest-asyncio
    pytest
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "packages" / "shared-schema", ROOT / "apps" / "gateway", ROOT / "agent"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tests.support import TEST_ADMIN_KEY, TEST_RUNTIME_LABEL, wait_for  # noqa: E402

__all__ = ["TEST_ADMIN_KEY", "TEST_RUNTIME_LABEL", "wait_for"]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """A fresh control-plane data directory per test."""
    path = tmp_path / "cdp-data"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture()
def settings(data_dir: Path) -> Any:
    from gateway.core.config import reset_settings

    os.environ["CDP_ADMIN_KEY"] = TEST_ADMIN_KEY
    os.environ["CDP_QUIET_BOOT"] = "1"
    return reset_settings(
        data_dir=data_dir,
        require_auth=True,
        allow_anonymous_inference=False,
        cache_backend="memory",
        cache_persist=False,
        bootstrap_admin_key=TEST_ADMIN_KEY,
        heartbeat_interval_s=1.0,
        heartbeat_stale_s=3.0,
        heartbeat_offline_s=6.0,
        session_max_hours=12.0,
        retry_after_s=5,
    )


@pytest.fixture()
def state(settings: Any) -> Any:
    from gateway.state import AppState, set_state

    app_state = AppState(settings)
    set_state(app_state)

    from gateway.core.security import AuthService

    AuthService(app_state.registry).ensure_bootstrap_key(TEST_ADMIN_KEY)
    yield app_state
    set_state(None)


@pytest.fixture()
def client(state: Any) -> Iterator[Any]:
    """TestClient wired to the same state the services use."""
    from fastapi.testclient import TestClient

    from gateway.main import create_app

    app = create_app(state)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_ADMIN_KEY}"}


@pytest.fixture()
def api(client: Any, auth: dict[str, str]) -> Any:
    """Small helper so tests read like the API docs."""

    class Api:
        def __init__(self) -> None:
            self.client = client
            self.headers = auth

        def get(self, path: str, **kw: Any) -> Any:
            return client.get(path, headers=auth, **kw)

        def post(self, path: str, **kw: Any) -> Any:
            return client.post(path, headers=auth, **kw)

        def patch(self, path: str, **kw: Any) -> Any:
            return client.patch(path, headers=auth, **kw)

        def delete(self, path: str, **kw: Any) -> Any:
            return client.delete(path, headers=auth, **kw)

    return Api()


@pytest.fixture()
def runtime_token(api: Any) -> dict[str, Any]:
    """Mint a runtime + token the way the UI's Connect page does."""
    response = api.post(f"/api/runtimes/token?label={TEST_RUNTIME_LABEL}&ttl_hours=1")
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def echo_artifact(tmp_path: Path) -> Path:
    """An artifact the echo adapter can load with zero ML dependencies."""
    path = tmp_path / "echo.echo.json"
    path.write_text('{"latency_ms": 0}', encoding="utf-8")
    return path
