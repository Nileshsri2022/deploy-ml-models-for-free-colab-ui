"""The agent's HTTP client for the control plane.

The WebSocket tunnel carries request/response traffic, but the agent also needs
plain HTTP for three things: claiming its runtime record (``register``), fetching
desired state when it first boots (before the tunnel is up), and shipping logs when
the socket is down. All of it is authenticated with the **runtime token**, which is
scoped to a single ``runtime_id`` and revocable from the UI.
"""

from __future__ import annotations

import contextlib
import platform
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from colabdeploy_shared.schemas import DesiredState


class GatewayError(RuntimeError):
    """The control plane refused us; ``code`` mirrors the gateway's error codes."""

    def __init__(self, message: str, *, code: str = "GATEWAY_ERROR", status: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass
class Registration:
    runtime_id: str
    relay_ws_url: str
    heartbeat_interval_s: float
    session_max_hours: float
    desired_state: DesiredState
    server_version: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class GatewayClient:
    """Thin, retrying HTTP client for ``/internal/*``."""

    def __init__(
        self,
        gateway_url: str,
        token: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.5,
    ) -> None:
        self.gateway_url = gateway_url.strip().rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._client = httpx.Client(
            base_url=self.gateway_url,
            timeout=timeout_s,
            headers=self.headers(),
            follow_redirects=True,
        )

    def headers(self) -> dict[str, str]:
        return {
            "X-Runtime-Token": self.token,
            "User-Agent": f"colabdeploy-agent/0.1.0 ({platform.system()} {platform.python_version()})",
        }

    # ------------------------------------------------------------------ #
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.request(method, path, **kwargs)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_s * (attempt + 1))
                    continue
                raise GatewayError(
                    f"cannot reach the control plane at {self.gateway_url}: {exc}",
                    code="GATEWAY_UNREACHABLE",
                ) from exc
            if response.status_code == 401:
                raise GatewayError(
                    "runtime token rejected (revoked or expired) - mint a new one in the UI",
                    code="BAD_RUNTIME_TOKEN",
                    status=401,
                )
            if response.status_code == 403:
                raise GatewayError(
                    "this token is not allowed to do that", code="FORBIDDEN", status=403
                )
            if response.status_code >= 500 and attempt < self.max_retries - 1:
                last_error = GatewayError(
                    f"gateway HTTP {response.status_code}", status=response.status_code
                )
                time.sleep(self.retry_backoff_s * (attempt + 1))
                continue
            if response.status_code >= 400:
                raise GatewayError(
                    _error_message(response), code=_error_code(response), status=response.status_code
                )
            if not response.content:
                return {}
            try:
                return response.json()
            except ValueError as exc:
                raise GatewayError(f"gateway returned non-JSON: {response.text[:200]}") from exc
        raise GatewayError(f"request failed after retries: {last_error}")

    # ------------------------------------------------------------------ #
    def register(self, capabilities: dict[str, Any], colab_url: str | None = None) -> Registration:
        payload = {
            "token": self.token,
            "capabilities": capabilities,
            "colab_url": colab_url,
            "session_started_at": time.time(),
            "agent_version": "0.1.0",
        }
        data = self._request("POST", "/internal/runtime/register", json=payload)
        runtime_id = str(data.get("runtime_id") or "")
        if not runtime_id:
            raise GatewayError("gateway did not return a runtime_id")
        desired_raw = data.get("desired_state") or {"runtime_id": runtime_id, "deployments": []}
        desired_raw["runtime_id"] = runtime_id
        return Registration(
            runtime_id=runtime_id,
            relay_ws_url=str(data.get("relay_ws_url") or ""),
            heartbeat_interval_s=float(data.get("heartbeat_interval_s") or 15.0),
            session_max_hours=float(data.get("session_max_hours") or 12.0),
            desired_state=DesiredState.model_validate(desired_raw),
            server_version=str(data.get("server_version") or ""),
            raw=data,
        )

    def desired_state(self, runtime_id: str) -> DesiredState:
        data = self._request(
            "GET", "/internal/runtime/desired-state", params={"runtime_id": runtime_id}
        )
        return DesiredState.model_validate(data)

    def heartbeat(self, runtime_id: str, status: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "runtime_id": runtime_id,
            "state": status.get("state", "online"),
            "capabilities": status.get("capabilities", {}),
            "loaded_models": status.get("loaded_models", []),
            "queue_depth": 0,
            "inflight": 0,
            "session_started_at": time.time(),
            "metrics": {
                k: float(v)
                for k, v in (status.get("counters") or {}).items()
                if isinstance(v, (int, float))
            },
        }
        return self._request("POST", "/internal/runtime/heartbeat", json=payload)

    def push_logs(self, runtime_id: str, lines: list[dict[str, Any]]) -> int:
        if not lines:
            return 0
        data = self._request(
            "POST", "/internal/runtime/logs", json={"runtime_id": runtime_id, "lines": lines}
        )
        return int(data.get("accepted", 0))

    def health(self) -> dict[str, Any]:
        """Unauthenticated liveness probe of the control plane."""
        try:
            return self._request("GET", "/v1/health")
        except GatewayError:
            return {"status": "unreachable", "gateway_url": self.gateway_url}

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()


def _error_message(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or data)
            if "detail" in data:
                return str(data["detail"])
        return str(data)[:400]
    except ValueError:
        return response.text[:400] or f"HTTP {response.status_code}"


def _error_code(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict) and error.get("code"):
                return str(error["code"])
    except ValueError:
        pass
    return f"HTTP_{response.status_code}"
