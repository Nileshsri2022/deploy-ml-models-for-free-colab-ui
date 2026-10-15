"""Auth dependencies and shared request helpers.

Three credential flavours, one extraction function:

* ``Authorization: Bearer <key>`` - normal clients and the UI
* ``X-API-Key: <key>``            - same thing, friendlier for curl/JS
* runtime tokens                  - only valid on ``/internal/*`` routes

Public inference can be left open (``CDP_ALLOW_ANONYMOUS_INFERENCE=true``) for a
demo deployment; the default requires a key.
"""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException, Request

from gateway.core.security import (
    ANONYMOUS,
    SCOPE_INFERENCE,
    SCOPE_MANAGE,
    AuthService,
    Principal,
)
from gateway.state import AppState, get_state


class AuthError(HTTPException):
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(status_code=status_code, detail={"code": "UNAUTHENTICATED", "message": message})


def app_state(request: Request) -> AppState:
    state = getattr(request.app.state, "cdp", None)
    return state or get_state()


def auth_service(request: Request) -> AuthService:
    return AuthService(app_state(request).registry)


def _extract_token(
    authorization: str | None, x_api_key: str | None, runtime_token: str | None
) -> str:
    for raw in (authorization, x_api_key, runtime_token):
        if not raw:
            continue
        value = raw.strip()
        if value.lower().startswith("bearer "):
            value = value[7:].strip()
        if value:
            return value
    return ""


def get_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    """Resolve the caller. Raises only if auth is required and missing/invalid."""
    state = app_state(request)
    token = _extract_token(authorization, x_api_key, None)
    if token:
        principal = AuthService(state.registry).verify_api_key(token)
        if principal is None:
            raise AuthError("invalid or revoked API key")
        return principal
    if state.settings.require_auth and not state.settings.allow_anonymous_inference:
        raise AuthError("missing API key - send 'Authorization: Bearer <key>' or 'X-API-Key'")
    return ANONYMOUS


def require_manage(principal: Principal = None) -> Principal:  # type: ignore[assignment]
    if principal is None or not principal.is_admin:
        raise AuthError("this action needs a key with the 'manage' scope", 403)
    return principal


def require_inference(principal: Principal) -> Principal:
    if principal.kind == "anonymous":
        return principal
    if not (principal.can(SCOPE_INFERENCE) or principal.can(SCOPE_MANAGE)):
        raise AuthError("this key cannot call inference endpoints", 403)
    return principal


def require_runtime(request: Request, token: str) -> dict[str, Any]:
    """Validate an agent's runtime token; 401 on anything else."""
    state = app_state(request)
    record = AuthService(state.registry).verify_runtime_token(token)
    if record is None:
        raise AuthError("invalid, revoked or expired runtime token")
    return record


def manage_dependency(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    """FastAPI dependency: authenticated caller that must hold ``manage``."""
    state = app_state(request)
    token = _extract_token(authorization, x_api_key, None)
    if not token:
        if not state.settings.require_auth:
            return Principal(kind="api_key", id="dev", label="dev", scopes=(SCOPE_MANAGE,))
        raise AuthError("missing API key")
    principal = AuthService(state.registry).verify_api_key(token)
    if principal is None:
        raise AuthError("invalid or revoked API key")
    if not principal.is_admin:
        raise AuthError("this action needs a key with the 'manage' scope", 403)
    return principal


def inference_dependency(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> Principal:
    """FastAPI dependency for ``/v1`` inference routes."""
    state = app_state(request)
    token = _extract_token(authorization, x_api_key, None)
    if token:
        principal = AuthService(state.registry).verify_api_key(token)
        if principal is None:
            raise AuthError("invalid or revoked API key")
        return require_inference(principal)
    if state.settings.allow_anonymous_inference:
        return ANONYMOUS
    if not state.settings.require_auth:
        return ANONYMOUS
    raise AuthError("missing API key - send 'Authorization: Bearer <key>' or 'X-API-Key'")


__all__ = [
    "ANONYMOUS",
    "AuthError",
    "AuthService",
    "Principal",
    "app_state",
    "auth_service",
    "get_principal",
    "inference_dependency",
    "manage_dependency",
    "require_inference",
    "require_manage",
    "require_runtime",
]
