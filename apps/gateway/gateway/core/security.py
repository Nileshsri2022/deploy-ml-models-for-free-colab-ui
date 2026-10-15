"""Authentication primitives.

Two credential types, both stored **hashed** - the plaintext is shown exactly once
in the UI and never written to disk or logs:

* ``CDP_API_...``  - caller/admin keys for the HTTP API (scopes: inference, manage)
* ``CDP_RT_...``   - runtime tokens; a Colab agent presents one to bind a VM to
  this control plane. Scoped to a single ``runtime_id``, revocable, TTL'd.

Hashing is SHA-256 with a per-installation pepper. For a self-hosted single-user
deployment this is the right trade-off (fast, no dependency); the ``hash_secret``
function is the single place to swap in argon2 if this ever becomes multi-tenant.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from colabdeploy_shared.schemas import ApiKeyInfo, utcnow
from colabdeploy_shared.utils import short_id
from gateway.core.repository import Registry

API_KEY_PREFIX = "CDP_API_"
RUNTIME_TOKEN_PREFIX = "CDP_RT_"

SCOPE_INFERENCE = "inference"
SCOPE_MANAGE = "manage"


def _pepper() -> bytes:
    """Stable per-installation salt, persisted next to the data dir."""
    env = os.getenv("CDP_SECRET_PEPPER")
    if env:
        return env.encode("utf-8")
    return b"colabdeploy-v1"  # documented default; override in production


def hash_secret(token: str) -> str:
    return hashlib.sha256(_pepper() + token.encode("utf-8")).hexdigest()


def constant_time_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def generate_api_key() -> str:
    return API_KEY_PREFIX + secrets.token_urlsafe(24)


def generate_runtime_token() -> str:
    return RUNTIME_TOKEN_PREFIX + secrets.token_urlsafe(24)


def key_prefix(token: str) -> str:
    """Identifying (non-secret) prefix shown in the UI, e.g. ``CDP_API_9f2a…``."""
    body = token.split("_", 2)[-1]
    return f"{'_'.join(token.split('_')[:2])}_{body[:6]}"


@dataclass
class Principal:
    """Who is calling."""

    kind: str  # "api_key" | "runtime" | "anonymous"
    id: str
    label: str = ""
    scopes: tuple[str, ...] = ()

    def can(self, scope: str) -> bool:
        return self.kind == "anonymous" or scope in self.scopes or SCOPE_MANAGE in self.scopes

    @property
    def is_admin(self) -> bool:
        return SCOPE_MANAGE in self.scopes


ANONYMOUS = Principal(kind="anonymous", id="anonymous", label="anonymous")


class AuthService:
    """Issues and verifies credentials against the registry."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    # ------------------------------------------------------------------ #
    # API keys
    # ------------------------------------------------------------------ #
    def create_api_key(
        self,
        label: str = "default",
        scopes: tuple[str, ...] = (SCOPE_INFERENCE, SCOPE_MANAGE),
        expires_in_days: int | None = None,
    ) -> tuple[ApiKeyInfo, str]:
        token = generate_api_key()
        info = ApiKeyInfo(
            id=short_id("key", 6),
            label=label,
            prefix=key_prefix(token),
            scopes=list(scopes),
            expires_at=(utcnow() + timedelta(days=expires_in_days)) if expires_in_days else None,
            token=token,
        )
        record = info.model_dump(mode="json")
        record["token_hash"] = hash_secret(token)
        record.pop("token", None)
        self.registry.store.put("api_keys", info.id, record)
        self.registry.audit.record("api_key.create", actor=label, key_id=info.id)
        return info, token

    def revoke_api_key(self, key_id: str) -> bool:
        updated = self.registry.store.update("api_keys", key_id, {"revoked": True})
        if updated:
            self.registry.audit.record("api_key.revoke", key_id=key_id)
        return bool(updated)

    def verify_api_key(self, token: str) -> Principal | None:
        if not token:
            return None
        digest = hash_secret(token)
        for record in self.registry.store.all("api_keys"):
            if not constant_time_eq(str(record.get("token_hash", "")), digest):
                continue
            if record.get("revoked"):
                return None
            expires = record.get("expires_at")
            if expires:
                try:
                    when = datetime.fromisoformat(str(expires))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    if when < utcnow():
                        return None
                except ValueError:  # pragma: no cover - defensive
                    return None
            self.registry.store.update(
                "api_keys", str(record["id"]), {"last_used_at": utcnow().isoformat()}
            )
            return Principal(
                kind="api_key",
                id=str(record["id"]),
                label=str(record.get("label", "")),
                scopes=tuple(record.get("scopes", [])),
            )
        return None

    def list_api_keys(self) -> list[ApiKeyInfo]:
        out = []
        for record in self.registry.store.all("api_keys"):
            record = {k: v for k, v in record.items() if k != "token_hash"}
            out.append(ApiKeyInfo.model_validate(record))
        return sorted(out, key=lambda k: k.created_at, reverse=True)

    def ensure_bootstrap_key(self, configured: str = "") -> tuple[str, bool]:
        """Make sure at least one admin key exists.

        Returns ``(token, created)``. If ``CDP_ADMIN_KEY`` is set we honour it so
        deploys are reproducible; otherwise we mint one and the caller prints it.
        """
        existing = self.registry.store.all("api_keys")
        if existing:
            return "", False
        token = configured.strip() or generate_api_key()
        info = ApiKeyInfo(
            id=short_id("key", 6),
            label="bootstrap-admin",
            prefix=key_prefix(token),
            scopes=[SCOPE_INFERENCE, SCOPE_MANAGE],
        )
        record = info.model_dump(mode="json")
        record["token_hash"] = hash_secret(token)
        record.pop("token", None)
        self.registry.store.put("api_keys", info.id, record)
        self.registry.audit.record("api_key.bootstrap", actor="system", key_id=info.id)
        return token, True

    # ------------------------------------------------------------------ #
    # Runtime tokens
    # ------------------------------------------------------------------ #
    def create_runtime_token(
        self, label: str = "colab-runtime", ttl_hours: float = 24.0, colab_url: str | None = None
    ) -> dict[str, Any]:
        """Mint a runtime record + token in one step (the UI's "Connect" action)."""
        token = generate_runtime_token()
        runtime_id = short_id("rt", 6)
        now = utcnow()
        record: dict[str, Any] = {
            "id": runtime_id,
            "label": label,
            "state": "registered",
            "capabilities": {},
            "colab_url": colab_url,
            "tunnel_kind": None,
            "tunnel_url": None,
            "last_heartbeat_at": None,
            "session_started_at": None,
            "session_max_hours": 12.0,
            "loaded_models": [],
            "deployments": [],
            "error": None,
            "registered_at": now.isoformat(),
            "token_hash": hash_secret(token),
            "token_prefix": key_prefix(token),
            "token_expires_at": (now + timedelta(hours=ttl_hours)).isoformat() if ttl_hours else None,
            "revoked": False,
        }
        self.registry.store.put("runtimes", runtime_id, record)
        self.registry.audit.record("runtime.token_created", runtime_id=runtime_id, label=label)
        return {
            "runtime_id": runtime_id,
            "token": token,
            "label": label,
            "expires_at": record["token_expires_at"],
            "created_at": now.isoformat(),
        }

    def verify_runtime_token(self, token: str) -> dict[str, Any] | None:
        if not token:
            return None
        digest = hash_secret(token)
        for record in self.registry.store.all("runtimes"):
            if not constant_time_eq(str(record.get("token_hash", "")), digest):
                continue
            if record.get("revoked"):
                return None
            expires = record.get("token_expires_at")
            if expires:
                try:
                    when = datetime.fromisoformat(str(expires))
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=timezone.utc)
                    if when < utcnow():
                        return None
                except ValueError:  # pragma: no cover
                    return None
            return record
        return None

    def revoke_runtime_token(self, runtime_id: str) -> bool:
        updated = self.registry.store.update(
            "runtimes", runtime_id, {"revoked": True, "state": "offline"}
        )
        ok = updated is not None
        if ok:
            self.registry.audit.record("runtime.token_revoked", runtime_id=runtime_id)
        return ok

    def rotate_runtime_token(self, runtime_id: str, ttl_hours: float = 24.0) -> dict[str, Any] | None:
        token = generate_runtime_token()
        now = utcnow()
        updated = self.registry.store.update(
            "runtimes",
            runtime_id,
            {
                "token_hash": hash_secret(token),
                "token_prefix": key_prefix(token),
                "token_expires_at": (now + timedelta(hours=ttl_hours)).isoformat() if ttl_hours else None,
                "revoked": False,
            },
        )
        if not updated:
            return None
        self.registry.audit.record("runtime.token_rotated", runtime_id=runtime_id)
        return {"runtime_id": runtime_id, "token": token, "expires_at": updated["token_expires_at"]}
