"""HTTP API surface: public inference, management, and agent-internal routes."""

from gateway.api.deps import (
    get_principal,
    require_manage,
    require_inference,
    require_runtime,
    AuthError,
)

__all__ = ["AuthError", "get_principal", "require_inference", "require_manage", "require_runtime"]
