"""HTTP API surface: public inference, management, and agent-internal routes."""

from gateway.api.deps import (
    AuthError,
    get_principal,
    require_inference,
    require_manage,
    require_runtime,
)

__all__ = ["AuthError", "get_principal", "require_inference", "require_manage", "require_runtime"]
