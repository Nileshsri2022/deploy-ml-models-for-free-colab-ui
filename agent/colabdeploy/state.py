"""Desired-state reconciliation glue between the tunnel and the model pool.

Kept separate from :mod:`colabdeploy.launch` so it can be unit-tested without
opening a socket: give it a desired-state payload, get back the per-deployment
results the gateway expects in an ``ack`` frame.
"""

from __future__ import annotations

from typing import Any, Callable

from colabdeploy_shared.schemas import DesiredDeployment, DesiredState

from .model_store import ModelStore, ReconcileResult


def build_desired_state(
    runtime_id: str, header: dict[str, Any], deployments: list[dict[str, Any]]
) -> DesiredState:
    """Assemble a :class:`DesiredState` from a ``desired_state`` frame."""
    return DesiredState(
        runtime_id=runtime_id,
        revision=int(header.get("revision", 0) or 0),
        deployments=[DesiredDeployment.model_validate(dep) for dep in deployments],
        warmup_top_k=int(header.get("warmup_top_k", 20) or 20),
        settings=dict(header.get("settings") or {}),
    )


def result_to_dict(result: ReconcileResult) -> dict[str, Any]:
    """Shape the gateway's ack frame expects (see ``AckFrame.deployment_states``)."""
    return {
        "deployment_id": result.deployment_id,
        "model_id": result.model_id,
        "model_version": result.model_version,
        "state": result.state,
        "warm": result.warm,
        "warmup_ms": result.warmup_ms,
        "error": result.error,
        "adapter": result.adapter,
        "elapsed_ms": result.elapsed_ms,
        "from_cache": result.from_cache,
    }


def make_reconciler(
    store: ModelStore,
    runtime_id: str,
    *,
    log: Callable[[str, str], None] | None = None,
    warmup_samples: list[dict[str, Any]] | None = None,
) -> Callable[[dict[str, Any], list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return the ``on_desired_state`` callback the tunnel needs."""
    emit = log or (lambda message, level="info": None)

    def reconcile(header: dict[str, Any], deployments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        desired = build_desired_state(runtime_id, header, deployments)
        if not desired.deployments:
            emit("desired state is empty - unloading everything", "info")
            store.unload_all()
            return []
        emit(f"reconciling {len(desired.deployments)} deployment(s) (rev {desired.revision})", "info")
        results = store.reconcile(desired, warmup_samples=warmup_samples)
        return [result_to_dict(result) for result in results]

    return reconcile
