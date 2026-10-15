"""Control-plane services: the logic the HTTP routers call into.

Split out from ``gateway.api`` so the routers stay thin and the behaviour can be
exercised without a transport:

* :mod:`~gateway.services.relay` - the websocket tunnel to each VM (exec
  request/response correlation, heartbeats, timeouts)
* :mod:`~gateway.services.runtime_manager` - runtime lifecycle (registered →
  connecting → online → stale → offline) and desired-state generation
* :mod:`~gateway.services.inference` - the request path: cache lookup, relay
  call, degradation policy, job queueing, telemetry
* :mod:`~gateway.services.deployments` - models, versions, endpoints,
  deployments, rollbacks and the UI-facing view helpers
"""

from __future__ import annotations

__all__ = ["deployments", "inference", "relay", "runtime_manager"]
