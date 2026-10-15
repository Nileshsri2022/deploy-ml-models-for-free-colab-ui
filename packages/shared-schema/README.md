# colabdeploy-shared

The Pydantic contracts that the **control plane** (`apps/gateway`) and the **Colab
agent** (`agent/`) both speak. One definition, two processes, no drift: if a field
changes here, both sides fail loudly at the boundary instead of silently
misbehaving at 3 a.m. when a notebook reconnects.

```
colabdeploy_shared/
├── enums.py      AdapterKind, RuntimeState, DeploymentState, DegradationMode,
│                 CacheStatus, EndpointVisibility, JobState, LogLevel …
├── schemas.py    domain models: ModelInfo, EndpointInfo, DeploymentInfo,
│                 RuntimeInfo, ArtifactSpec, ModelSchema, InferenceResponse,
│                 CacheStats, DesiredState …
├── protocol.py   the wire frames that cross the websocket tunnel
└── utils.py      canonical_json, stable_hash, slugify, utcnow, new_id …
```

## Two conventions worth knowing before you edit this package

**1. Wire frames carry ISO-8601 *strings*; domain models carry `datetime`.**

`protocol.py` describes bytes on a socket, where a timestamp is text. `schemas.py`
describes objects inside a process, where it is a `datetime`. Mixing them up is the
one mistake that produces an `AttributeError` deep inside a relay handler rather
than a validation error at the edge, so the boundary is explicit: parse with
`utils`/`_parse_dt` when you need a `datetime`, and use the string as-is otherwise.

**2. Don't reach for `model_config = {"use_enum_values": True}`.**

It looks convenient and it is a trap: enum fields become plain strings, so
`state.value` raises and `state == RuntimeState.ONLINE` silently returns `False`.
Serialise with `model_dump(mode="json")` when you want strings, and normalise mixed
input with `schemas.enum_value()`.

## Cache key contract

```python
from colabdeploy_shared.utils import cache_key
cache_key(endpoint_id, model_version, payload)   # canonical_json + stable_hash
```

The key includes the model **version**, so a rollback can never serve the previous
version's cached answers. A cached value is stored as
`{"output": …, "__request__": payload, "endpoint_route": route}` — `__request__` is
what lets the gateway rebuild hot warm-up samples for a reconnecting VM.

## Development

```bash
pip install -e packages/shared-schema
pytest tests/ -q          # from the repo root
```
