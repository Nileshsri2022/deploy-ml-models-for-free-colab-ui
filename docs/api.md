# ColabDeploy API reference

Three surfaces, one process:

| Prefix | Audience | Auth |
| --- | --- | --- |
| `/v1` | your callers | API key with the `inference` scope |
| `/api` | the UI / you | API key with the `manage` scope |
| `/internal` | the agent on a VM | runtime token (`CDP_RT_…`) |

Live, interactive versions: **`/docs`** (Swagger) and **`/redoc`**, generated from
the same Pydantic models in `packages/shared-schema`.

---

## Conventions

**Authentication.** Any of these works, checked in this order:

```
Authorization: Bearer <key>
X-API-Key: <key>
?api_key=<key>            # last resort: browser EventSource cannot set headers,
                          # and the UI live-tails logs over SSE
```

Agent routes accept `X-Runtime-Token` instead. With `CDP_REQUIRE_AUTH=0`
(development only) everything is open.

**Errors.** One envelope, always:

```json
{ "error": { "code": "RUNTIME_OFFLINE", "message": "no runtime is attached", "detail": null } }
```

`code` is stable and machine-readable; `message` is for humans; `detail` carries
structured extras (validation errors, the endpoint's expected schema). The HTTP
status mirrors the code, and `X-ColabDeploy-Error: <code>` repeats it as a header
so a proxy that swallows bodies still tells you what happened.

| Code | Status | Meaning |
| --- | --- | --- |
| `UNAUTHORIZED` / `FORBIDDEN` | 401 / 403 | missing, invalid or insufficiently scoped key |
| `NOT_FOUND` | 404 | unknown model, endpoint, runtime, job or cache key |
| `INVALID_PAYLOAD` | 422 | body failed the endpoint's input schema (`detail.errors`) |
| `VALIDATION_ERROR` | 422 | FastAPI/Pydantic rejected the request shape |
| `PAYLOAD_TOO_LARGE` / `BATCH_TOO_LARGE` | 413 | over `CDP_MAX_BODY_BYTES` / `CDP_MAX_BATCH` |
| `RUNTIME_OFFLINE` | 503 | no runtime attached and the policy is `fail` |
| `AGENT_TIMEOUT` | 504 | the VM did not answer within the endpoint timeout |
| `MODEL_ERROR` | 500 | the adapter raised; `detail` has the adapter's message |
| `RATE_LIMITED` | 429 | over the endpoint's `rate_limit_rpm` |
| `QUOTA_EXCEEDED` | 429 | the API key used its daily inference request allowance; resets at 00:00 UTC |
| `STALE_RESPONSE` | 200 | served from cache while the runtime was unreachable |
| `QUEUED` | 202 | accepted as a job; runs when a runtime returns |

**Daily API-key quota.** Each authenticated API key receives up to 1,000 HTTP inference requests per UTC day by default. Calls to `predict`, `predict:batch`, and `jobs` count once each, including cache hits; a batch counts as one HTTP request, not one request per item. Anonymous requests are not charged. Set `CDP_DAILY_INFERENCE_QUOTA` on the gateway to change the default; set it to `0` to disable this quota. Usage is persisted by the gateway's local store. Exceeding the limit returns `429 QUOTA_EXCEEDED` with `Retry-After` seconds until midnight UTC and `X-ColabDeploy-Quota-Limit` / `X-ColabDeploy-Quota-Remaining` headers.

**Response headers on `/v1` inference calls**

| Header | Example |
| --- | --- |
| `X-ColabDeploy-Cache` | `HIT` \| `MISS` \| `STALE` \| `BYPASS` |
| `X-ColabDeploy-Request-Id` | `f2dbdee1bf0249dc` (or your `X-Request-Id`, echoed) |
| `X-ColabDeploy-Latency-Ms` | `1.4` |
| `X-ColabDeploy-Stale` | `true` (only when served stale) |
| `X-ColabDeploy-Runtime` | `rt_9cd2c7cf87e3` (when a VM actually ran it) |
| `X-ColabDeploy-Job` | `job_…` (when queued) |
| `Retry-After` | `30` (on 503/429) |

---

## Public inference API — `/v1`

### `POST /v1/models/{identifier}/predict`

`identifier` is the endpoint **route** (preferred: stable across model versions) or
a model id.

```bash
curl -X POST "$GW/v1/models/demo-sentiment/predict" \
  -H "Authorization: Bearer $CDP_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {"text": "this is a great little demo"}}'
```

Request body:

| Field | Type | Notes |
| --- | --- | --- |
| `inputs` | object | canonical shape; passed to the adapter as the payload |
| `payload` | object | accepted alias, so Gradio/HF-style clients work unchanged |
| `bypass_cache` | bool | force a real model call (`cache: "BYPASS"`) |

Response — `200` (or `202` when queued):

```json
{
  "model_id": "mdl_0e7270ac91bafcbd",
  "endpoint_id": "ep_3403cb56f803db73",
  "model_version": 1,
  "output": { "label": "positive", "score": 2, "words": 10, "chars": 46 },
  "cache": "MISS",
  "latency_ms": 2.1,
  "runtime_id": "rt_9cd2c7cf87e3",
  "request_id": "f2dbdee1bf0249dc",
  "job_id": null,
  "stale": false,
  "served_at": "2026-10-15T19:50:04.123456Z",
  "error": null
}
```

`output` is whatever your model returned, JSON-normalised. `cache` tells you
whether a VM was involved: `HIT` means the answer never left the gateway.

### `POST /v1/models/{identifier}/predict:batch`

One round trip, one row per result. Concurrency to the VM is capped at 4.

```json
{ "instances": [ {"text": "great"}, {"text": "awful"} ] }
```

`inputs_list` and `payloads` are accepted aliases. Over `CDP_MAX_BATCH` (default
64) → `413 BATCH_TOO_LARGE`.

```json
{
  "endpoint_id": "ep_…", "model_id": "mdl_…", "model_version": 1, "count": 2,
  "results": [ { "output": {"label": "positive"}, "cache": "MISS", "latency_ms": 1.8 },
               { "output": {"label": "negative"}, "cache": "HIT",  "latency_ms": 0.1 } ]
}
```

### `GET /v1/models/{identifier}/schema`

The request/response contract, which is what the UI's playground form is built
from. No auth required, so a client can discover the shape before it has a key.

### `GET /v1/endpoints` · `GET /v1/status` · `GET /v1/health` · `GET /v1/metrics`

* `endpoints` — published routes with their policy and live state
* `status` — one call for a dashboard: runtimes, cache stats, queue depth
* `health` — liveness (does not require a runtime to be attached)
* `metrics` — Prometheus text format

### `POST /v1/jobs` · `GET /v1/jobs/{job_id}`

Submit a request to be executed when a runtime is available, and poll for it.
This is what the `queue` degradation policy produces automatically.

```json
{ "endpoint_id": "ep_…", "inputs": {"text": "…"}, "webhook_url": "https://…" }
```

Job states: `queued → running → succeeded | failed | expired`. A job is retried up
to `CDP_JOB_MAX_ATTEMPTS` (default 3) and expires after `CDP_JOB_TTL_S`.

---

## Management API — `/api`

Everything here needs the `manage` scope. This is exactly what the UI calls.

### Setup

| Route | Purpose |
| --- | --- |
| `GET /api/bootstrap` | gateway URL, relay websocket URL, Colab relaunch link, the notebook cell to paste, and what's already installed |
| `POST /api/keys` | mint an API key — `{label, scopes[], expires_in_days}`; the plaintext is returned **once** |
| `GET /api/keys` · `DELETE /api/keys/{id}` | list / revoke |

### Runtimes

| Route | Purpose |
| --- | --- |
| `POST /api/runtimes/token?label=&ttl_hours=` | create a runtime record + token (201). Response includes `notebook_cell` and `colab_relaunch_url` |
| `GET /api/runtimes` · `GET /api/runtimes/{id}` | list / inspect, including capabilities, session countdown and loaded models |
| `POST /api/runtimes/{id}/rotate` | new token, old one dies immediately |
| `POST /api/runtimes/{id}/reconcile` | force-push desired state + hot warm-up keys now |
| `DELETE /api/runtimes/{id}` | revoke and drop the tunnel |

Runtime states: `registered → connecting → online → stale → offline`, plus
`degraded` when the tunnel is up but a deployment failed to load.

### Models

| Route | Purpose |
| --- | --- |
| `POST /api/models` | register by artifact URI (`ModelCreate`) |
| `POST /api/models/hf-check` | name-based checks for a Hub URI: `{uri, task?}` → `{repo, filename, task, ok, blockers[], warnings[]}`. Does not contact the Hub. Registration of an `hf://` model returns `400` when there are blockers (GGUF, diffusion pipelines, unsupported task). |
| `POST /api/models/upload` | multipart upload: `file`, `name`, `framework`, `adapter`, `schema_in`, `schema_out`, `warmup_sample`, `entrypoint`, `tags` — stored and served back to agents as `store://art_<id>` |
| `GET /api/models` | `{count, models[], latest[]}` — `latest` is one row per model name |
| `GET /api/models/{id}` | includes `all_versions` and the endpoints using it |

```json
{
  "name": "churn",
  "framework": "sklearn",
  "adapter": "sklearn",
  "artifact": { "uri": "gdrive://1AbCdef…", "checksum": "sha256:…", "size_bytes": 812345 },
  "schema_in": {
    "fields": { "tenure": {"type": "number"} },
    "required": ["tenure"],
    "example": { "tenure": 24 }
  },
  "warmup_sample": { "inputs": { "tenure": 24 } },
  "entrypoint": null,
  "tags": ["demo"]
}
```

Registering the same `name` again creates **v2, v3, …** — versions are the rollback
mechanism, and the cache key includes the version so a rollback can never serve the
new version's answers.

Artifact URI schemes: `file://` or a bare path (already on the VM), `http(s)://`,
`gdrive://<file_id>`, `hf://<repo>[/<file>]`, `store://art_<id>` (uploaded here),
`inline:<json or source>`.

### Deploy

```http
POST /api/deploy
```

```json
{
  "model_id": "mdl_…",
  "model_version": null,
  "endpoint_id": null,
  "runtime_id": null,
  "endpoint": {
    "route": "churn",
    "visibility": "public",
    "cache_ttl_s": 300,
    "stale_while_revalidate_s": 3600,
    "degradation_mode": "serve_cached",
    "timeout_ms": 30000,
    "rate_limit_rpm": 600
  }
}
```

* `model_version` — defaults to latest; pin an older one to roll back
* `endpoint_id` — set it to update an existing endpoint in place (its policy is
  kept and `endpoint` is ignored)
* `runtime_id` — defaults to any online runtime

The response says plainly what happened:

```json
{
  "deployment": { "id": "dep_…", "state": "pending", "…": "…" },
  "endpoint":   { "route": "churn", "url": "https://…/v1/models/churn/predict", "curl": "curl -X POST …" },
  "runtime_id": "rt_9cd2c7cf87e3",
  "runtime_online": true,
  "pushed_to_runtime": true,
  "message": "deployed and pushed to the runtime"
}
```

When nothing is online you get `"deployed; the runtime will load it on next
connect"` — the deploy is **not** a failure, because state is declarative: the VM
reconciles against desired state the moment it re-attaches.

### Endpoints

| Route | Purpose |
| --- | --- |
| `GET /api/endpoints` | list with `url`, `path`, `curl`, `deployment_state`, `runtime_state`, `live` |
| `POST /api/endpoints` | create directly (needs `model_id`) |
| `PATCH /api/endpoints/{id}` | partial update — `model_version`, `visibility`, `cache_ttl_s`, `stale_while_revalidate_s`, `degradation_mode`, `timeout_ms`, `rate_limit_rpm`, `status`. Setting `cache_ttl_s: 0` also flushes that endpoint's cache |
| `POST /api/endpoints/{id}/rollback?model_version=N` | repoint + invalidate cache + push to the VM if it is live |
| `DELETE /api/endpoints/{id}` | drop the endpoint, its deployments and its cache entries |
| `GET /api/deployments` | every deployment row across runtimes |

**Degradation policies** — what a caller sees when no VM can answer:

| `degradation_mode` | Behaviour |
| --- | --- |
| `serve_cached` *(default)* | `200` with the last known answer, `cache: "STALE"`, `stale: true`, `X-ColabDeploy-Error: STALE_RESPONSE`. Only within `stale_while_revalidate_s` |
| `queue` | `202` + `job_id`; executed when a runtime returns |
| `fail` | `503` + `Retry-After` |

### Cache

| Route | Purpose |
| --- | --- |
| `GET /api/cache/stats` | entries, hits/misses/stale hits, hit ratio, bytes, evictions, pinned |
| `GET /api/cache/entries?limit=&endpoint_id=` | the entries themselves, with freshness timestamps and a preview |
| `GET /api/cache/entry/{key}` | one stored value (the model output **plus** the original request) |
| `POST /api/cache/entry/{key}/pin?pinned=true` | pinned entries are never evicted |
| `DELETE /api/cache/entry/{key}` | drop one |
| `POST /api/cache/invalidate?endpoint_id=` | drop one endpoint's entries, or everything |
| `POST /api/cache/save` | force a snapshot to disk (the state-retention guarantee) |
| `GET /api/cache/hot?limit=20` | top-K keys by hits — these are replayed to an agent as warm-up samples on reconnect |

The cache key is `hash(endpoint_id, model_version, canonical_json(payload))`.

### Observability

| Route | Purpose |
| --- | --- |
| `GET /api/logs?limit=&level=&runtime_id=` | the persisted ring buffer, newest first |
| `GET /api/logs/stream?api_key=…` | **SSE** live tail (gateway + relay + agent lines) |
| `GET /api/audit?limit=` | who changed what: deploys, key mints, rollbacks, revocations |
| `GET /api/telemetry` | request counters, latency percentiles, relay health, cache stats, SSE subscribers |
| `GET /api/jobs?state=` · `POST /api/jobs/{id}/retry` | the queue |

---

## Agent API — `/internal`

Authenticated with `X-Runtime-Token`. You should never need to call these by hand;
they are documented so the contract is reviewable.

| Route | Purpose |
| --- | --- |
| `POST /internal/runtime/register` | capabilities in → `runtime_id`, `relay_ws_url`, initial desired state out |
| `GET /internal/runtime/desired-state` | what this VM should be serving right now |
| `POST /internal/runtime/heartbeat` | liveness + loaded models + counters |
| `POST /internal/runtime/logs` | batched agent log lines |
| `GET /internal/artifacts/{artifact_id}` | download a `store://` artifact uploaded through the UI |
| `WS /internal/relay` | the tunnel |

### The relay protocol

Frames are JSON, one per message, defined in
[`packages/shared-schema/colabdeploy_shared/protocol.py`](../packages/shared-schema/colabdeploy_shared/protocol.py).
Timestamps on the wire are **ISO-8601 strings**; only in-process domain models use
`datetime`.

```
agent → gateway   Hello {protocol: 1, runtime_id, token, capabilities, session_started_at}
gateway → agent   HelloAck {ok, desired_state_revision, server_version}

gateway → agent   ExecRequest {request_id, method, path, body|body_b64, timeout_ms}
agent → gateway   ExecResponse {request_id, status, body|body_b64, agent_latency_ms}

gateway → agent   DesiredStateFrame {revision, deployments[], warmup_samples[], settings}
agent → gateway   Heartbeat {loaded_models[], counters, metrics, session_started_at}
agent → gateway   LogFrame {lines[]}
```

Exec paths the agent implements: `/execute`, `/warmup`, `/reconcile`, `/unload`,
`/models`. An unknown path returns `404 {"error": …, "code": "UNKNOWN_PATH"}`; an
internal failure returns `500 {"code": "AGENT_ERROR"}` — **an agent exception never
tears down the tunnel**.

Close codes: `4400` bad hello, `4401` bad token, `4403` revoked, `4404` unknown
runtime, `4426` protocol error. Idle sockets are dropped after 120 s; the agent
reconnects forever with exponential backoff (1 → 60 s) plus jitter.

---

## Meta

| Route | Purpose |
| --- | --- |
| `GET /` | the UI for a browser (`Accept: text/html`), JSON service metadata for scripts |
| `GET /ui`, `GET /ui/{path}` | the built SPA, with a fallback to `index.html` |
| `GET /healthz` | liveness probe: version, runtimes online, cache entries, uptime |
| `GET /docs` · `/redoc` · `/openapi.json` | interactive API docs |
