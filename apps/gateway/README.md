# ColabDeploy control plane

One FastAPI process that is simultaneously: the **public API** your callers use,
the **management API** the UI uses, the **relay** every Colab VM dials into, and
the **cache** that keeps answering while those VMs are gone.

That single-process design is deliberate — the whole point is that this can run on
a free tier (Render/Fly/Railway/a $0 dyno) while the GPU work happens in Colab.

```
gateway/
├── main.py            app factory, error envelope, SPA mounting, cli()
├── state.py           AppState: wiring + startup/shutdown of every service
├── core/
│   ├── config.py      Settings (every knob is a CDP_* env var)
│   ├── store.py       JSON/SQLite persistence for the registry
│   ├── repository.py  models, versions, endpoints, deployments, jobs, audit
│   ├── cache.py       TTL + stale-while-revalidate cache, memory or SQLite
│   ├── security.py    API keys, runtime tokens, principals, scopes
│   └── logging.py     ring-buffer log + SSE hub
├── services/
│   ├── relay.py            websocket tunnel: exec correlation, timeouts
│   ├── runtime_manager.py  runtime lifecycle + desired-state generation
│   ├── inference.py        the request path and the degradation policies
│   └── deployments.py      register/deploy/rollback + UI view helpers
└── api/
    ├── v1.py           public inference surface
    ├── management.py   /api — what the UI calls
    ├── internal.py     /internal — what the agent calls (+ WS /internal/relay)
    └── deps.py         auth dependencies and request-scoped helpers
```

## Run it

```bash
pip install -e packages/shared-schema -e apps/gateway
export CDP_ADMIN_KEY="CDP_API_$(openssl rand -hex 16)"   # or let it mint one
python -m gateway.main                                    # 0.0.0.0:8000
```

First boot prints the bootstrap admin key **once**. Open `http://localhost:8000/`
for the UI (if `apps/ui/dist` is built), `/docs` for OpenAPI, `/healthz` for a
liveness probe.

To run the whole system — gateway, a real agent, and a seeded demo model — from one
command:

```bash
python scripts/e2e_local.py
```

## Configuration

Everything is an environment variable; the defaults suit a free-tier single process.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CDP_HOST` / `CDP_PORT` | `0.0.0.0` / `8000` | bind address (0.0.0.0 so previews/proxies work) |
| `CDP_DATA_DIR` | `.data` | registry, artifacts, cache snapshots |
| `CDP_ADMIN_KEY` | *(minted)* | bootstrap key; printed once on first boot |
| `CDP_REQUIRE_AUTH` | `1` | set `0` only for local experiments |
| `CDP_ALLOW_ANONYMOUS_INFERENCE` | `0` | let `/v1` be called without a key |
| `CDP_PUBLIC_URL` | *(from request)* | base URL used in generated links and curl snippets |
| `CDP_CORS_ORIGINS` | `*` | comma-separated |
| `CDP_CACHE_BACKEND` | `memory` | `memory` or `sqlite` |
| `CDP_CACHE_PERSIST` | `1` | snapshot the memory cache to disk |
| `CDP_CACHE_MAX_ENTRIES` / `_MAX_BYTES` | `20000` / `256MB` | cache bounds |
| `CDP_HEARTBEAT_S` / `_STALE_S` / `_OFFLINE_S` | `15` / `45` / `120` | runtime liveness thresholds |
| `CDP_SESSION_MAX_HOURS` | `12` | Colab session budget shown in the UI |
| `CDP_RUNTIME_TOKEN_TTL_HOURS` | `24` | default token lifetime |
| `CDP_TIMEOUT_MS` / `CDP_MAX_BODY_BYTES` / `CDP_MAX_BATCH` | `30000` / `10MB` / `64` | request limits |
| `CDP_JOB_TTL_S` / `CDP_JOB_MAX_ATTEMPTS` | `3600` / `3` | queued-request budget |
| `CDP_UI_DIST` | `apps/ui/dist` | where the built SPA lives |

## The request path

```
POST /v1/models/{route}/predict
  → auth (scope: inference)            → rate limit (per key + per endpoint)
  → cache lookup (endpoint, version, canonical payload hash)
      HIT   → 200, X-ColabDeploy-Cache: HIT
      MISS  → relay exec to the runtime over its websocket
                ↳ agent runs the adapter, replies with JSON
              → cache store → 200, Cache: MISS
      runtime offline / timed out → the endpoint's degradation policy:
        serve_cached → 200 with the last answer, Cache: STALE, `stale: true`
        queue        → 202 with a job_id; runs when the VM returns
        fail         → 503 + Retry-After
```

Every response carries `X-ColabDeploy-Request-Id`, `-Cache`, `-Latency-Ms` and,
when relevant, `-Stale`, `-Job`, `-Runtime` and `-Error`. Errors use one envelope:

```json
{ "error": { "code": "RUNTIME_OFFLINE", "message": "…", "detail": null } }
```

## Resilience notes

* **Desired state, not imperative commands.** A deploy writes intent to the
  registry; a VM reconciles against it on connect and on every change. Nothing
  depends on the VM being online at the moment you click deploy.
* **Warm-up on reconnect.** The gateway ships the top-K hot cache keys (which
  include the original request payloads) with the desired state, so the agent can
  re-warm the models that matter instead of guessing.
* **An agent crash cannot take the tunnel down.** Errors inside a handler become
  `{error, code}` bodies on the exec response; the relay logs the exception *type*
  so a silent teardown is diagnosable from one log line.
* **Cache snapshots.** `POST /api/cache/save` forces a snapshot; with
  `CDP_CACHE_PERSIST=1` it also happens on an interval, so a gateway restart keeps
  serving stale-but-correct answers.

## Tests

```bash
pytest tests/ -q                       # 244 tests, ~19s
pytest tests/test_gateway_api.py -q    # HTTP surface only
pytest tests/test_e2e_agent.py -q      # real uvicorn + real agent over one tunnel
```

The e2e tests are the ones worth reading if you change anything in `services/`:
they assert what a *caller* sees when the VM disappears and comes back.
