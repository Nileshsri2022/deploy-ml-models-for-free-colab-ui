# Architecture — ColabDeploy

**A UI-driven platform that turns a Google Colab session into an always-available ML inference host, with a single common API for deploying *any* model, published endpoints, and a cache layer that keeps the system usable across Colab disconnects.**

> Status: Design (v0.1) · Owner: @Nileshsri2022 · Last updated: 2026-10-15

---

## 1. Problem Statement

Deploying an ML model normally requires a paid GPU box, a container registry, and a serving framework. A student/hobbyist already has a **free Colab session with a GPU** and a trained model — but Colab gives no public ingress, dies on idle, and loses all state.

We want:

1. **A UI** that attaches to a running Colab session and keeps it alive **24/7** (as far as Colab's ToS and quotas allow).
2. **A common interface** to deploy *any* ML model — scikit-learn, XGBoost, PyTorch, TensorFlow/Keras, ONNX, HuggingFace — without writing serving code.
3. **Published REST endpoints** shown in the UI, ready to be consumed by other projects through one **common API** (stable URL, stable schema, versioned).
4. **Resilience**: on disconnect, a **cache + control plane** retains state so the service re-arms in seconds, not minutes, and callers never see a hard failure.

### Non-goals (v1)

- Not a replacement for production MLOps (no autoscaling to N replicas, no SLA stronger than Colab's).
- No model *training* orchestration — inference/serving only (training can be triggered later, see §14).
- No billing/multi-tenant SaaS isolation.

---

## 2. Hard Constraints from Colab

Everything in this design flows from these facts:

| Constraint | Consequence for the design |
|---|---|
| Colab VM has **no inbound public IP / no port forwarding** | We must tunnel out (WebSocket reverse tunnel or `cloudflared`/`ngrok`) — the runtime **initiates** all connections. |
| Sessions die on **idle (~90 min)**, **max lifetime (~12 h)**, or resource pressure | The runtime is **cattle, not pets**. State lives in the control plane + artifact store, never only in the VM. |
| Restarting a notebook re-runs from scratch | **Bootstrap must be one cell, idempotent, and fully automated** (install → download artifacts → warm models → reconnect). |
| Notebook can be **resumed** only by a human/browser, or re-launched via a URL | Provide a "one-click relaunch" deep link: `https://colab.research.google.com/github/<user>/<repo>/blob/main/agent/colab_agent.ipynb#...` plus a `runtime_status` banner in the UI. |
| Free-tier GPU is shared and may be pre-empted | Gateway must degrade gracefully: cached predictions, queued async jobs, `503 + Retry-After`, and a stale-but-available fallback mode. |
| ToS: Colab is for interactive use | Keep traffic human-scale, document it, and make the *control plane* the system of record so nothing depends on a single long-lived VM. |

**Design principle:** *the Colab runtime is a disposable worker.* Anything that must survive — model registry, endpoint definitions, request/response cache, metrics, secrets — lives outside the VM.

---

## 3. High-Level Architecture

```
                                    ┌───────────────────────────────────────────┐
                                    │            CONTROL PLANE (always on)      │
                                    │  free tier: Render / Fly.io / Railway /   │
                                    │  Oracle Free / GitHub Pages (UI only)     │
   Browser                          │                                           │
 ┌────────┐   HTTPS    ┌──────────┐ │  ┌───────────┐  ┌──────────┐  ┌────────┐  │
 │  UI    │──────────▶│  API     │◀┼─▶│  Model &  │  │ Request  │  │ Health │  │
 │ (SPA)  │            │ Gateway  │ │  │ Endpoint  │  │  Cache   │  │ /Heart │  │
 └────────┘            │ (FastAPI)│ │  │ Registry  │  │ (Redis/  │  │  beat  │  │
     ▲                 └────┬─────┘ │  │ (SQLite/  │  │  SQLite) │  └────────┘  │
     │                      │       │  │  Postgres)│  └──────────┘              │
     │ SSE/WebSocket        │       │  └───────────┘                            │
     │ live status          │       │            ┌────────────────┐             │
     └──────────────────────┼───────┼───────────▶│ Relay / Tunnel │             │
                            │       │            │ (WS broker or  │             │
                            ▼       │            │ cloudflared)   │             │
                     ┌────────────┐ │            └───────┬────────┘             │
                     │  Artifact  │◀┼────────────────────┼──────────────────────┤
                     │   Store    │ │                    │                      │
                     │ (GDrive/HF │ │                    ▼                      │
                     │  /S3/R2)   │ │     ┌──────────────────────────────────┐  │
                     └────────────┘ │     │   COLAB RUNTIME (disposable)     │  │
                            ▲       │     │  ┌────────────────────────────┐  │  │
                            │       │     │  │ colab-agent (Python)       │  │  │
                            └───────┼─────┼─▶│  • bootstrap / keep-alive  │  │  │
                                    │     │  │  • reverse WS tunnel client│  │  │
                                    │     │  │  • model loader & warm pool│  │  │
                                    │     │  │  • inference server (uvicorn)│ │  │
                                    │     │  │  • local disk cache (/content)│ │  │
                                    │     │  └────────────────────────────┘  │  │
                                    │     │  GPU: T4 / CPU fallback           │  │
                                    │     └──────────────────────────────────┘  │
                                    └───────────────────────────────────────────┘
```

Three planes:

| Plane | Runs on | Lifetime | Responsibility |
|---|---|---|---|
| **Client plane** — UI SPA | GitHub Pages / Netlify / served by the API | permanent | Deploy wizard, endpoint explorer, playground, live health, logs, cache inspector |
| **Control plane** — Gateway + Registry + Cache + Relay | Free PaaS (1 small instance) | 24/7 | System of record. Auth, routing, endpoint URLs, cached predictions, health, queueing, graceful degradation |
| **Compute plane** — colab-agent | Google Colab VM | minutes → ~12 h | Loads models, runs inference, streams logs/metrics, reports heartbeats |

---

## 4. Component Design

### 4.1 `colab-agent` (runs inside the notebook)

A pip-installable Python package (`pip install colabdeploy-agent` or `pip install git+https://…`) driven by **one cell**:

```python
# Cell 1 — the entire deployment UX inside Colab
!pip -q install git+https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui#subdirectory=agent
from colabdeploy import launch
launch(token="CDP_xxx")          # token from the UI; binds this VM to your account/project
```

Responsibilities:

- **Bootstrap**: detect GPU (`nvidia-smi`, `torch.cuda`), report hardware, create `/content/.colabdeploy/{models,cache,logs}`.
- **Pull state**: fetch the desired-state manifest from the control plane (`GET /internal/runtime/desired-state`) — list of models that *should* be loaded, with artifact URIs, versions, checksums, adapter type, warm-up config.
- **Artifact hydration**: download from the artifact store, but check the **local disk cache first** and verify SHA-256. On a Colab *resume*, `/content` often survives → cold start drops from minutes to seconds.
- **Serve**: start a FastAPI/uvicorn inference server on `127.0.0.1:8000` inside the VM (never exposed directly).
- **Tunnel**: open an outbound WebSocket to the relay and multiplex HTTP frames over it (§6). Also registers a `cloudflared` quick-tunnel URL as an optional low-latency fast path.
- **Keep-alive**: heartbeat every 15 s; a lightweight background thread touches the notebook's activity so the idle timer resets (this is the honest limit of "24/7" on free Colab — see §12).
- **Self-healing**: on socket close, exponential-backoff reconnect with jitter, re-hydrate, re-warm, re-register. On model load failure, report the error to the control plane and continue serving the rest.
- **Streaming**: push logs, metrics, and prediction traces to the control plane (batched, back-pressured).

Agent internal layout:

```
agent/
├── colabdeploy/
│   ├── launch.py            # the one-cell entrypoint
│   ├── bootstrap.py         # env/GPU detection, dir setup, idempotency
│   ├── tunnel.py            # WS client, framing, reconnect, backpressure
│   ├── server.py            # local FastAPI inference app
│   ├── state.py             # desired-state poller + reconciler loop
│   ├── registry_client.py   # talk to control plane (auth via runtime token)
│   ├── cache.py             # on-disk artifact + prediction cache (SQLite)
│   ├── adapters/            # §5 pluggable model adapters
│   │   ├── base.py          # ModelAdapter protocol
│   │   ├── sklearn.py
│   │   ├── xgboost.py
│   │   ├── pytorch.py
│   │   ├── keras.py
│   │   ├── onnx.py
│   │   └── huggingface.py
│   └── metrics.py
└── colab_agent.ipynb        # copy-paste notebook with the launch cell
```

### 4.2 Control-plane API Gateway (FastAPI)

The **only** thing the outside world talks to. Single stable base URL, e.g. `https://api.colabdeploy.dev/v1`.

Roles:

- **AuthN/AuthZ**: API keys for programmatic callers, JWT/session for the UI. Runtime tokens for agents.
- **Routing**: map a published endpoint (`/v1/models/{model_id}/predict`) to a live runtime; consistent-hashing by `model_id`.
- **Degradation policy** (per endpoint, configurable): `fail` → `serve-cached` → `queue` → `reject-with-retry-after`.
- **Rate limiting & quotas** per API key (token bucket, in Redis or in-process).
- **Request/response cache** in front of the runtime (§8).
- **Async job mode** for long inferences: `POST /v1/jobs` → `202 + job_id`, poll or webhook.
- **Schema validation** of request/response using the registered model schema (Pydantic generated from the adapter's `describe()`), so callers get a stable contract even if the underlying model changes.
- **OpenAPI docs** at `/docs` — auto-generated, and this *is* the "common API" documentation surface.

### 4.3 Model & Endpoint Registry

System of record. SQLite (dev / free tier) → Postgres (scale). Tables:

```
models(id, name, framework, adapter, version, artifact_uri, checksum,
       schema_in JSON, schema_out JSON, created_at, owner_id, status)

endpoints(id, model_id, route, visibility, auth_policy, cache_ttl_s,
          stale_while_revalidate_s, degradation_mode, rate_limit, status)

runtimes(id, token_hash, colab_url, gpu, cpu, ram_mb, state,
         last_heartbeat_at, tunnel_id, capabilities JSON)

deployments(id, endpoint_id, runtime_id, model_version, state,
            deployed_at, warm, error)

predictions_log(id, endpoint_id, key_hash, status, latency_ms,
                cache_hit, created_at)          -- bounded, sampled

jobs(id, endpoint_id, payload, state, result, attempts, created_at, done_at)
```

### 4.4 Relay / Tunnel broker

Two supported transports, chosen at runtime:

1. **WS relay (default, zero external dependency)** — the control plane runs a WebSocket broker. Agent dials out; gateway pushes HTTP request frames into the agent's socket and awaits response frames. Works anywhere Colab can reach `wss://`.
2. **cloudflared quick tunnel (fast path)** — agent runs `cloudflared tunnel --url http://localhost:8000`, reports the `https://<random>.trycloudflare.com` URL to the control plane; gateway proxies to it directly. Lower latency, no frame multiplexing, but the URL changes on every restart (registry updates it on reconnect).

The relay is **stateless per request** and holds a `runtime_id → socket` map in memory, mirrored to Redis so multiple gateway instances can route.

### 4.5 UI (React + TypeScript + Vite)

Pages:

| Page | Purpose |
|---|---|
| **Dashboard** | Runtime status card (connected / stale / dead + countdown to max session life), GPU util sparkline, request volume, cache hit rate, one-click **Relaunch Colab** deep link |
| **Connect** | Paste/generate runtime token → shows the exact notebook cell + "Open in Colab" badge button |
| **Deploy** | Wizard: pick source (upload `.pkl/.pt/.onnx/.h5`, Google Drive path, HuggingFace repo, GitHub URL) → auto-detect framework → pick adapter → define input/output schema (auto-inferred where possible) → warm-up sample → Deploy |
| **Endpoints** | List of live endpoints with copy-paste `curl`/`fetch`/Python snippets, API key, docs link, cache TTL, degradation mode |
| **Playground** | Try an endpoint: form generated from the input schema, JSON editor fallback, response + latency + `X-Cache: HIT/STALE/MISS` badge |
| **Logs & Metrics** | Live tail (SSE), p50/p95 latency, error rate, per-model throughput |
| **Cache** | Inspect cached keys, hit ratio, TTLs, manual invalidate, "pin" hot keys |
| **Settings** | API keys, webhooks, offline behaviour, budget guardrails |

Live updates over **SSE** (simpler than WS through proxies, one-directional is enough); WS only if we add collaborative features.

---

## 5. The Common Model Interface (how "any model" works)

One protocol, many adapters. The adapter is the only framework-specific code.

```python
class ModelAdapter(Protocol):
    name: str                                   # "sklearn" | "pytorch" | "onnx" | ...

    def load(self, artifact_path: Path, meta: ModelMeta) -> None: ...
    def predict(self, payload: dict) -> dict: ...
    def predict_batch(self, payloads: list[dict]) -> list[dict]: ...
    def describe(self) -> Schema:               # input/output JSON schema + example
    def warmup(self, sample: dict | None) -> float:  # returns ms, primes CUDA/caches
    def health(self) -> Health: ...
    def unload(self) -> None: ...
```

Rules:

- **Payload in/out is always JSON.** Tabular models accept `{"columns": [...], "data": [[...]]}` or `{"instances": [{...}]}`; vision models accept `{"image": "<url|b64>"}`; text models accept `{"text": "..."}`; generic fallback `{"inputs": <anything>}`. The gateway normalises to the adapter's expectation via the registered schema.
- **Auto-detection** at deploy time: inspect file magic/extension (`joblib`, `torch.save`, `onnx`, `SavedModel`, `.h5`), and where ambiguous, try adapters in a ranked order in a sandboxed subprocess and pick the first that loads *and* survives `warmup()`.
- **Custom escape hatch**: `adapter: "python_function"` with a user snippet stored in the registry (`def predict(payload, model): ...`) — covers anything exotic without us writing an adapter.
- **Versioning**: each deploy creates an immutable `model_version`; endpoints pin a version or track `latest`. Rollback = repoint, instant, because artifacts stay in the store.
- **Model warm pool**: the agent keeps N most-used models in memory (LRU by traffic) and lazy-loads the rest on first request (with the request served from cache or queued while loading).

---

## 6. Request Flow

### 6.1 Happy path (runtime connected)

```
Caller ──POST /v1/models/sentiment/predict──▶ Gateway
  1. authenticate API key, rate-limit
  2. validate payload against registered input schema
  3. compute cache key = sha256(endpoint_id + model_version + canonical(payload))
  4. cache lookup ── HIT & fresh ──▶ 200 {…, "cache":"HIT"}            (≈5 ms)
  5. MISS/STALE → resolve deployment → pick live runtime (consistent hash)
  6. forward over relay WS frame (or cloudflared URL)  ── timeout budget T
  7. agent: adapter.predict() on GPU ──▶ result
  8. gateway: validate output schema, write cache (TTL), log metrics
  9. 200 {…, "cache":"MISS", "latency_ms": 84, "runtime":"rt_9f2"}
     (STALE-while-revalidate: step 4b returns stale immediately + background refresh)
```

### 6.2 Runtime dead / Colab disconnected

```
Gateway detects: heartbeat gap > 45s OR tunnel closed OR forward timeout
  → mark runtime `degraded` then `offline`
  → endpoint enters its configured degradation_mode:
      serve-cached : return best-effort cached response with 200 + "cache":"STALE"
                     + header X-ColabDeploy-Stale: true
      queue        : 202 + job_id, caller polls /v1/jobs/{id} or gets a webhook
      fail         : 503 + Retry-After + JSON error {code:"RUNTIME_OFFLINE"}
  → UI flips to orange banner, shows "Relaunch Colab" button
  → gateway keeps the endpoint record, cache, and schema — nothing is lost
  → on reconnect: agent re-registers, re-hydrates from disk cache, warms models,
    queued jobs drain automatically, UI flips green
```

---

## 7. Keep-Alive & "24/7" — the honest design

Colab will not give a free VM forever. We get *as close as possible* with layers:

| Layer | Mechanism | Effect |
|---|---|---|
| L1 | Agent heartbeat + background activity thread (small CPU touch, log line, WebSocket ping) | Defeats the **idle** timeout |
| L2 | `cloudflared`/WS socket always open with traffic | Keeps the session "interactive" |
| L3 | **Session watchdog** in the control plane: tracks `session_started_at`; before the ~12 h hard cap it warns the UI and pre-stages the relaunch | No surprise death |
| L4 | **One-click relaunch**: deep link `https://colab.research.google.com/github/<user>/<repo>/blob/main/agent/colab_agent.ipynb` + a saved token, so re-arming is 2 clicks and ~60 s | Human-in-the-loop recovery |
| L5 | **Multi-runtime pool**: register 2+ notebooks (or Colab + Kaggle + a free-tier CPU box). Gateway load-balances and fails over | True continuity without a single point of failure |
| L6 | **Cache + queue** absorb the gap so callers see latency, not errors | Perceived 24/7 |

L5 is what makes the system genuinely 24/7: the *service* is always up because the control plane and cache never depend on any one VM. The UI exposes the pool as "Runtimes" with a green/amber/red state each.

> Note on ToS: keeping a Colab session artificially busy may conflict with Google's usage policy. This project is intended for learning/prototyping; the README will state that clearly and recommend paid runtimes for production. The architecture deliberately makes the compute plane swappable — replace `colab-agent` with `docker-agent` on Modal/RunPod/Fly and **nothing else changes**.

---

## 8. Caching & State Retention

Three cache tiers, all keyed the same way so they compose:

```
cache_key = sha256(f"{endpoint_id}:{model_version}:{canonical_json(payload)}")
```

**Tier 1 — Browser / UI cache (offline-first).**
React Query with `staleTime` + IndexedDB persistence (`idb-keyval`). The UI's registry snapshot, endpoint list, last playground responses, and recent logs are all readable with **zero network**. Service worker caches the app shell → the dashboard opens instantly and shows "last known state · 3 min ago" instead of a spinner.

**Tier 2 — Control-plane cache (the important one).**
Redis if available, else an in-process LRU + SQLite-backed store (free-tier friendly).

- `response_cache`: key → `{payload, status, created_at, ttl, endpoint_id, model_version}`, LRU-evicted, size-capped (e.g. 256 MB / 500k entries).
- `stale_while_revalidate`: serve an expired entry immediately, refresh in the background.
- `hot-key pinning`: the UI can pin frequent keys so a runtime restart still serves them instantly.
- **Warm-on-reconnect**: when a runtime registers, the gateway sends the top-K hot keys as warm-up samples so the model's first real request is already hot (CUDA graphs, tokenizer caches, JIT).
- Cache is the **degradation substrate** — it's what lets an endpoint answer while Colab is dead.

**Tier 3 — Agent disk cache (`/content/.colabdeploy/cache`).**
SQLite + blob files. Caches downloaded artifacts (keyed by checksum), tokenizers, feature-engineering intermediates, and recent predictions. Survives a Colab **resume**; lost on a fresh VM, in which case Tier 2 covers us while artifacts re-download from Drive/HF (which is fast and free).

**State retention summary:**

| State | Lives in | Survives VM death? |
|---|---|---|
| Model artifacts | Artifact store (Drive / HF / S3) + agent disk cache | ✅ |
| Model versions, endpoints, schemas | Registry DB | ✅ |
| Recent predictions & hot keys | Tier-2 cache | ✅ |
| API keys, users, config | Registry DB | ✅ |
| Logs & metrics | Control plane (bounded ring buffer + optional export) | ✅ |
| Loaded weights in VRAM | GPU memory only | ❌ (re-warmed in seconds from disk cache) |
| In-flight requests | Job queue (persisted) | ✅ retried on reconnect |

---

## 9. Security

- **Secrets**: runtime tokens (`CDP_…`) shown once, stored **hashed** (argon2/sha256+salt). Caller API keys likewise hashed, with a prefix for identification.
- **Least privilege**: agent token can only read desired-state and push results for its own `runtime_id`; it cannot read the registry or other runtimes.
- **Transport**: TLS everywhere; `wss://` for the relay; certificate pinning optional.
- **Isolation**: the relay never interprets model payloads beyond framing; the agent validates sizes (max body 10 MB default) and content type.
- **Abuse**: per-key rate limits, global concurrency cap per runtime, payload size caps, and a kill-switch endpoint in the UI.
- **Custom-code adapters** are the main risk: they execute in *your own* Colab VM under your Google account — the UI states this explicitly at deploy time. No custom code ever runs in the control plane.
- **Audit log** for deploy/rollback/key-rotation actions.

---

## 10. Observability

- **Heartbeats** (15 s) carrying GPU/CPU/RAM/VRAM, loaded models, queue depth, agent version.
- **Metrics** per request: latency (gateway + agent split), cache status, token/byte size, error class. Exported as Prometheus `/metrics` from the gateway.
- **Structured logs** (JSON) shipped in batches; UI live-tails via SSE; ring buffer keeps the last 5k lines per runtime.
- **SLOs**: gateway p95 overhead < 30 ms (excluding model); cache hit ratio target > 60% for repeated payloads; reconnect → first successful prediction < 90 s.
- **Alerts**: webhook/Discord/Telegram on `runtime_offline`, `session_end_imminent`, `error_rate > 5%`.

---

## 11. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| UI | React 18 + TypeScript + Vite + Tailwind + shadcn/ui + TanStack Query + Recharts | Fast, free to host, great DX |
| Control plane | Python 3.11 + FastAPI + Uvicorn + Pydantic v2 | Same language as the agent → one shared schema package |
| DB | SQLite (SQLModel/SQLAlchemy) → Postgres when needed | Zero-ops free tier, easy migration |
| Cache | `cachetools` LRU + SQLite; Redis when available | Interface `CacheBackend` keeps both swappable |
| Relay | `websockets` / FastAPI WS + in-memory map (Redis mirror later) | No extra infra |
| Agent | Python + `httpx` + `websockets` + `uvicorn`; adapters use lazy imports | Only install what the model needs |
| Queue | SQLite-backed durable queue (v1) → Redis/RQ (v2) | Jobs must survive gateway restarts |
| Hosting | Control plane: Fly.io/Render/Oracle free tier · UI: GitHub Pages/Cloudflare Pages | Free |
| Artifacts | Google Drive (default, since Colab mounts it) · HuggingFace Hub · S3/R2 | Free + already in the user's flow |
| Packaging | `uv` workspace monorepo: `packages/shared-schema`, `apps/gateway`, `apps/ui`, `agent` | One repo, shared Pydantic contracts |
| CI | GitHub Actions: lint (ruff) + types (mypy) + tests (pytest) + UI build + agent smoke test | Cheap and fast |

---

## 12. Repository Layout

```
deploy-ml-models-for-free-colab-ui/
├── architecture.md              ← this file
├── README.md
├── agent/                       # Colab-side package
│   ├── colab_agent.ipynb        # the one-cell launcher
│   ├── colabdeploy/             # §4.1
│   └── pyproject.toml
├── apps/
│   ├── gateway/                 # control plane (FastAPI)
│   │   ├── app/{main,api,core,models,services,relay,cache}.py
│   │   └── pyproject.toml
│   └── ui/                      # React SPA
│       └── src/{pages,components,api,hooks,lib}
├── packages/
│   └── shared-schema/           # Pydantic contracts used by agent + gateway
├── deploy/                      # fly.toml / render.yaml / Dockerfile
├── docs/                        # api.md, deployment.md, adapters.md, runbook.md
└── .github/workflows/ci.yml
```

---

## 13. API Surface (v1 contract)

```
# Public — the "common API" consumers use
POST   /v1/models/{model_id}/predict          sync inference
POST   /v1/models/{model_id}/predict:batch    batch
GET    /v1/models/{model_id}/schema           input/output JSON schema
GET    /v1/endpoints                          list published endpoints
POST   /v1/jobs · GET /v1/jobs/{id}           async / long-running
GET    /v1/health · GET /v1/status            service + runtime state
GET    /v1/metrics                            prometheus text

# Management — used by the UI
POST   /api/auth/login · /api/keys
GET    /api/runtimes · POST /api/runtimes/token
POST   /api/models                            register/upload
POST   /api/models/{id}/deploy                pin to endpoint + runtime
POST   /api/endpoints · PATCH /api/endpoints/{id}
GET    /api/logs/stream (SSE) · GET /api/cache · DELETE /api/cache/{key}

# Internal — agent only (runtime-token auth)
GET    /internal/runtime/desired-state
POST   /internal/runtime/heartbeat
WS     /internal/relay                        # request/response framing
POST   /internal/runtime/logs (batched)
```

Every response is versioned and additive-only; breaking changes go to `/v2`.

---

## 14. Roadmap

| Milestone | Scope | Exit criteria |
|---|---|---|
| **M0 — Foundations (wk 1)** | Monorepo scaffold, CI, shared schema, `ModelAdapter` protocol, SQLite registry | `pytest` green, `ruff`/`mypy` clean |
| **M1 — Spike: prove the tunnel (wk 1–2)** | Colab notebook runs FastAPI locally; WS relay in the gateway forwards one sklearn prediction end-to-end; heartbeat + reconnect | `curl` from a laptop → Colab GPU → JSON back, and it recovers after killing the notebook |
| **M2 — Deploy any model (wk 2–3)** | Artifact upload (Drive/HF/upload), auto-detect framework, sklearn + pytorch + onnx adapters, warm-up, versioning | Deploy 3 different model types with zero serving code |
| **M3 — UI (wk 3–4)** | Dashboard, Connect, Deploy wizard, Endpoints, Playground; SSE live status | A new user goes from notebook to a working public endpoint in < 10 min |
| **M4 — Cache & resilience (wk 4–5)** | Tier-2 cache, stale-while-revalidate, degradation modes, job queue, hot-key warm-up, UI offline mode | Pull the plug on Colab → endpoint still answers from cache; reconnect → drains queue automatically |
| **M5 — Multi-runtime & keep-alive (wk 5–6)** | Runtime pool, failover, session watchdog, relaunch deep link, alerts | Two notebooks: kill one, zero dropped requests |
| **M6 — Polish & publish (wk 6+)** | API keys/quotas, docs site, `pip install colabdeploy-agent`, demo video, HF/Gradio interop | Public README + live demo endpoint |
| **Later** | Training jobs, autoscaling to Modal/RunPod, canary deploys, A/B endpoints, embeddings/vector search endpoint, webhook subscriptions | — |

---

## 15. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Colab hard-kills the session mid-request | Dropped requests | Idempotent retries, persisted job queue, timeout budget + `Retry-After` |
| ToS violation for artificial keep-alive | Account warning | Document clearly; make keep-alive opt-in; design for swappable compute (§7) |
| `trycloudflare` URL churn | Broken fast path | Registry re-registers on reconnect; WS relay is the always-available fallback |
| Model too big for `/content` disk or VRAM | Deploy fails | Pre-flight size check in the wizard; quantisation hints; CPU fallback; clear error with remediation |
| Cold start after fresh VM (re-download weights) | Slow first request | Disk cache on resume, Drive/HF fast pulls, warm-up on register, serve stale cache meanwhile |
| Free PaaS sleeps the control plane | Whole system unreachable | Prefer always-on free tiers (Fly/Oracle); health-check pinger; UI hosted separately on static CDN |
| Single-gateway SPOF | Outage | Stateless gateway → scale horizontally; relay map in Redis when we get there |
| Secret leakage in a shared notebook | Compromised account | Tokens are scoped + revocable from the UI; short default TTL; never logged |

---

## 16. Decisions Log

| # | Decision | Rationale | Alternatives rejected |
|---|---|---|---|
| D1 | Colab agent **dials out** over WebSocket | Colab has no inbound ingress | Port forwarding, SSH — impossible |
| D2 | Control plane is the **system of record** | VMs are disposable | Stateful notebook-only design — dies with the VM |
| D3 | **FastAPI everywhere** (agent + gateway) | One mental model, shared Pydantic schema | gRPC (heavier in Colab), Flask (no async WS) |
| D4 | **Adapter protocol** for model types | "Any model" without per-model serving code | Force everything to ONNX (lossy for HF/torch), Triton (too heavy for free tier) |
| D5 | **SQLite default**, Postgres optional | Zero-ops free tier | Postgres-first — friction for a free project |
| D6 | Cache is a **first-class resilience feature**, not an optimisation | It's what makes disconnects invisible | Cache only for speed — endpoint dies with the VM |
| D7 | UI is **offline-first** (SW + IndexedDB) | Users watch a flaky connection; the UI must stay legible | Server-rendered only |
| D8 | Monorepo with **shared schema package** | Contract drift between agent/gateway is the #1 integration bug | Separate repos |

---

## 17. Next Actions

1. Scaffold the monorepo (§12) + CI.
2. Build **M1 spike** end-to-end — that single proof de-risks the whole project.
3. Write `docs/api.md` from the shared schema, and `agent/colab_agent.ipynb`.
4. Then the UI.
