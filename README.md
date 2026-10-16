# ColabDeploy

**Deploy any ML model from a free Colab session behind one stable API — with a UI,
published endpoints, and a cache that keeps answering when the notebook dies.**

You already have a GPU in Colab and a trained model. What Colab does *not* give you
is a public URL, a process that survives idle, or anywhere to put state. This project
supplies the missing three-quarters: a control plane you host for free, an agent that
runs in one notebook cell, and a UI that ties them together.

```
                 your callers
                      │  POST /v1/models/churn/predict
                      ▼
   ┌──────────────────────────────────────────────┐
   │  control plane  (one FastAPI process, free)  │
   │  registry · cache · job queue · relay · UI   │
   └──────────────────────────────────────────────┘
                      ▲  outbound websocket (no ports, no ngrok)
                      │
   ┌──────────────────┴───────────────┐   ┌────────────────────────┐
   │ Colab VM · agent · adapters      │   │ another VM (Kaggle,    │
   │ sklearn / torch / onnx / hf / …  │   │ Modal, your laptop)    │
   └──────────────────────────────────┘   └────────────────────────┘
```

The VM initiates every connection, which is the only reason this works on a free
notebook at all. When it disappears, the control plane keeps serving from cache,
queues what it can't, and re-warms the models the moment a VM returns.

---

## See it work in one command

No Colab required — this boots the gateway, attaches a **real agent** in a thread,
registers two demo models and warms the cache:

```bash
git clone https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui.git
cd deploy-ml-models-for-free-colab-ui

python -m venv .venv && . .venv/bin/activate
pip install -e packages/shared-schema -e apps/gateway -e agent

cd apps/ui && npm install && npm run build && cd ../..   # optional: builds the UI
python scripts/e2e_local.py
```

Then open **http://127.0.0.1:8000/** and paste the printed API key into
*API keys → This browser*. You'll land on a dashboard with one runtime online, two
endpoints serving, and a cache that already has entries in it.

```bash
curl -X POST http://127.0.0.1:8000/v1/models/demo-sentiment/predict \
  -H "Authorization: Bearer $CDP_API_KEY" -H 'Content-Type: application/json' \
  -d '{"inputs": {"text": "this is a great little demo"}}'
# → {"output": {"label": "positive", "score": 2, …}, "cache": "MISS", "latency_ms": 2.1}
# run it again → "cache": "HIT", "latency_ms": 0.1  (the VM was never involved)
```

Kill the agent thread (or the whole script) mid-request and re-run: the endpoint
answers `STALE` from cache while nothing is attached, then the reattached VM
reconciles and re-warms on its own. That behaviour is the point of the design, and
it is covered by tests rather than by hope.

## Now do it with a real Colab GPU

1. **Host the control plane** somewhere free and public — Render, Fly.io, Railway,
   Oracle Always-Free. See [`docs/deployment.md`](docs/deployment.md).
2. **Mint a runtime token**: UI → *Connect a runtime* → Mint token.
3. **Open the notebook**: [**agent/colab_agent.ipynb**](https://colab.research.google.com/github/Nileshsri2022/deploy-ml-models-for-free-colab-ui/blob/main/agent/colab_agent.ipynb)
   and paste the URL + token into cell 2. Run cells 1–3.

```python
!pip -q install \
  "git+https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui#subdirectory=packages/shared-schema" \
  "git+https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui#subdirectory=agent"

from colabdeploy import launch
launch(gateway_url="https://your-gateway.onrender.com", token="CDP_RT_…")
```

The notebook stays interactive — the tunnel lives in a daemon thread, so you can
keep training while the VM serves.

4. **Deploy a model.** Either register it in the UI (upload a file, or point at
   Drive/Hugging Face/HTTPS), or register the file you just trained *without
   moving it*:

```python
import joblib, httpx
joblib.dump(model, "/content/model.joblib")

client = httpx.Client(base_url=GW, headers={"Authorization": f"Bearer {MANAGE_KEY}"})
created = client.post("/api/models", json={
    "name": "churn", "adapter": "sklearn",
    "artifact": {"uri": "file:///content/model.joblib"},
    "schema_in": {"fields": {"tenure": {"type": "number"}}, "required": ["tenure"],
                  "example": {"tenure": 24}},
}).json()
client.post("/api/deploy", json={"model_id": created["id"]})
# → https://your-gateway.onrender.com/v1/models/churn/predict is live
```

Re-registering the same name creates **v2, v3, …**; rolling back is one click, and
the cache key includes the version so a rollback can never serve the wrong answers.

---

## What's in the box

| | |
| --- | --- |
| **Deploy any model** | adapters for scikit-learn, PyTorch, ONNX, XGBoost, Keras/TensorFlow, Hugging Face, plus a `python_function` escape hatch for anything custom. `adapter: "auto"` detects by name first, then probes cheapest-first and reports every failure it collected. Frameworks are imported lazily, so the agent costs nothing until you use it. |
| **One common API** | `POST /v1/models/{route}/predict`, `:batch`, `/schema`, `/v1/endpoints`, `/v1/status`, `/v1/metrics`, `/v1/jobs`. One response envelope whichever framework produced it. |
| **A UI that stays useful** | Dashboard, Connect, Models, Endpoints, Playground, Cache inspector, Runtimes, Logs (live SSE tail), API keys. Every read is shelved in `localStorage`, so when the control plane is unreachable the UI shows "last known state · 2 min ago" instead of going blank. |
| **Cache that survives disconnects** | two-tier: fresh entries answer without touching a VM, stale entries answer *while* the gateway re-fetches (`stale-while-revalidate`), and the top-K hot keys are replayed to a reconnecting agent as warm-up samples so the requests that matter are warm first. Snapshots to disk so a gateway restart doesn't empty it. |
| **Declared, not commanded** | a deploy writes intent to the registry. A VM reconciles against desired state on connect and on every change, so you can deploy with nothing online and it just works when a VM appears. |
| **Observable** | request ids, latency percentiles, cache hit ratio, relay health, agent logs batched back over the tunnel, an audit trail, Upstash failure diagnostics, Prometheus at `/v1/metrics`, and optional Sentry error reporting/issues in Logs. |

Full reference: [`docs/api.md`](docs/api.md) · design rationale: [`architecture.md`](architecture.md)

## Resilience, concretely

Each endpoint carries its own policy for "no VM can answer right now":

| `degradation_mode` | Caller sees |
| --- | --- |
| `serve_cached` *(default)* | `200` with the last known answer, `cache: "STALE"`, `stale: true` |
| `queue` | `202` + `job_id`; executed automatically when a runtime returns |
| `fail` | `503` + `Retry-After` |

Runtime lifecycle is derived from heartbeats — `registered → connecting → online →
stale → offline` — and the agent reconnects forever with exponential backoff plus
jitter. An exception inside the agent becomes a `500 {"code": "AGENT_ERROR"}` *body*
on that one request; it never takes the tunnel down.

## Repository layout

```
architecture.md              the design document (read this first)
packages/shared-schema/      Pydantic contracts both sides speak
apps/gateway/                control plane: FastAPI, registry, relay, cache, UI hosting
apps/ui/                     React + TypeScript + Vite control plane UI
agent/                       colabdeploy: the notebook-side runtime + adapters
  colab_agent.ipynb          the one-click Colab notebook
scripts/e2e_local.py         gateway + agent + seeded model, one command
tests/                       244 tests, including real-tunnel end-to-end
docs/                        api.md, deployment.md
deploy/                      Dockerfile, render.yaml, fly.toml
```

## Development

```bash
pip install -e packages/shared-schema -e apps/gateway -e agent
pip install pytest ruff
pytest tests/ -q                     # 244 passed in ~19s
ruff check .
cd apps/ui && npm install && npm run dev     # UI on :5173, proxying to :8000
```

`pytest tests/test_e2e_agent.py -q` is the interesting one: it boots a real uvicorn
server and a real agent thread over one websocket, then asserts what a caller sees
when the VM disappears and comes back.

## Status against the roadmap

| Milestone | State |
| --- | --- |
| M0 foundations (monorepo, shared schema, adapter protocol, registry, CI) | ✅ |
| M1 prove the tunnel (relay, heartbeat, reconnect, e2e test) | ✅ |
| M2 deploy any model (artifacts, auto-detect, adapters, warm-up, versioning) | ✅ |
| M3 UI (dashboard, connect, deploy wizard, endpoints, playground, SSE) | ✅ |
| M4 cache & resilience (SWR, degradation modes, job queue, hot-key warm-up, UI offline) | ✅ |
| M5 multi-runtime & keep-alive (pool, failover, watchdog, relaunch link) | 🟡 partial — ready replicas support one automatic retry after a runtime transport failure; watchdog and relaunch link exist, but alerting is not implemented |
| M6 polish & publish (PyPI packages, quotas, demo video) | 🟡 partial — daily key quotas and build/release workflow are implemented; PyPI trusted-publisher setup and first release, plus recording the demo video, remain |

## Limitations — please read

* **Colab ends every session after ~12 hours** and nothing inside the VM can stop
  it. The keep-alive thread defeats the ~90-minute *idle* timeout only. The UI shows
  a countdown and a one-click relaunch link, and re-attaching is one cell run
  (artifacts are cached, deployments reconcile automatically) — but a human has to
  run it. This is not 24/7 in the way a paid box is.
* **Keeping a free session artificially busy may conflict with Google's usage
  policy.** The browser-side keep-alive (`browser_activity=True`) is the part most
  likely to, so it is off by default and should stay off. Use this for learning,
  prototyping and demos.
* **`python_function` runs your code** — in your own VM, under your own account,
  never in the control plane. That is the safety model, and it is only as safe as
  the notebook you ran it in.
* **One gateway process, one SQLite/JSON store.** Fine for a hobby project or a
  class demo; not a multi-tenant SaaS.
* **The compute plane is deliberately swappable.** Point the same agent at Modal,
  RunPod, Fly Machines or a laptop and *nothing else changes* — that is the escape
  hatch when you outgrow Colab.

## License

MIT.
