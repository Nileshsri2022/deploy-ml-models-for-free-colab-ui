# ColabDeploy agent

Turns a Colab (or Kaggle, or any Linux) VM into a supervised inference runtime
with **one cell**. The VM dials *out* to your control plane over a websocket, so
there is no public IP, no port forwarding and no ngrok — which is the only reason
this works on a free notebook at all.

```python
!pip -q install \
  "git+https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui#subdirectory=packages/shared-schema" \
  "git+https://github.com/Nileshsri2022/deploy-ml-models-for-free-colab-ui#subdirectory=agent"

from colabdeploy import launch
launch(gateway_url="https://your-gateway.onrender.com", token="CDP_RT_…")
```

That's it. `colab_agent.ipynb` in this directory is the same thing as a
[one-click Colab notebook](https://colab.research.google.com/github/Nileshsri2022/deploy-ml-models-for-free-colab-ui/blob/main/agent/colab_agent.ipynb).

```
colabdeploy/
├── launch.py          launch() / status() / stop() — the notebook-facing API
├── bootstrap.py       workspace layout, GPU probe, RuntimeCapabilities
├── tunnel.py          websocket client: handshake, heartbeats, reconnect forever
├── runtime.py         the sync route table the tunnel dispatches into
├── model_store.py     LRU pool of loaded models + reconcile against desired state
├── artifacts.py       blob cache (SQLite index) + resolver for every URI scheme
├── registry_client.py talks to the gateway's /internal API
├── keepalive.py       defeats the idle timeout, warns before the 12h cap
├── state.py           reconciler helpers shared by launch() and the tunnel
└── adapters/          echo, python_function, sklearn, pytorch, onnx, xgboost,
                       keras, huggingface
```

## What happens when you run the cell

1. **Bootstrap** — picks a workspace (`/content/.colabdeploy`), probes the GPU
   (`torch.cuda`, then `nvidia-smi`), counts CPUs/RAM/disk, and lists the adapters
   whose frameworks are actually importable.
2. **Register** — `POST /internal/runtime/register` with those capabilities; the
   gateway replies with a `runtime_id`, the relay websocket URL and the desired
   state it currently wants.
3. **Connect** — a daemon thread runs an asyncio loop and opens
   `WS /internal/relay`. Hello → HelloAck, then heartbeats every 15 s.
4. **Reconcile** — for each wanted deployment: resolve the artifact (cache first),
   pick an adapter, load it, then warm it up. Hot cache keys from the gateway are
   replayed as warm-up samples so the *useful* paths are hot, not arbitrary ones.
5. **Serve** — exec frames arrive on the socket, run in a small thread pool
   (4 workers), and reply with JSON. Logs are batched back to the gateway.

Re-running the cell is safe: the previous session is stopped first, and artifacts
come from the on-disk cache, so a resume takes seconds rather than minutes.

## Adapters

Set `adapter` explicitly to skip probing, or leave it `auto`:

| Adapter | Priority | Loads | Extra install |
| --- | --- | --- | --- |
| `python_function` | 5 | your `entrypoint` source, or the artifact as a `.py` file | — |
| `echo` | 10 | a JSON config; returns a hash of the payload | — |
| `onnx` | 15 | `.onnx` via `onnxruntime` | `onnxruntime` |
| `xgboost` | 18 | `.json`/`.ubj`/`.bst` | `xgboost` |
| `sklearn` | 20 | `.pkl`/`.joblib` pickles | `scikit-learn`, `joblib` |
| `huggingface` | 25 | `hf://repo[/file]` via `transformers` | `transformers` |
| `pytorch` | 30 | `.pt`/`.pth` state dicts or scripted modules | `torch` |
| `keras` | 40 | `.h5`/`.keras`/SavedModel | `tensorflow` |

`auto` works in two stages: **detect** (cheap name/URI inspection, no heavy
imports) and then **probe** (try `load()` in priority order and keep the first
that survives a warm-up call). Every failure is collected, so the UI shows
"tried sklearn: …; tried onnx: …" rather than a mystery error.

Heavy frameworks are imported *inside* `load()`. Installing the agent costs
nothing on a VM that only has scikit-learn, and `available_adapters()` tells the
gateway what this particular machine can serve.

## Artifact URIs

| URI | Meaning |
| --- | --- |
| `file:///content/model.pkl` or a bare path | already on the VM |
| `https://…` / `http://…` | plain download |
| `gdrive://<file_id>` | Google Drive (uses `gdown`) |
| `hf://<repo>[/<file>]` | Hugging Face Hub; a bare repo is handled by the pipeline |
| `store://art_<id>` | uploaded through the UI, served back by the gateway |
| `inline:<json or source>` | the bytes are in the URI itself |

Downloads stream to a `.part` file and are `os.replace`d into place, so a
half-written blob can never be loaded. If you supply `checksum`, a mismatch
deletes the file and fails the deployment loudly. Artifacts are cached under
`.colabdeploy/cache` keyed by checksum (or by URI hash), with an 8 GB LRU prune so
a long-lived session cannot fill the disk.

## Keeping the session alive — and the honest caveat

Colab ends a session for two unrelated reasons:

* **Idle timeout (~90 min).** The keep-alive thread does a few milliseconds of I/O
  per minute, which resets this. The open websocket also counts as activity on
  most Colab builds.
* **Maximum session length (~12 h).** Nothing running inside the VM can prevent
  this. The agent warns through the gateway as the cap approaches so the UI can
  show a countdown and a one-click relaunch link; re-attaching then costs one cell
  run because the artifact cache and desired state are already in place.

> **Please read this.** Artificially keeping a free Colab session busy may conflict
> with Google's usage policy, and `browser_activity=True` (which drives the Colab
> frontend from inside the notebook) is the part most likely to. It is **off by
> default** and you should leave it off. This project is meant for learning,
> prototyping and demos. For anything production-shaped, point the same agent at a
> paid runtime — the compute plane is deliberately swappable, and nothing in the
> control plane changes when you do.

## From the notebook

```python
import colabdeploy
colabdeploy.status()   # tunnel state, loaded models, counters, session budget
colabdeploy.stop()     # unload models, close the tunnel, stop keep-alive
```

`launch()` returns immediately (the tunnel lives in a daemon thread), so the
notebook stays interactive — you can keep training in other cells while it serves.
Pass `block=True` if you would rather the cell hold the session open.

## Deploying a model you just trained

The artifact never has to leave the VM:

```python
import joblib, httpx
joblib.dump(model, "/content/model.joblib")

gw = "https://your-gateway.onrender.com"
key = "CDP_API_…"                      # manage scope
client = httpx.Client(base_url=gw, headers={"Authorization": f"Bearer {key}"})

created = client.post("/api/models", json={
    "name": "churn",
    "adapter": "sklearn",
    "artifact": {"uri": "file:///content/model.joblib"},
    "schema_in": {"fields": {"tenure": {"type": "number"}}, "required": ["tenure"],
                  "example": {"tenure": 24}},
    "warmup_sample": {"inputs": {"tenure": 24}},
}).json()

client.post("/api/deploy", json={"model_id": created["id"]})
# → POST {gw}/v1/models/churn/predict  is live on this VM
```

Because the artifact is a local path, only *this* VM can serve it — which is
exactly what you want while iterating, and a Drive/HF URI when you want any VM to
be able to pick it up.

## Local development

```bash
pip install -e packages/shared-schema -e agent
python scripts/e2e_local.py            # gateway + agent + seeded model, one command
pytest tests/test_adapters.py tests/test_model_store.py tests/test_artifacts.py -q
pytest tests/test_e2e_agent.py -q      # real tunnel, no Colab involved
```
