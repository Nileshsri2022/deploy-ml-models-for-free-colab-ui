# Adding models from the UI — no notebook code

The control plane can serve **any** model, and most of them need zero code from you:
you fill in one form (`Models → Register by URI`), press **Deploy**, and the agent on
your VM does the loading.

This folder holds the exceptions — models whose "model" is really *a pipeline plus
custom preprocessing*. You point the form at one of these files by URL and still
write nothing yourself.

## Which form do I use?

| Your model | Artifact URI | Adapter | Entrypoint | Code needed |
| --- | --- | --- | --- | --- |
| scikit-learn / joblib pickle | `file:///content/models/churn.joblib` (or a URL, Drive id, `store://…`) | `sklearn` or `auto` | — | none |
| ONNX Runtime | `…/model.onnx` | `onnx` or `auto` | — | none |
| XGBoost | `…/model.ubj` | `xgboost` or `auto` | — | none |
| Keras / TF SavedModel | `…/model.keras` or a SavedModel dir | `keras` or `auto` | — | none |
| PyTorch / TorchScript | `…/model.pt` | `pytorch` or `auto` | — | none |
| Hugging Face **transformers** model | `/content/models/<name>` (downloaded snapshot) or a single-segment `hf://repo` | `huggingface` | pipeline task, e.g. `text-classification` | none |
| Custom function, ensemble, heuristic | `inline:…`, a `.py` path, or a **URL to a `.py`** | `python_function` | callable name, e.g. `predict` | a `predict(payload, model=None, meta=None)` function |
| **diffusers pipeline** (Stable Diffusion, ControlNet, SDXL…) | a **URL to a `.py`** like the one below | `python_function` | `predict` | none — use the file in this folder |

Two reasons diffusers needs an entrypoint:

1. The built-in `huggingface` adapter builds a `transformers.pipeline()` from the
   repo. That cannot express "ControlNet weights **+** a base checkpoint **+** an
   image as the conditioning input" — that is a *diffusers* pipeline.
2. Anything beyond a single file — QR preprocessing, base64 encoding the PNG,
   clamping steps, seeding — is your pipeline's job, so the gateway needs a callable.

## QR Code Monster: register it from the UI

Colab: switch to a **GPU runtime** first, then run one install cell (this is the only
"command" involved — the code already lives in this repo):

```python
!pip -q install -U diffusers transformers accelerate safetensors qrcode pillow
!pip -q uninstall -y torchao        # Colab's torchao is older than diffusers expects
```

### Pitfall: `cannot import name 'FqnToConfig' from 'torchao.quantization'`

Colab preinstalls an old `torchao` (0.14.x), while current diffusers imports
`torchao.quantization.FqnToConfig` unconditionally — and that name only exists from
**torchao 0.15.0**. So `import diffusers` dies before this file runs a single line,
and the agent reports it as a `MODEL_ERROR` raised by `predict()`.

Fix it in Colab, in this order:

```python
!pip -q uninstall -y torchao        # recommended: diffusers only needs it for quantization
# or, if you need torchao:
!pip -q install -U torchao          # >= 0.15 has FqnToConfig
```

Then re-attach the runtime (`launch(...)` cell again, or `Runtimes → Reconcile` in the
UI) and call the endpoint once more. Nothing in the deployment has to change — the model
is already registered and deployed; only the VM's package set was wrong.

On the gateway (Render → your service → Environment), raise the timeout, because a
diffusion step is slower than the 30 s default:

```
CDP_TIMEOUT_MS=120000
```

Then, in the UI, `Models → Register by URI`:

| Field | Value |
| --- | --- |
| Artifact URI | `https://raw.githubusercontent.com/<owner>/<repo>/<branch>/examples/qr_monster_predict.py` |
| Name | `qrmonster` |
| Framework hint | `auto` |
| Adapter | `python_function` |
| Entrypoint | `predict` |
| Checksum | *(empty)* |
| Tags | `diffusion, controlnet, qr` |
| Input schema | see below |
| Warm-up sample | `{"prompt": "warmup", "qr_text": "warmup", "steps": 2, "size": 512}` |

Input schema:

```json
{
  "fields": {
    "prompt": {"type": "string", "description": "what the image should look like"},
    "qr_text": {"type": "string", "description": "what the QR code should decode to"},
    "steps": {"type": "integer", "description": "1-40, default 20"},
    "scale": {"type": "number", "description": "controlnet strength, 0-2, default 1.3"},
    "size": {"type": "integer", "description": "512 (default) or 768"}
  },
  "required": ["prompt"],
  "example": {"prompt": "a lush jungle with a hidden glowing monster", "qr_text": "https://qrcode.monster"}
}
```

Then **Registry → Deploy**:

| Wizard field | Value |
| --- | --- |
| Cache TTL (s) | `300` — one seed per prompt is generated once, repeats are instant |
| Serve-stale window (s) | `3600` |
| timeout_ms | `120000` |
| rate_limit_rpm | `30` (a GPU is a single lane) |
| Degradation mode | `serve_cached` |

Call it from anywhere:

```bash
curl -s -X POST "$GW/v1/models/qrmonster/predict" \
  -H "Authorization: Bearer $CDP_API_KEY" -H 'Content-Type: application/json' \
  -d '{"inputs": {"prompt": "a glowing cyberpunk city, neon", "qr_text": "https://qrcode.monster",
                  "steps": 25, "scale": 1.4, "seed": 42}}' \
| python3 -c "import sys,json,base64; d=json.load(sys.stdin)['output']; open('qr.png','wb').write(base64.b64decode(d['image_b64']))"
```

`image_b64` is the generated PNG and `qr_b64` is the QR code that was used as the
condition, so you can compare them side by side. Not every seed scans — vary
`seed`, `scale` and the prompt; a gray (`#808080`) QR background is what the model
card recommends, and that is what this file generates.

## Notes

* The first request after a deploy loads ~4 GB of weights (base SD 1.5 + the
  ControlNet) — the warm-up sample pays that cost at load time, so it happens while
  the deployment is still `loading`, not while a caller waits.
* Keep the file URL pinned to a branch you control; the artifact URL is fetched once
  and then cached on the runtime by content hash, so later deploys are instant.
* Everything here runs on **your** VM. The control plane only stores the URL.
