# Hosting the control plane (for free)

The control plane is one Python process with no external dependencies: no Postgres,
no Redis, no message broker. State lives in `CDP_DATA_DIR` as JSON/SQLite plus an
optional cache snapshot. That is what makes the free tier viable.

The only hard requirement is **websockets on the public URL**, because every VM
dials into `WS /internal/relay`. Everything else is ordinary HTTPS.

| Host | Free tier | Persistent disk | Websockets | Notes |
| --- | --- | --- | --- | --- |
| **Render** | yes (spins down after 15 min idle) | paid only | ✅ | easiest: push a blueprint |
| **Fly.io** | limited allowance, needs a card | ✅ volumes | ✅ | best fit: `min_machines_running = 1` keeps tunnels alive |
| **Railway** | trial credit | ✅ | ✅ | one-click from the repo |
| **Oracle Cloud Always Free** | genuinely free, no spin-down | ✅ | ✅ | most work, best uptime: an ARM VM you own |
| **Your own box / a $5 VPS** | — | ✅ | ✅ | `systemd` + Caddy is ~20 lines |
| **Google Cloud Run** | yes | ❌ (ephemeral) | ✅ | works, but state dies with the container — use `CDP_CACHE_BACKEND=sqlite` on a mounted volume or accept it |

Avoid anything that terminates idle connections aggressively (some serverless
gateways kill a socket at 60 s). If you must, raise the agent's heartbeat rate:
`launch(heartbeat_interval_s=10)`.

---

## Option A — Docker (any host)

```bash
docker build -f deploy/Dockerfile -t colabdeploy .
docker run -d --name colabdeploy -p 8000:8000 \
  -v cdp-data:/data \
  -e CDP_ADMIN_KEY="CDP_API_$(openssl rand -hex 16)" \
  -e CDP_PUBLIC_URL="https://your-host.example.com" \
  colabdeploy
```

The image builds the SPA in a Node stage and ships only the static files, so the
runtime image has no Node and — more importantly — **no agent code**. Nothing in
the control plane can execute model code; that boundary is deliberate.

## Option B — Render

[`deploy/render.yaml`](../deploy/render.yaml) is a blueprint: *New → Blueprint →
point at this repo*. Set `CDP_ADMIN_KEY` and `CDP_PUBLIC_URL` when prompted.

The free tier spins down after 15 minutes without traffic and cold-starts in ~30 s.
Tunnels drop when it does; agents reconnect on their own with backoff, so nothing
is lost except the seconds it takes to come back. If the container is recycled,
in-memory state goes with it unless you attach a disk.

## Option C — Fly.io

```bash
fly launch --copy-config --path deploy --no-deploy
fly volumes create cdp_data --size 1 --region ams
fly secrets set CDP_ADMIN_KEY="CDP_API_$(openssl rand -hex 16)" \
                CDP_PUBLIC_URL="https://<app>.fly.dev"
fly deploy --dockerfile deploy/Dockerfile
```

[`deploy/fly.toml`](../deploy/fly.toml) sets `auto_stop_machines = false` and
`min_machines_running = 1`: a stopped machine drops every tunnel, which is exactly
what you don't want.

## Option D — bare metal / VPS with systemd

```ini
# /etc/systemd/system/colabdeploy.service
[Unit]
Description=ColabDeploy control plane
After=network-online.target

[Service]
User=colabdeploy
WorkingDirectory=/opt/colabdeploy
Environment=CDP_DATA_DIR=/var/lib/colabdeploy
Environment=CDP_ADMIN_KEY=CDP_API_...
Environment=CDP_PUBLIC_URL=https://your-host.example.com
ExecStart=/opt/colabdeploy/.venv/bin/uvicorn gateway.main:app \
  --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips '*' \
  --ws-ping-interval 20 --ws-ping-timeout 60
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Put Caddy or nginx in front for TLS. With nginx, websockets need the explicit
upgrade headers on `/internal/relay`:

```nginx
location /internal/relay {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;      # long-lived tunnel, not a request
    proxy_buffering off;
}
location /api/logs/stream {         # SSE needs buffering off too
    proxy_pass http://127.0.0.1:8000;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
}
```

---

## First boot

1. The gateway mints a bootstrap admin key and prints it **once** to stdout
   (`docker logs` / `fly logs` / Render's log tab). Copy it immediately.
2. Open `https://your-host/` — the UI. Paste the key into *API keys → This browser*.
3. *Connect a runtime → Mint token* → paste the generated cell into
   [`agent/colab_agent.ipynb`](../agent/colab_agent.ipynb).
4. *Models → Register*, then *Deploy*. The endpoint URL is in the response, with a
   ready-made curl.

If you would rather create keys first, set `CDP_ADMIN_KEY` yourself — the value you
provide becomes the bootstrap key instead of a generated one.

## Configuration that matters in production

| Variable | Recommended | Why |
| --- | --- | --- |
| `CDP_PUBLIC_URL` | your public origin | generated curl snippets, the relay URL handed to agents and the Colab relaunch link all use it. Behind a proxy the request host can be internal |
| `CDP_REQUIRE_AUTH` | `1` | the default; never turn this off on a public host |
| `CDP_ALLOW_ANONYMOUS_INFERENCE` | `0` | `1` lets anyone call your models without a key |
| `CDP_CORS_ORIGINS` | your UI origin(s) | `*` is fine while the UI is served from the same origin |
| `CDP_CACHE_BACKEND` | `memory` + `CDP_CACHE_PERSIST=1`, or `sqlite` | sqlite keeps entries across restarts without a snapshot file; memory is faster |
| `CDP_CACHE_MAX_BYTES` | fit your RAM | the cache is the thing keeping you alive during an outage |
| `CDP_SESSION_MAX_HOURS` | `12` | drives the countdown shown in the UI |
| `CDP_MAX_BODY_BYTES` / `CDP_MAX_BATCH` | per model | a 10 MB body limit and 64-row batch cap are the defaults |
| `CDP_DAILY_INFERENCE_QUOTA` | `1000` | maximum authenticated inference HTTP requests per API key per UTC day; `0` disables the quota |

**Run one worker.** The relay holds each tunnel's websocket in-process; `--workers 4`
would give you four gateways that cannot see each other's runtimes, and requests
would randomly fail. Scale the VMs, not the control plane. If you ever genuinely
need horizontal scale, the relay is the piece to replace (Redis pub/sub or NATS),
and nothing else in the codebase has to change.

## Operations

**Backups.** Everything worth keeping is under `CDP_DATA_DIR`: the registry
(models, endpoints, deployments, keys as hashes, audit), uploaded artifacts, and the
cache snapshot. Copy that directory and you can restore on another host in minutes.

**Upgrading.** `git pull && docker build && deploy`. The registry is additive and
the wire protocol is versioned (`Hello.protocol = 1`); an older agent connecting to
a newer gateway is rejected with a clear message rather than misbehaving. Rebuild the
UI (`npm run build`) whenever `apps/ui/src` changes — the image does it for you.

**Rotating the admin key.** Create a new key with the `manage` scope in the UI, then
revoke the bootstrap key. `CDP_ADMIN_KEY` only seeds the first boot.

**A runtime token leaked.** *Runtimes → rotate*. The old token dies immediately and
the VM needs one cell re-run.

## Runbook: what to do when

| Symptom | First check | Then |
| --- | --- | --- |
| UI says "no runtime online" | *Runtimes* → state and last heartbeat | Colab recycled the session: re-run the notebook's launch cell. Same token, same deployments |
| Runtime stuck at `registered` | the agent cannot reach `WS /internal/relay` | proxy/nginx upgrade headers, `CDP_PUBLIC_URL`, or a firewall |
| Runtime `stale` but the notebook is open | heartbeats are being dropped | look at *Logs* for relay lines; raise `heartbeat_interval_s` if a proxy idles sockets out |
| Deployment stuck `loading` | *Runtimes → deployment error* | usually an artifact download failure (Drive quota, private HF repo, bad checksum) |
| Deployment `failed` with "adapter not available" | the framework isn't installed on that VM | `!pip install scikit-learn` in the notebook, or set `adapter` explicitly at register time |
| Endpoint answers but everything is `STALE` | no VM online; the cache is doing its job | re-attach a runtime, or switch the endpoint to `fail` if stale answers are unacceptable |
| `504 AGENT_TIMEOUT` | the model is slower than `timeout_ms` | raise the endpoint timeout; the gateway marks the runtime unregistered after a timeout so it cannot keep serving bad answers |
| Requests are `202` forever | `queue` policy with no runtime | *Jobs* shows the queue; they run when a VM returns, and expire after `CDP_JOB_TTL_S` |
| Cache is empty after a restart | `CDP_CACHE_PERSIST` off, or the volume wasn't mounted | `POST /api/cache/save` snapshots on demand; mount `CDP_DATA_DIR` |
| Gateway restarted and every VM reconnected at once | expected | agents back off with jitter, so the thundering herd is spread over ~60 s |

## Scaling beyond Colab

Nothing in the control plane knows it is talking to Colab. Run the same agent on
Modal, RunPod, Fly Machines, a Lambda-shaped container or a laptop:

```bash
pip install ./packages/shared-schema ./agent
CDP_GATEWAY_URL=https://your-host CDP_RUNTIME_TOKEN=CDP_RT_… python -m colabdeploy.launch
```

Register that VM as a second runtime and both serve the same endpoints — the gateway
picks any online VM that has (or can load) the model. That is the intended graduation
path when a hobby project needs to become a product.
