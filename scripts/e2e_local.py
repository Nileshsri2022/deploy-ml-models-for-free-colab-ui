#!/usr/bin/env python3
"""Run the whole system on one machine: gateway + agent + seeded demo model.

This is the fastest way to see ColabDeploy work end to end without opening
Colab, and it is the same topology the free-tier deployment uses - one process
serving the API *and* the UI, with a runtime attached over the outbound
websocket tunnel::

    python scripts/e2e_local.py                 # gateway + agent + demo model
    python scripts/e2e_local.py --no-seed       # gateway + agent, empty registry
    python scripts/e2e_local.py --no-agent      # gateway only (UI shows "no runtime")
    python scripts/e2e_local.py --reset         # start from an empty data dir

Then open http://127.0.0.1:8000/ and paste the printed API key into
**API keys → This browser**.

The "agent" here is the real ``colabdeploy.launch()`` running in a thread inside
this process. That is deliberate: it exercises the actual tunnel, artifact
resolver, model store and adapters - nothing is stubbed - while keeping the demo
to one command. In Colab the only difference is that ``launch()`` runs in a
notebook cell on a VM with a GPU.

Nothing is installed by this script; it puts the repo's own packages on
``sys.path`` so it works from a plain checkout with the gateway's dependencies
present (``pip install -e apps/gateway -e agent``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# A demo model that needs no ML framework: pure Python behind the
# python_function adapter. It is deterministic, so cached answers are *correct*
# answers - which is what makes the HIT/STALE paths visible in the UI.
# --------------------------------------------------------------------------- #
DEMO_FUNCTION = '''
POSITIVE = {"good", "great", "excellent", "love", "amazing", "wonderful", "best", "happy"}
NEGATIVE = {"bad", "awful", "terrible", "hate", "worst", "sad", "broken", "slow"}


def predict(payload, model=None, meta=None):
    """Toy sentiment + text statistics. Stand-in for any custom model."""
    text = ""
    if isinstance(payload, dict):
        text = str(payload.get("text") or payload.get("inputs", {}).get("text") or "")
    else:
        text = str(payload)

    words = [w.strip(".,!?;:").lower() for w in text.split() if w.strip(".,!?;:")]
    score = sum(1 for w in words if w in POSITIVE) - sum(1 for w in words if w in NEGATIVE)
    if not words:
        label = "empty"
    elif score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": score,
        "words": len(words),
        "chars": len(text),
        "model_version": (meta or {}).get("model_version") if isinstance(meta, dict) else None,
        "device": "cpu",
    }
'''

DEMO_SAMPLE = {"text": "this is a great little demo, and it is amazing"}


def _add_paths() -> None:
    for relative in ("packages/shared-schema", "apps/gateway", "agent"):
        path = str(REPO_ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(predicate: Any, timeout_s: float, message: str, interval_s: float = 0.1) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise SystemExit(f"timed out after {timeout_s}s waiting for {message}")


class Gateway:
    """uvicorn in a thread, so this script can seed and idle in the main one."""

    def __init__(self, host: str, port: int) -> None:
        import uvicorn

        from gateway.main import create_app

        self.host = host
        self.port = port
        self.app = create_app()
        self.config = uvicorn.Config(
            self.app, host=host, port=port, log_level="info", ws_ping_interval=None
        )
        self.server = uvicorn.Server(self.config)
        self.server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
        self.thread = threading.Thread(target=self.server.run, name="cdp-gateway", daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.thread.start()
        _wait_for(lambda: self.server.started, 30.0, "the gateway to bind its port")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10.0)


def _client(base_url: str, api_key: str) -> Any:
    import httpx

    return httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )


def _post(client: Any, path: str, **kwargs: Any) -> dict[str, Any]:
    response = client.post(path, **kwargs)
    if response.status_code >= 400:
        raise SystemExit(f"POST {path} -> {response.status_code}: {response.text[:600]}")
    return response.json()


def _get(client: Any, path: str, **kwargs: Any) -> dict[str, Any]:
    response = client.get(path, **kwargs)
    if response.status_code >= 400:
        raise SystemExit(f"GET {path} -> {response.status_code}: {response.text[:600]}")
    return response.json()


def seed(base_url: str, api_key: str, *, with_agent: bool) -> dict[str, Any]:
    """Mint a runtime token, attach an agent, register + deploy the demo model."""
    client = _client(base_url, api_key)
    summary: dict[str, Any] = {"runtime_token": None, "runtime_id": None, "models": [], "endpoints": []}

    if with_agent:
        token = _post(client, "/api/runtimes/token", params={"label": "local-demo", "ttl_hours": 12})
        summary["runtime_token"] = token["token"]
        summary["runtime_id"] = token["runtime_id"]
        summary["notebook_cell"] = token["notebook_cell"]

        from colabdeploy import launch

        workdir = Path(os.getenv("CDP_AGENT_WORKDIR", REPO_ROOT / ".data" / "demo-agent"))
        session = launch(
            gateway_url=base_url,
            token=token["token"],
            workdir=str(workdir),
            label="local-demo",
            heartbeat_interval_s=5.0,
            quiet=False,
        )
        summary["session"] = session
        _wait_for(
            lambda: session.tunnel.connected,
            30.0,
            "the agent tunnel to connect",
        )
        print(f"  ✓ agent attached as {session.runtime_id}")

    # A model that runs anywhere (no framework), and one that is pure echo.
    for spec in (
        {
            "name": "demo-sentiment",
            "framework": "python",
            "adapter": "python_function",
            "artifact": {"uri": "inline:{\"kind\": \"snippet\"}", "filename": "model.py"},
            "entrypoint": DEMO_FUNCTION,
            "schema_in": {
                "fields": {"text": {"type": "string", "description": "text to classify"}},
                "required": ["text"],
                "example": DEMO_SAMPLE,
                "description": "Toy sentiment model - stand-in for any custom function.",
            },
            "schema_out": {
                "fields": {
                    "label": {"type": "string", "enum": ["positive", "negative", "neutral", "empty"]},
                    "score": {"type": "integer"},
                    "words": {"type": "integer"},
                },
                "example": {"label": "positive", "score": 2, "words": 9},
            },
            "warmup_sample": {"inputs": DEMO_SAMPLE},
            "tags": ["demo", "nlp"],
        },
        {
            "name": "echo",
            "framework": "none",
            "adapter": "echo",
            "artifact": {"uri": "inline:{\"latency_ms\": 120}", "filename": "m.echo.json"},
            "schema_in": {"fields": {}, "required": [], "example": {"anything": "goes"}},
            "warmup_sample": {"inputs": {"ping": True}},
            "tags": ["demo"],
        },
    ):
        model = _post(client, "/api/models", json=spec)
        summary["models"].append(f"{model['name']} v{model['version']}")
        print(f"  ✓ registered {model['name']} v{model['version']} ({model['adapter']})")

        deployed = _post(
            client,
            "/api/deploy",
            json={
                "model_id": model["id"],
                "endpoint": {
                    "route": model["name"],
                    "cache_ttl_s": 300,
                    "stale_while_revalidate_s": 3600,
                    "degradation_mode": "serve_cached",
                    "timeout_ms": 20000,
                },
            },
        )
        summary["endpoints"].append(deployed["endpoint"]["route"])
        print(f"  ✓ deployed → {deployed['endpoint']['url']} ({deployed['message']})")

    if with_agent:
        # Warm the cache so the UI opens on a non-empty HIT/MISS history.
        _wait_for(
            lambda: _get(client, "/v1/status")["runtimes_online"] > 0,
            30.0,
            "the runtime to report online",
        )
        for payload in ({"inputs": DEMO_SAMPLE}, {"inputs": {"text": "the worst demo ever, broken and slow"}}):
            result = _post(client, "/v1/models/demo-sentiment/predict", json=payload)
            print(
                f"  ✓ predict {json.dumps(payload['inputs'])[:44]}… → "
                f"{json.dumps(result.get('output'))[:64]} [{result.get('cache')} "
                f"{round(result.get('latency_ms') or 0)}ms]"
            )
        # Same payload again: this is the cache working, not the model.
        repeat = _post(client, "/v1/models/demo-sentiment/predict", json={"inputs": DEMO_SAMPLE})
        print(f"  ✓ repeat call → {repeat['cache']} in {round(repeat['latency_ms'] or 0)}ms")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.getenv("CDP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("CDP_PORT", "8000")))
    parser.add_argument("--data-dir", default=os.getenv("CDP_DATA_DIR", str(REPO_ROOT / ".data" / "demo")))
    parser.add_argument("--admin-key", default=os.getenv("CDP_ADMIN_KEY", ""))
    parser.add_argument("--no-agent", action="store_true", help="gateway only, no runtime attached")
    parser.add_argument("--no-seed", action="store_true", help="attach an agent but register nothing")
    parser.add_argument("--reset", action="store_true", help="delete the data dir first")
    parser.add_argument("--ui", default=os.getenv("CDP_UI_DIST", str(REPO_ROOT / "apps" / "ui" / "dist")))
    parser.add_argument("--public-url", default=os.getenv("CDP_PUBLIC_URL", ""))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if args.reset and data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
        print(f"• removed {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.admin_key or f"CDP_API_demo_{secrets.token_hex(12)}"
    os.environ.update(
        {
            "CDP_DATA_DIR": str(data_dir),
            "CDP_HOST": args.host,
            "CDP_PORT": str(args.port),
            "CDP_ADMIN_KEY": api_key,
            "CDP_QUIET_BOOT": "1",
            "CDP_CACHE_PERSIST": "1",
            "CDP_CACHE_BACKEND": "memory",
            "CDP_HEARTBEAT_S": "5",
            "CDP_HEARTBEAT_STALE_S": "15",
            "CDP_HEARTBEAT_OFFLINE_S": "30",
            "CDP_REQUIRE_AUTH": "1",
            "CDP_UI_DIST": args.ui,
        }
    )
    if args.public_url:
        os.environ["CDP_PUBLIC_URL"] = args.public_url

    _add_paths()

    port = args.port or _free_port()
    gateway = Gateway(args.host, port)
    gateway.start()

    ui_built = Path(args.ui).joinpath("index.html").exists()
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    base = f"http://{shown_host}:{port}"
    print()
    print("─" * 72)
    print("  ColabDeploy · local end-to-end")
    print("─" * 72)
    print(f"  gateway      {base}")
    print(f"  ui           {base}/            {'(built)' if ui_built else '(not built - run: cd apps/ui && npm run build)'}")
    print(f"  api docs     {base}/docs")
    print(f"  data dir     {data_dir}")
    print(f"  api key      {api_key}")
    print("─" * 72)

    summary: dict[str, Any] = {}
    if not args.no_seed:
        print("\nseeding…")
        try:
            summary = seed(base, api_key, with_agent=not args.no_agent)
        except SystemExit as exc:
            print(f"  ! {exc}")
        except Exception as exc:
            import traceback

            print(f"  ! seeding failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    elif not args.no_agent:
        token = _post(_client(base, api_key), "/api/runtimes/token", params={"label": "local-demo"})
        summary["runtime_token"] = token["token"]
        summary["notebook_cell"] = token["notebook_cell"]

    if summary.get("runtime_token"):
        print("\n  Paste this into a Colab cell instead of the local agent, and the same")
        print("  deployments come up on that VM (the local agent keeps running too):")
        for line in str(summary["notebook_cell"]).splitlines():
            print(f"    {line}")

    print("\nready — ctrl-c to stop\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopping…")
        session = summary.get("session")
        if session is not None:
            # Shutdown must never mask ctrl-c with a second traceback.
            with contextlib.suppress(Exception):
                session.stop()
        gateway.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
