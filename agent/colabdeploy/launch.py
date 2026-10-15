"""``launch()`` - the one cell a Colab user runs.

    from colabdeploy import launch
    launch(gateway_url="https://my-gateway.fly.dev", token="CDP_RT_...")

What it does, in order:

1. **bootstrap** - detect GPU/CPU, create the workspace (``/content/.colabdeploy``
   on Colab, so a session *resume* keeps the artifact cache warm).
2. **register** - claim the runtime record with the control plane and pull the
   desired state (which models should be loaded).
3. **hydrate** - resolve artifacts cache-first and load models, failing per-model
   so one bad artifact cannot block the rest.
4. **tunnel** - dial out over WebSocket and start heartbeating. Reconnects forever
   with backoff; a notebook restart re-arms automatically.
5. **keep-alive** - a conservative background thread that defeats the *idle*
   timeout. It cannot defeat Colab's ~12 h hard cap; we warn before it hits and the
   UI offers a one-click relaunch.

Idempotent: re-running the cell stops the previous session and re-attaches, which
is exactly what you want after "Runtime → Restart" or a network blip.

``launch()`` returns immediately (the tunnel runs in a daemon thread) so the
notebook stays interactive; pass ``block=True`` to keep the cell attached.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from colabdeploy_shared.schemas import DesiredState

from .artifacts import ArtifactCache, ArtifactResolver
from .bootstrap import Environment, bootstrap, runtime_label, write_probe_file
from .keepalive import KeepAlive, KeepAliveConfig
from .model_store import ModelStore
from .registry_client import GatewayClient, GatewayError
from .runtime import AgentRuntime
from .state import make_reconciler
from .tunnel import TunnelClient, TunnelConfig, ws_url_from_http

__all__ = ["AgentSession", "launch", "launch_in_thread", "status", "stop"]

_SESSION: AgentSession | None = None
_SESSION_LOCK = threading.RLock()


@dataclass
class AgentSession:
    """A live attachment between this VM and one control plane."""

    env: Environment
    client: GatewayClient
    runtime: AgentRuntime
    store: ModelStore
    tunnel: TunnelClient
    keepalive: KeepAlive
    runtime_id: str
    gateway_url: str
    label: str
    started_at: float = field(default_factory=time.time)
    registration: dict[str, Any] = field(default_factory=dict)
    stopped: bool = False

    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "label": self.label,
            "gateway_url": self.gateway_url,
            "connected": self.tunnel.connected,
            "tunnel": self.tunnel.snapshot(),
            "loaded_models": self.store.keys,
            "models": self.store.snapshot(),
            "failures": self.store.failures(),
            "artifact_cache": self.store.cache.stats(),
            "keepalive": self.keepalive.snapshot(),
            "environment": self.env.describe(),
            "counters": self.runtime.counters,
            "uptime_s": round(time.time() - self.started_at, 1),
        }

    def stop(self) -> None:
        global _SESSION
        if self.stopped:
            return
        self.stopped = True
        self.keepalive.stop()
        self.tunnel.stop()
        self.runtime.close()
        self.client.close()
        with _SESSION_LOCK:
            if _SESSION is self:
                _SESSION = None
        print(f"[colabdeploy] session {self.runtime_id} stopped")


# --------------------------------------------------------------------------- #
def launch(
    gateway_url: str | None = None,
    token: str | None = None,
    *,
    workdir: str | None = None,
    label: str | None = None,
    heartbeat_interval_s: float = 15.0,
    max_loaded_models: int = 4,
    keepalive: bool = True,
    keepalive_interval_s: float = 60.0,
    browser_activity: bool = False,
    reconnect_max_s: float = 60.0,
    block: bool = False,
    quiet: bool = False,
    warm: bool = True,
    session_max_hours: float = 12.0,
) -> AgentSession:
    """Attach this VM to a ColabDeploy control plane.

    Args:
        gateway_url: base URL of the control plane (``https://…`` or ``http://…``).
            Falls back to ``$CDP_GATEWAY_URL``.
        token: runtime token minted in the UI (``CDP_RT_…``). Falls back to
            ``$CDP_RUNTIME_TOKEN``.
        workdir: override the workspace directory (default ``/content/.colabdeploy``).
        max_loaded_models: LRU cap on simultaneously warm models (VRAM guard).
        keepalive: defeat the idle timeout with a background thread.
        browser_activity: opt-in browser-side activity; see ``keepalive`` docs for
            the ToS caveat. Off by default.
        block: keep the notebook cell attached instead of returning immediately.
        warm: run each model's warm-up sample after loading.
    """
    gateway_url = (gateway_url or os.getenv("CDP_GATEWAY_URL") or "").strip().rstrip("/")
    token = (token or os.getenv("CDP_RUNTIME_TOKEN") or "").strip()
    if not gateway_url:
        raise ValueError(
            "gateway_url is required (or set $CDP_GATEWAY_URL). "
            "Find it in the UI under Connect."
        )
    if not token:
        raise ValueError(
            "token is required (or set $CDP_RUNTIME_TOKEN). "
            "Mint one in the UI: Connect → New runtime token."
        )

    with _SESSION_LOCK:
        previous = _SESSION
    if previous is not None and not previous.stopped:
        _log("re-launching: stopping the previous session", quiet)
        previous.stop()

    # 1) environment --------------------------------------------------------
    env = bootstrap(workdir, quiet=quiet)
    write_probe_file(env)
    session_started = time.time()

    # 2) register -----------------------------------------------------------
    client = GatewayClient(gateway_url, token)
    capabilities = env.capabilities("0.1.0", _available_adapters()).model_dump(mode="json")
    try:
        registration = client.register(capabilities, env.colab_url)
    except GatewayError as exc:
        client.close()
        raise RuntimeError(
            f"could not register with {gateway_url}: {exc.message}"
            + ("  (hint: mint a fresh runtime token in the UI)" if exc.code == "BAD_RUNTIME_TOKEN" else "")
        ) from exc

    runtime_id = registration.runtime_id
    _log(
        f"registered as {runtime_id} · gateway v{registration.server_version} · "
        f"{len(registration.desired_state.deployments)} deployment(s) wanted",
        quiet,
    )

    # 3) artifact cache + model pool ---------------------------------------
    cache = ArtifactCache(env.cache_dir)
    resolver = ArtifactResolver(
        cache,
        gateway_url=gateway_url,
        runtime_token=token,
        log=(lambda message, **kw: _log(message, quiet)) if not quiet else None,
    )
    store = ModelStore(
        resolver,
        cache,
        max_loaded=max_loaded_models,
        log=_make_logger(quiet),
    )

    # Hydrate whatever the control plane already wants, before the tunnel opens,
    # so the first heartbeat reports a serving runtime.
    if registration.desired_state.deployments:
        _reconcile_initial(store, registration.desired_state, warm=warm, quiet=quiet)

    # 4) local server + tunnel ---------------------------------------------
    runtime = AgentRuntime(
        env,
        store,
        runtime_id=runtime_id,
        agent_version="0.1.0",
        session_started_at=session_started,
    )
    relay_url = registration.relay_ws_url or ws_url_from_http(gateway_url)
    tunnel = TunnelClient(
        TunnelConfig(
            relay_url=relay_url,
            runtime_id=runtime_id,
            token=token,
            agent_version="0.1.0",
            heartbeat_interval_s=registration.heartbeat_interval_s or heartbeat_interval_s,
            reconnect_max_s=reconnect_max_s,
            colab_url=env.colab_url,
            session_started_at=session_started,
        ),
        env,
        handle_exec=runtime.handle_exec,
        status_fn=runtime.status,
        on_desired_state=make_reconciler(store, runtime_id, log=_make_logger(quiet)),
        log=_make_logger(quiet),
    )
    tunnel.start()

    # 5) keep-alive ---------------------------------------------------------
    keeper = KeepAlive(
        env.workdir,
        KeepAliveConfig(
            enabled=keepalive,
            interval_s=keepalive_interval_s,
            browser_activity=browser_activity,
            session_max_hours=session_max_hours,
        ),
        is_colab=env.is_colab,
        log=lambda message, level="info": _log(message, quiet, level),
        on_session_warning=lambda seconds: _warn_session(seconds, quiet),
    )
    keeper.start()

    session = AgentSession(
        env=env,
        client=client,
        runtime=runtime,
        store=store,
        tunnel=tunnel,
        keepalive=keeper,
        runtime_id=runtime_id,
        gateway_url=gateway_url,
        label=label or runtime_label(env),
        started_at=session_started,
        registration=registration.raw,
    )
    _set_session(session)

    connected = tunnel.wait_connected(timeout_s=30.0)
    if connected:
        _log(f"online · tunnel to {relay_url} established", quiet)
    else:
        _log(
            "tunnel not up yet - retrying in the background. "
            "Check that the gateway is reachable from this VM.",
            quiet,
            "warning",
        )
    _print_summary(session, quiet)

    if block:
        try:
            while not session.stopped:
                time.sleep(5.0)
        except KeyboardInterrupt:
            session.stop()
    return session


def _set_session(session: AgentSession | None) -> None:
    global _SESSION
    with _SESSION_LOCK:
        _SESSION = session


def launch_in_thread(**kwargs: Any) -> AgentSession:
    """Alias for ``launch(block=False)`` - explicit about the notebook behaviour."""
    kwargs.pop("block", None)
    return launch(block=False, **kwargs)


def status() -> dict[str, Any]:
    """Inspect the live session from another cell (``colabdeploy.status()``)."""
    with _SESSION_LOCK:
        session = _SESSION
    if session is None:
        return {"running": False, "message": "launch() has not been called in this kernel"}
    return {"running": not session.stopped, **session.status()}


def stop() -> None:
    """Detach cleanly: unload models, close the tunnel, stop keep-alive."""
    with _SESSION_LOCK:
        session = _SESSION
    if session is None:
        print("[colabdeploy] no active session")
        return
    session.stop()


# --------------------------------------------------------------------------- #
def _reconcile_initial(store: ModelStore, desired: DesiredState, *, warm: bool, quiet: bool) -> None:
    started = time.perf_counter()
    results = store.reconcile(desired)
    ready = [r for r in results if r.state == "ready"]
    failed = [r for r in results if r.state == "failed"]
    elapsed = (time.perf_counter() - started) * 1000
    _log(
        f"hydrated {len(ready)}/{len(results)} model(s) in {elapsed:.0f} ms"
        + (f"; failures: {failed[0].error[:120]}" if failed else ""),
        quiet,
        "warning" if failed else "info",
    )


def _available_adapters() -> list[str]:
    try:
        from .adapters import available_adapters

        return available_adapters()
    except Exception:
        return []


def _make_logger(quiet: bool) -> Any:
    if quiet:
        return lambda message, level="info", **data: None

    def log(message: str, level: str = "info", **data: Any) -> None:
        prefix = {"error": "!! ", "warning": "! ", "debug": "   "}.get(level, "")
        stamp = time.strftime("%H:%M:%S")
        suffix = ""
        if data:
            suffix = " " + " ".join(f"{k}={v}" for k, v in list(data.items())[:3])
        print(f"[colabdeploy {stamp}] {prefix}{message}{suffix}")

    return log


def _log(message: str, quiet: bool, level: str = "info") -> None:
    if not quiet:
        _make_logger(False)(message, level)


def _warn_session(seconds: float, quiet: bool) -> None:
    minutes = int(seconds // 60)
    _log(
        f"Colab session cap reached in ~{minutes} min. The control plane keeps your "
        "endpoints, schemas and cache; relaunch the notebook to re-arm the runtime.",
        quiet,
        "warning",
    )


# --------------------------------------------------------------------------- #
# Command-line entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    """``colabdeploy-agent`` - run the agent outside a notebook.

    Same code path as ``launch()``; this just makes it usable as a service on a
    Modal/RunPod/Fly machine, a container, or a laptop. The control plane does not
    know or care which one it is talking to, which is the point: the compute plane
    is swappable and nothing else in the system changes.

        colabdeploy-agent --gateway-url https://gw.example.com --token CDP_RT_...
        CDP_GATEWAY_URL=... CDP_RUNTIME_TOKEN=... colabdeploy-agent   # env only
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="colabdeploy-agent",
        description="Attach this machine to a ColabDeploy control plane as an inference runtime.",
    )
    parser.add_argument("--gateway-url", default=os.getenv("CDP_GATEWAY_URL", ""),
                        help="control-plane base URL (or $CDP_GATEWAY_URL)")
    parser.add_argument("--token", default=os.getenv("CDP_RUNTIME_TOKEN", ""),
                        help="runtime token from the UI (or $CDP_RUNTIME_TOKEN)")
    parser.add_argument("--label", default=None, help="how this runtime appears in the UI")
    parser.add_argument("--workdir", default=None, help="workspace dir (default /content/.colabdeploy)")
    parser.add_argument("--max-loaded-models", type=int, default=4,
                        help="LRU cap on simultaneously warm models (VRAM guard)")
    parser.add_argument("--heartbeat", type=float, default=15.0, help="heartbeat interval seconds")
    parser.add_argument("--no-keepalive", action="store_true",
                        help="do not run the idle-timeout keep-alive thread")
    parser.add_argument("--browser-activity", action="store_true",
                        help="opt-in browser-side keep-alive (Colab only; see the ToS note in the README)")
    parser.add_argument("--quiet", action="store_true", help="only log warnings and errors")
    parser.add_argument("--once", action="store_true",
                        help="attach, reconcile, print status and exit (smoke test)")
    args = parser.parse_args(argv)

    if not args.gateway_url or not args.token:
        parser.error("gateway URL and token are required (flags or $CDP_GATEWAY_URL/$CDP_RUNTIME_TOKEN)")

    session = launch(
        gateway_url=args.gateway_url,
        token=args.token,
        workdir=args.workdir,
        label=args.label,
        heartbeat_interval_s=args.heartbeat,
        max_loaded_models=args.max_loaded_models,
        keepalive=not args.no_keepalive,
        browser_activity=args.browser_activity,
        quiet=args.quiet,
        block=False,
    )

    if args.once:
        connected = session.tunnel.wait_connected(30.0)
        print(json.dumps({"connected": connected, **session.status()}, indent=2, default=str))
        session.stop()
        return 0 if connected else 1

    print(
        "attached - ctrl-c to detach. "
        f"models: {', '.join(session.store.keys) or '(none yet)'}",
        flush=True,
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\ndetaching...")
        session.stop()
    return 0


def _print_summary(session: AgentSession, quiet: bool) -> None:
    if quiet:
        return
    loaded = session.store.keys
    line = "-" * 68
    print(line)
    print(f"  runtime    : {session.runtime_id} ({session.label})")
    print(f"  gateway    : {session.gateway_url}")
    print(f"  hardware   : {session.env.gpu_name or 'CPU'}"
          f"{' · ' + str(session.env.gpu_memory_mb) + ' MB VRAM' if session.env.gpu else ''}"
          f" · {session.env.cpu_count} vCPU · {session.env.ram_mb} MB RAM")
    print(f"  models     : {', '.join(loaded) if loaded else '(none yet - deploy from the UI)'}")
    print(f"  tunnel     : {'connected' if session.tunnel.connected else 'connecting…'}")
    print(f"  workspace  : {session.env.workdir}")
    print("  next       : open the UI → Endpoints, then call")
    print(f"               POST {session.gateway_url}/v1/models/<route>/predict")
    print(line)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
