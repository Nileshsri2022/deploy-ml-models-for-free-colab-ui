"""FastAPI application entrypoint for the ColabDeploy control plane.

Run it with either::

    python -m gateway.main                  # binds 0.0.0.0:8000 (preview-friendly)
    uvicorn gateway.main:app --host 0.0.0.0 --port 8000

On first boot it mints a bootstrap admin key and prints it once - that key is the
only way into ``/api`` and ``/v1`` until you create more.

If ``apps/ui/dist`` exists it is mounted as the SPA, so a single free-tier process
can serve both the UI and the API (the recommended zero-cost topology).
"""

from __future__ import annotations

import contextlib
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gateway import __version__
from gateway.api import internal as internal_api, management as management_api, v1 as v1_api
from gateway.api.deps import app_state
from gateway.core.config import get_settings
from gateway.state import AppState, build_state, set_state


def create_app(state: AppState | None = None) -> FastAPI:
    """Application factory - tests pass a state built against a temp dir."""
    settings = (state.settings if state else get_settings())
    settings.ensure_dirs()

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_state_obj: AppState = app.state.cdp  # type: ignore[attr-defined]
        await app_state_obj.startup()
        _print_bootstrap(app_state_obj)
        try:
            yield
        finally:
            await app_state_obj.shutdown()

    app = FastAPI(
        title="ColabDeploy Control Plane",
        version=__version__,
        description=(
            "Deploy any ML model from a free Colab session behind one stable API. "
            "See `architecture.md` for the full design."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.cdp = state or build_state(settings)
    set_state(app.state.cdp)

    # CORS: the UI is often served from a different origin (Vite dev, Pages).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-ColabDeploy-Cache",
            "X-ColabDeploy-Stale",
            "X-ColabDeploy-Request-Id",
            "X-ColabDeploy-Latency-Ms",
            "X-ColabDeploy-Job",
            "X-ColabDeploy-Runtime",
            "X-ColabDeploy-Error",
            "Retry-After",
        ],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-ColabDeploy-Version"] = __version__
        if request.url.path.startswith(("/v1/", "/api/", "/internal/")):
            response.headers["X-ColabDeploy-Duration-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    app.include_router(v1_api.router)
    app.include_router(management_api.router)
    app.include_router(internal_api.router)

    @app.get("/", tags=["meta"])
    async def root(request: Request) -> Any:
        """Site root: the UI for browsers, JSON service metadata for clients.

        Content negotiation matters here because the recommended free-tier
        topology is one process serving both. Opening the URL in a browser
        should show the dashboard, while ``curl``/SDKs keep getting a
        machine-readable description of what this gateway exposes.
        """
        index = _spa_index()
        if index is not None and _wants_html(request):
            return FileResponse(str(index))
        st = app_state(request)
        return {
            "name": settings.app_name,
            "version": __version__,
            "env": settings.env,
            "docs": "/docs",
            "public_api": "/v1",
            "management_api": "/api",
            "agent_api": "/internal",
            "ui": "/ui" if _spa_dist().exists() else None,
            "colab_relaunch_url": settings.colab_relaunch_url,
            "runtimes_connected": st.relay.connected_runtime_ids,
            "endpoints": [e.route for e in st.registry.endpoints.list()],
        }

    @app.get("/healthz", tags=["meta"])
    async def healthz(request: Request) -> Any:
        st = app_state(request)
        return {
            "status": "ok",
            "version": __version__,
            "runtimes_online": len(st.relay.connected_runtime_ids),
            "cache_entries": st.cache.stats().entries,
            "uptime_s": round(time.time() - st.telemetry.started_at, 1),
        }

    @app.exception_handler(HTTPException)
    async def http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        """Keep our error envelope on FastAPI/Starlette HTTP errors too."""
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            body = {"error": detail}
        else:
            body = {"error": {"code": _status_code_name(exc.status_code), "message": str(detail)}}
        headers = dict(getattr(exc, "headers", None) or {})
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers or None)

    @app.exception_handler(RequestValidationError)
    async def validation_exception(request: Request, exc: RequestValidationError) -> JSONResponse:
        """422s carry the field-level detail the deploy wizard needs to show."""
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "request body does not match the expected schema",
                    "detail": {"errors": jsonable_errors(exc.errors())},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        st = app_state(request)
        st.telemetry.log(f"unhandled error on {request.url.path}: {exc}", level="error")
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "INTERNAL", "message": str(exc), "path": request.url.path}},
        )

    _mount_spa(app)
    return app


# --------------------------------------------------------------------------- #
# UI mounting
# --------------------------------------------------------------------------- #
def _status_code_name(status_code: int) -> str:
    try:
        from http import HTTPStatus

        return HTTPStatus(status_code).name
    except ValueError:  # pragma: no cover - non-standard code
        return f"HTTP_{status_code}"


def jsonable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim Pydantic's error dicts down to the fields a UI can render."""
    out = []
    for error in errors[:20]:
        out.append(
            {
                "field": ".".join(str(p) for p in error.get("loc", [])[1:]) or None,
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
        )
    return out


def _spa_dist() -> Path:
    override = os.getenv("CDP_UI_DIST")
    if override:
        return Path(override)
    here = Path(__file__).resolve().parents[3]
    return here / "apps" / "ui" / "dist"


def _spa_index() -> Path | None:
    """Path to the built UI's ``index.html``, or ``None`` if it was never built."""
    index = _spa_dist() / "index.html"
    return index if index.exists() else None


def _wants_html(request: Request) -> bool:
    """True when the caller is a browser rather than a script.

    ``httpx`` and ``curl`` send ``Accept: */*`` (or nothing), browsers put
    ``text/html`` first - so the API root stays JSON for programmatic callers.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return True
    # A few webviews send only ``*/*`` plus an html-ish user agent.
    agent = request.headers.get("user-agent", "").lower()
    return "*/*" in accept and any(
        marker in agent for marker in ("mozilla", "chrome", "safari", "edge", "firefox")
    )


def _mount_spa(app: FastAPI) -> None:
    """Serve the built SPA if it exists - single-process, zero-cost hosting.

    Two mount points, because the bundle is built with a relative asset base
    (``base: './'``) so the same ``dist/`` works from any prefix:

    * ``/assets/...`` and ``/`` - the SPA at the site root, which is what you
      want when the gateway *is* the deployment (Render/Fly/Railway free tier).
    * ``/ui/`` (``html=True``, so unknown paths fall back to ``index.html``) -
      handy when something else owns the root, and it keeps deep links working.

    The UI uses hash routing, so no server-side rewrite rules are needed for
    ``/models``, ``/cache`` etc. That is deliberate: this project targets free
    hosts where you do not control the reverse proxy in front of you.
    """
    dist = _spa_dist()
    index = dist / "index.html"
    if not index.exists():
        return

    assets = dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    root = dist.resolve()

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/{path:path}", include_in_schema=False)
    async def spa(path: str = "") -> Any:
        """Serve a real file from ``dist/``, else fall back to ``index.html``.

        A ``StaticFiles(html=True)`` mount would 404 on ``/ui/models``; the UI
        hash-routes so it never asks for that, but a pasted link should still
        land on the app rather than an error page.
        """
        candidate = (dist / path).resolve() if path else None
        if candidate and candidate.is_file() and str(candidate).startswith(str(root)):
            return FileResponse(str(candidate))
        return FileResponse(str(index))

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Any:
        candidate = dist / "favicon.svg"
        if candidate.exists():
            return FileResponse(str(candidate))
        # A missing favicon must not become a 500 in the logs of a healthy app.
        return JSONResponse({"ui": "no favicon"}, status_code=404)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _print_bootstrap(state: AppState) -> None:
    from gateway.core.security import AuthService

    if os.getenv("CDP_QUIET_BOOT", "").lower() in {"1", "true", "yes"}:
        AuthService(state.registry).ensure_bootstrap_key(state.settings.bootstrap_admin_key)
        return
    token, created = AuthService(state.registry).ensure_bootstrap_key(state.settings.bootstrap_admin_key)
    line = "=" * 72
    print(line)
    print(f"  ColabDeploy control plane v{__version__}  ·  env={state.settings.env}")
    print(f"  data dir : {state.settings.data_dir.resolve()}")
    print(f"  cache    : {state.cache.name} ({state.cache.stats().entries} entries restored)")
    print(f"  api docs : http://localhost:{state.settings.port}/docs")
    if state.cache_loaded_entries:
        print(f"  cache    : restored {state.cache_loaded_entries} entries from disk")
    if created:
        print(line)
        print("  Bootstrap admin API key (shown ONCE - store it now):")
        print(f"    {token}")
        print("  Use it as:  curl -H 'Authorization: Bearer <key>' …")
    else:
        print("  API key    : already provisioned (see /api/keys)")
    print(f"  Colab      : {state.settings.colab_relaunch_url}")
    print(line, flush=True)


app = create_app()


def cli() -> None:
    """``colabdeploy-gateway`` console entrypoint."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "gateway.main:app",
        host=settings.host,
        port=settings.port,
        reload=os.getenv("CDP_RELOAD", "").lower() in {"1", "true", "yes"},
        log_level=os.getenv("CDP_LOG_LEVEL", "info"),
        proxy_headers=True,
        forwarded_allow_ips="*",
        ws_ping_interval=20.0,
        ws_ping_timeout=60.0,
    )


if __name__ == "__main__":  # pragma: no cover
    with contextlib.suppress(KeyboardInterrupt):
        cli()
