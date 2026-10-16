"""Runtime configuration, all overridable by environment variables.

Every default is chosen so that ``uvicorn gateway.main:app`` works on a free-tier
box with zero setup: a SQLite-free JSON store under ``.data/``, an in-process
cache, and a bootstrap admin key printed once at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- identity ---------------------------------------------------------
    app_name: str = "ColabDeploy"
    version: str = "0.1.0"
    env: str = field(default_factory=lambda: os.getenv("CDP_ENV", "development"))

    # --- server -----------------------------------------------------------
    host: str = field(default_factory=lambda: os.getenv("CDP_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("CDP_PORT", 8000))
    public_base_url: str = field(
        default_factory=lambda: os.getenv("CDP_PUBLIC_URL", "").rstrip("/")
    )
    cors_origins: list[str] = field(
        default_factory=lambda: [
            o.strip() for o in os.getenv("CDP_CORS_ORIGINS", "*").split(",") if o.strip()
        ]
    )

    # --- persistence ------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("CDP_DATA_DIR", ".data")))

    # --- registry mirror (optional; survives free-tier container recycling) ---
    # "file"  -> JSON under data_dir/store (original behaviour);
    # "redis" -> same files, mirrored per-record to Redis and restored on boot.
    #           Point CDP_REDIS_URL at a free Upstash database (rediss://...);
    #           Render's own free Key Value is in-memory only and will NOT do.
    registry_backend: str = field(default_factory=lambda: os.getenv("CDP_REGISTRY_BACKEND", "file"))
    redis_url: str = field(default_factory=lambda: os.getenv("CDP_REDIS_URL", ""))
    # Coalesce mirror writes: mutations of the same key within this window cost
    # one SET. 0 = write through on every mutation. The local disk copy is
    # always written immediately either way; this only paces Redis traffic.
    registry_sync_s: float = field(default_factory=lambda: _float("CDP_REGISTRY_SYNC_S", 30.0))
    # Periodic no-op write so free tiers that archive idle databases (Upstash,
    # after ~30 days) keep the mirror alive while the gateway runs.
    registry_keepalive_s: float = field(
        default_factory=lambda: _float("CDP_REGISTRY_KEEPALIVE_S", 86400.0)
    )

    # --- cache ------------------------------------------------------------
    cache_backend: str = field(default_factory=lambda: os.getenv("CDP_CACHE_BACKEND", "memory"))
    cache_max_entries: int = field(default_factory=lambda: _int("CDP_CACHE_MAX_ENTRIES", 20000))
    cache_max_bytes: int = field(
        default_factory=lambda: _int("CDP_CACHE_MAX_BYTES", 256 * 1024 * 1024)
    )
    cache_persist: bool = field(default_factory=lambda: _bool("CDP_CACHE_PERSIST", True))

    # --- runtime supervision ---------------------------------------------
    heartbeat_interval_s: float = field(default_factory=lambda: _float("CDP_HEARTBEAT_S", 15.0))
    heartbeat_stale_s: float = field(default_factory=lambda: _float("CDP_HEARTBEAT_STALE_S", 45.0))
    heartbeat_offline_s: float = field(
        default_factory=lambda: _float("CDP_HEARTBEAT_OFFLINE_S", 120.0)
    )
    session_max_hours: float = field(default_factory=lambda: _float("CDP_SESSION_MAX_HOURS", 12.0))
    session_warn_minutes: float = field(
        default_factory=lambda: _float("CDP_SESSION_WARN_MINUTES", 20.0)
    )
    runtime_token_ttl_hours: float = field(
        default_factory=lambda: _float("CDP_RUNTIME_TOKEN_TTL_HOURS", 24.0)
    )

    # --- inference --------------------------------------------------------
    default_timeout_ms: int = field(default_factory=lambda: _int("CDP_TIMEOUT_MS", 30000))
    max_body_bytes: int = field(
        default_factory=lambda: _int("CDP_MAX_BODY_BYTES", 10 * 1024 * 1024)
    )
    max_batch_size: int = field(default_factory=lambda: _int("CDP_MAX_BATCH", 64))
    retry_after_s: int = field(default_factory=lambda: _int("CDP_RETRY_AFTER_S", 30))
    daily_inference_quota: int = field(
        default_factory=lambda: _int("CDP_DAILY_INFERENCE_QUOTA", 1000)
    )

    # --- jobs -------------------------------------------------------------
    job_ttl_s: float = field(default_factory=lambda: _float("CDP_JOB_TTL_S", 3600.0))
    job_max_attempts: int = field(default_factory=lambda: _int("CDP_JOB_MAX_ATTEMPTS", 3))

    # --- logs / metrics / Sentry -----------------------------------------
    log_buffer_lines: int = field(default_factory=lambda: _int("CDP_LOG_BUFFER", 5000))
    metrics_buffer_points: int = field(default_factory=lambda: _int("CDP_METRICS_BUFFER", 20000))
    sentry_dsn: str = field(default_factory=lambda: os.getenv("CDP_SENTRY_DSN", "").strip())
    sentry_ui_dsn: str = field(default_factory=lambda: os.getenv("CDP_SENTRY_UI_DSN", "").strip())
    sentry_environment: str = field(
        default_factory=lambda: os.getenv("CDP_SENTRY_ENVIRONMENT", "").strip()
    )
    sentry_traces_sample_rate: float = field(
        default_factory=lambda: _float("CDP_SENTRY_TRACES_SAMPLE_RATE", 0.0)
    )
    sentry_ui_traces_sample_rate: float = field(
        default_factory=lambda: _float("CDP_SENTRY_UI_TRACES_SAMPLE_RATE", 0.0)
    )
    sentry_read_token: str = field(
        default_factory=lambda: os.getenv("CDP_SENTRY_READ_TOKEN", "").strip()
    )
    sentry_org_slug: str = field(
        default_factory=lambda: os.getenv("CDP_SENTRY_ORG_SLUG", "").strip()
    )
    sentry_project_slug: str = field(
        default_factory=lambda: os.getenv("CDP_SENTRY_PROJECT_SLUG", "").strip()
    )
    sentry_api_base_url: str = field(
        default_factory=lambda: (
            os.getenv("CDP_SENTRY_API_BASE_URL", "https://sentry.io").strip().rstrip("/")
        )
    )

    # --- security ---------------------------------------------------------
    bootstrap_admin_key: str = field(default_factory=lambda: os.getenv("CDP_ADMIN_KEY", ""))
    require_auth: bool = field(default_factory=lambda: _bool("CDP_REQUIRE_AUTH", True))
    allow_anonymous_inference: bool = field(
        default_factory=lambda: _bool("CDP_ALLOW_ANONYMOUS_INFERENCE", False)
    )

    # --- colab relaunch ---------------------------------------------------
    colab_repo: str = field(
        default_factory=lambda: os.getenv(
            "CDP_COLAB_REPO", "Nileshsri2022/deploy-ml-models-for-free-colab-ui"
        )
    )
    colab_notebook_path: str = field(
        default_factory=lambda: os.getenv("CDP_COLAB_NOTEBOOK", "agent/colab_agent.ipynb")
    )
    colab_branch: str = field(default_factory=lambda: os.getenv("CDP_COLAB_BRANCH", "main"))

    # --- artifact store ---------------------------------------------------
    artifact_dir: Path | None = None  # derived: data_dir / "artifacts"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.artifact_dir = self.data_dir / "artifacts"

    # -- derived helpers ---------------------------------------------------
    @property
    def colab_relaunch_url(self) -> str:
        return (
            "https://colab.research.google.com/github/"
            f"{self.colab_repo}/blob/{self.colab_branch}/{self.colab_notebook_path}"
        )

    def public_url(self, path: str, request_base: str | None = None) -> str:
        base = self.public_base_url or (request_base or "").rstrip("/")
        if not path.startswith("/"):
            path = "/" + path
        return f"{base}{path}" if base else path

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        assert self.artifact_dir is not None
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "store").mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings(**overrides: object) -> Settings:
    """Rebuild settings (used by tests to point at a temp data dir)."""
    global _settings
    _settings = Settings(**overrides)  # type: ignore[arg-type]
    _settings.ensure_dirs()
    return _settings
