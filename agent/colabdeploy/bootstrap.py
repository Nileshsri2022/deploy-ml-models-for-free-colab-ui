"""Environment detection and workspace bootstrap.

Answers three questions the control plane needs before it can trust a runtime:

1. **What hardware is this?** GPU model/VRAM, vCPUs, RAM, CUDA - reported as
   ``RuntimeCapabilities`` so the UI can show "T4 · 15 GB VRAM" and the gateway
   can prefer a GPU runtime for heavy models.
2. **Are we actually on Colab?** Decides the work dir (``/content`` survives a
   *resume*, which is the whole point of the agent-side disk cache) and whether a
   keep-alive thread is worth running.
3. **Where do we put things?** ``{models,cache,logs,artifacts}`` under one root,
   created idempotently.

Every probe is defensive: a missing ``nvidia-smi`` or an import error must never
stop the agent from serving a CPU model.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colabdeploy_shared.schemas import RuntimeCapabilities

COLAB_MARKER = "/content"
DEFAULT_WORKDIR = ".colabdeploy"


@dataclass
class Environment:
    """Everything the agent knows about the machine it landed on."""

    is_colab: bool = False
    is_kaggle: bool = False
    workdir: Path = field(default_factory=lambda: Path.cwd())
    gpu: bool = False
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    gpu_count: int = 0
    cuda_version: str | None = None
    cpu_count: int | None = None
    ram_mb: int | None = None
    disk_free_mb: int | None = None
    python_version: str = field(default_factory=lambda: platform.python_version())
    session_started_at: float = field(default_factory=time.time)
    colab_url: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def models_dir(self) -> Path:
        return self.workdir / "models"

    @property
    def cache_dir(self) -> Path:
        return self.workdir / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.workdir / "logs"

    @property
    def artifacts_dir(self) -> Path:
        return self.workdir / "artifacts"

    def ensure_dirs(self) -> None:
        for path in (self.workdir, self.models_dir, self.cache_dir, self.logs_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

    def capabilities(self, agent_version: str, adapters: list[str]) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            gpu=self.gpu,
            gpu_name=self.gpu_name,
            gpu_memory_mb=self.gpu_memory_mb,
            cpu_count=self.cpu_count,
            ram_mb=self.ram_mb,
            cuda_version=self.cuda_version,
            python_version=self.python_version,
            agent_version=agent_version,
            disk_free_mb=self.disk_free_mb,
            adapters=adapters,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "is_colab": self.is_colab,
            "is_kaggle": self.is_kaggle,
            "workdir": str(self.workdir),
            "gpu": self.gpu,
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "cuda_version": self.cuda_version,
            "cpu_count": self.cpu_count,
            "ram_mb": self.ram_mb,
            "disk_free_mb": self.disk_free_mb,
            "python_version": self.python_version,
            "session_started_at": self.session_started_at,
            "colab_url": self.colab_url,
        }


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def _detect_colab() -> bool:
    if os.path.isdir(COLAB_MARKER) and "google.colab" in sys.modules:
        return True
    try:  # pragma: no cover - only true inside a real Colab kernel
        import google.colab  # noqa: F401

        return True
    except Exception:
        pass
    # Fallback: /content exists and we look like a Colab image.
    return (os.path.isdir(COLAB_MARKER) and os.path.exists("/usr/local/lib/python3.10/dist-packages")) or (
        os.path.isdir(COLAB_MARKER) and "COLAB_" in "".join(os.environ)
    )


def _detect_kaggle() -> bool:
    return os.path.isdir("/kaggle") or bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE"))


def _colab_url() -> str | None:
    try:  # pragma: no cover - Colab only
        from google.colab import output  # type: ignore

        proxy = getattr(output, "_proxy_url", None)
        return str(proxy) if proxy else None
    except Exception:
        return None


def _probe_gpu() -> tuple[bool, str | None, int | None, int, str | None]:
    """Prefer torch (accurate VRAM), fall back to nvidia-smi, else CPU."""
    try:  # pragma: no cover - requires torch
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return (
                True,
                props.name,
                int(getattr(props, "total_memory", 0) // (1024 * 1024)) or None,
                int(torch.cuda.device_count()),
                torch.version.cuda,
            )
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            line = (out.stdout or "").strip().splitlines()
            if out.returncode == 0 and line:
                parts = [p.strip() for p in line[0].split(",")]
                name = parts[0] if parts else None
                memory = int(float(parts[1])) if len(parts) > 1 and parts[1] else None
                driver = parts[2] if len(parts) > 2 else None
                return True, name, memory, len(line), driver
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
    return False, None, None, 0, None


def _probe_cpu_ram() -> tuple[int | None, int | None]:
    cpus = os.cpu_count()
    ram_mb: int | None = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    ram_mb = int(line.split()[1]) // 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    return cpus, ram_mb


def _probe_disk(path: Path) -> int | None:
    try:
        usage = shutil.disk_usage(str(path))
        return int(usage.free // (1024 * 1024))
    except OSError:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def bootstrap(workdir: str | Path | None = None, *, quiet: bool = False) -> Environment:
    """Detect the environment and create the agent workspace. Idempotent."""
    is_colab = _detect_colab()
    is_kaggle = _detect_kaggle()

    if workdir:
        root = Path(workdir).expanduser()
    elif is_colab or is_kaggle:
        # /content survives a session *resume* -> the disk cache stays warm.
        root = Path(COLAB_MARKER if is_colab else "/kaggle/working") / DEFAULT_WORKDIR
    else:
        root = Path(os.getenv("CDP_AGENT_WORKDIR", Path.cwd() / DEFAULT_WORKDIR)).expanduser()

    root.mkdir(parents=True, exist_ok=True)
    gpu, gpu_name, gpu_mem, gpu_count, cuda = _probe_gpu()
    cpus, ram_mb = _probe_cpu_ram()

    env = Environment(
        is_colab=is_colab,
        is_kaggle=is_kaggle,
        workdir=root.resolve(),
        gpu=gpu,
        gpu_name=gpu_name,
        gpu_memory_mb=gpu_mem,
        gpu_count=gpu_count,
        cuda_version=cuda,
        cpu_count=cpus,
        ram_mb=ram_mb,
        disk_free_mb=_probe_disk(root),
        session_started_at=time.time(),
        colab_url=_colab_url(),
    )
    env.ensure_dirs()

    if not quiet:
        hardware = f"{gpu_name} ({gpu_mem} MB VRAM)" if gpu else f"CPU x{cpus}"
        print(
            f"[colabdeploy] workspace {env.workdir} · {hardware} · "
            f"{'Colab' if is_colab else 'Kaggle' if is_kaggle else 'local'} · "
            f"python {env.python_version}"
        )
    return env


def runtime_label(env: Environment) -> str:
    """A readable label for the UI's runtime list."""
    kind = "colab" if env.is_colab else "kaggle" if env.is_kaggle else "local"
    device = (env.gpu_name or "cpu").replace(" ", "-").lower()
    return f"{kind}-{device}-{int(env.session_started_at) % 10000}"


def write_probe_file(env: Environment) -> Path:
    """Persist the probe result - useful for debugging a dead session afterwards."""
    path = env.workdir / "environment.json"
    path.write_text(json.dumps(env.describe(), indent=2, default=str), encoding="utf-8")
    return path
