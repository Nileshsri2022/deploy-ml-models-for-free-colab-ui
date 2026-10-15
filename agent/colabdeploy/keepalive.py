"""Keep-alive layers (architecture.md §7).

Colab kills a session for two independent reasons, and only one of them is
solvable from inside the VM:

* **Idle timeout (~90 min without interaction).** A background thread that does a
  little real work - touching a file, logging, checking hardware - resets this.
  The open WebSocket also counts as activity on most Colab builds.
* **Max session lifetime (~12 h).** Nothing inside the VM can prevent this. We
  detect the approaching cap, warn through the gateway so the UI can show a
  countdown and a one-click relaunch button, and make re-arming as cheap as
  possible (artifact cache + automatic desired-state reconcile).

Everything here is **opt-in and conservative** by default: the worker thread does
a few milliseconds of I/O per minute, and the browser-side "click the connect
button" trick is off unless explicitly enabled, because it is the part most likely
to conflict with Google's usage policy. The README says so plainly.
"""

from __future__ import annotations

import contextlib
import os
import platform
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LogFn = Callable[[str, str], None]


@dataclass
class KeepAliveConfig:
    enabled: bool = True
    interval_s: float = 60.0
    touch_file: bool = True
    log_heartbeat: bool = False
    #: Opt-in browser-side activity (Colab only). Off by default - see module docstring.
    browser_activity: bool = False
    browser_interval_s: float = 90.0
    session_warn_minutes: float = 20.0
    session_max_hours: float = 12.0


@dataclass
class KeepAliveState:
    started_at: float = field(default_factory=time.time)
    ticks: int = 0
    last_tick_at: float | None = None
    warnings_sent: int = 0
    browser_injected: bool = False
    errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        uptime = time.time() - self.started_at
        return {
            "ticks": self.ticks,
            "uptime_s": round(uptime, 1),
            "last_tick_at": self.last_tick_at,
            "warnings_sent": self.warnings_sent,
            "browser_injected": self.browser_injected,
            "errors": self.errors,
            "session_hours_remaining": round(max(0.0, 12.0 - uptime / 3600.0), 2),
        }


class KeepAlive:
    """Background thread that keeps an interactive session from idling out."""

    def __init__(
        self,
        workdir: Any,
        config: KeepAliveConfig | None = None,
        *,
        is_colab: bool = False,
        log: LogFn | None = None,
        on_session_warning: Callable[[float], None] | None = None,
    ) -> None:
        self.workdir = str(workdir)
        self.config = config or KeepAliveConfig()
        self.is_colab = is_colab
        self.log = log or (lambda message, level="info": None)
        self.on_session_warning = on_session_warning
        self.state = KeepAliveState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._warned = False

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if not self.config.enabled:
            self.log("keep-alive disabled", "info")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cdp-keepalive", daemon=True)
        self._thread.start()
        self.log(
            f"keep-alive on (every {self.config.interval_s:.0f}s"
            + (", browser activity enabled" if self.config.browser_activity else "")
            + ")",
            "info",
        )
        if self.config.browser_activity and self.is_colab:
            self.inject_browser_activity()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        interval = max(10.0, self.config.interval_s)
        while not self._stop.wait(interval):
            try:
                self._tick()
            except Exception as exc:
                self.state.errors += 1
                self.log(f"keep-alive tick failed: {exc}", "warning")

    def _tick(self) -> None:
        self.state.ticks += 1
        self.state.last_tick_at = time.time()
        if self.config.touch_file:
            path = os.path.join(self.workdir, "keepalive")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(f"{time.time()}\n{platform.node()}\n{self.state.ticks}\n")
        if self.config.log_heartbeat:
            self.log(f"keep-alive tick #{self.state.ticks}", "debug")
        self._check_session_cap()

    def _check_session_cap(self) -> None:
        """Warn before Colab's hard 12 h cap so the UI can prompt a relaunch."""
        uptime_h = (time.time() - self.state.started_at) / 3600.0
        remaining_min = (self.config.session_max_hours - uptime_h) * 60.0
        if 0 < remaining_min <= self.config.session_warn_minutes:
            self.state.warnings_sent += 1
            self.log(
                f"Colab session ends in ~{remaining_min:.0f} min - relaunch the notebook to stay online",
                "warning",
            )
            if self.on_session_warning is not None and not self._warned:
                self._warned = True
                # A broken warning callback must not kill the keep-alive thread.
                with contextlib.suppress(Exception):
                    self.on_session_warning(remaining_min * 60.0)

    # ------------------------------------------------------------------ #
    def inject_browser_activity(self) -> bool:
        """Opt-in: poke the Colab UI so the browser side also looks interactive.

        Only works inside a real Colab kernel. Disabled by default - see the module
        docstring for the ToS caveat.
        """
        if not self.is_colab:
            return False
        try:  # pragma: no cover - Colab only
            from IPython import get_ipython  # type: ignore
            from IPython.display import HTML, display  # type: ignore

            shell = get_ipython()
            if shell is None:
                return False
            interval_ms = int(max(15.0, self.config.browser_interval_s) * 1000)
            html = (
                "<div id='cdp-keepalive' style='display:none'></div>"
                "<script>(function(){const el=document.querySelector('#top-toolbar "
                ".colab-connect-button');if(!el)return;setInterval(function(){"
                "el.click();el.focus();},"
                f"{interval_ms}"
                ");})();</script>"
            )
            display(HTML(html))
            self.state.browser_injected = True
            self.log("browser keep-alive injected (opt-in)", "info")
            return True
        except Exception as exc:
            self.log(f"browser keep-alive unavailable: {exc}", "debug")
            return False

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        return {"config": self.config.__dict__, **self.state.as_dict()}
