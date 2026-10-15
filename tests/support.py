"""Helpers shared by the HTTP tests and the end-to-end agent tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

TEST_ADMIN_KEY = "CDP_API_test_admin_key_0000000000"
TEST_RUNTIME_LABEL = "pytest-runtime"


def wait_for(
    predicate: Callable[[], Any],
    timeout_s: float = 10.0,
    interval_s: float = 0.05,
    message: str = "",
) -> Any:
    """Poll until ``predicate()`` is truthy.

    Tunnel handshakes, heartbeats and reconciles are genuinely asynchronous, so
    every cross-process assertion goes through here instead of ``time.sleep`` -
    a fixed sleep is either flaky or slow, usually both.
    """
    deadline = time.time() + timeout_s
    last: Any = None
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        try:
            last = predicate()
        except Exception as exc:
            last = f"raised {type(exc).__name__}: {exc}"
        else:
            if last:
                return last
        time.sleep(interval_s)
    raise AssertionError(
        f"timed out after {timeout_s}s ({attempts} polls) waiting for "
        f"{message or 'condition'}; last={last!r}"
    )
