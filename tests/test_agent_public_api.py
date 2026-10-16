"""The notebook-facing surface must stay importable.

Everything documented in ``agent/README.md`` and called by
``agent/colab_agent.ipynb`` has to be reachable as ``colabdeploy.<name>``. It is easy
to add a function to ``launch.py`` and forget the re-export in ``__init__.py`` -
``colabdeploy.stop()`` shipped exactly like that and failed with
``AttributeError: module 'colabdeploy' has no attribute 'stop'`` in a notebook.

Cheap test, high value: it fails at import time instead of at 2 a.m. in Colab.
"""

from __future__ import annotations

import json
import pathlib

import pytest

import colabdeploy

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Names the notebook and the README promise to users.
PUBLIC_API = ("launch", "launch_in_thread", "status", "stop", "main", "AgentSession")


@pytest.mark.parametrize("name", PUBLIC_API)
def test_public_api_is_exported(name: str) -> None:
    assert name in colabdeploy.__all__, f"{name} missing from colabdeploy.__all__"
    assert hasattr(colabdeploy, name), f"colabdeploy.{name} is not importable"


def test_status_reports_no_session_before_launch() -> None:
    result = colabdeploy.status()
    assert result["running"] is False
    assert "launch()" in result["message"]


def test_stop_without_a_session_is_a_no_op(capsys: pytest.CaptureFixture[str]) -> None:
    assert colabdeploy.stop() is None
    assert "no active session" in capsys.readouterr().out


def test_every_colabdeploy_call_in_the_notebook_resolves() -> None:
    """Whatever the shipped notebook calls must exist on the package."""
    notebook = json.loads((REPO_ROOT / "agent" / "colab_agent.ipynb").read_text())
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    for line in source.splitlines():
        stripped = line.strip()
        for prefix in ("colabdeploy.", "from colabdeploy import "):
            if not stripped.startswith(prefix):
                continue
            if prefix == "colabdeploy.":
                name = stripped.split("(", 1)[0].split(".")[1].strip()
                assert hasattr(colabdeploy, name), f"notebook calls colabdeploy.{name}(), which does not exist"
            else:
                for name in stripped[len(prefix) :].split(","):
                    name = name.strip().split(" ")[0]
                    if name:
                        assert hasattr(colabdeploy, name), f"notebook imports {name}, which does not exist"
