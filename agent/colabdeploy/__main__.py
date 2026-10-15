"""``python -m colabdeploy`` - the agent as a service rather than a notebook cell.

Identical to the ``colabdeploy-agent`` console script; handy in a container where
the entrypoint is easier to express as a module.
"""

from __future__ import annotations

from colabdeploy.launch import main

if __name__ == "__main__":
    raise SystemExit(main())
