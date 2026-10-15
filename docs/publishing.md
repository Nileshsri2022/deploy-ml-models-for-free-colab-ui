# Publishing the Python packages

ColabDeploy consists of three distributions. Publish the shared schema first because the gateway and agent depend on its exact version.

| Distribution | Source directory | Current version |
| --- | --- | --- |
| `colabdeploy-shared` | `packages/shared-schema` | `0.1.0` |
| `colabdeploy-gateway` | `apps/gateway` | `0.1.0` |
| `colabdeploy-agent` | `agent` | `0.1.0` |

## First release setup

1. Confirm the package names and versions in the three `pyproject.toml` files. These names returned `404` from PyPI when checked during this change, so no existing project was found under those names at that time.
2. On PyPI, add a **pending trusted publisher** for each project. Use GitHub owner `Nileshsri2022`, repository `deploy-ml-models-for-free-colab-ui`, workflow `.github/workflows/publish-python.yml`, and GitHub environment `pypi`.
3. In GitHub, create the `pypi` environment if it does not exist. Add any desired release approvals to that environment.
4. Confirm CI passes, then create and publish a GitHub Release with tag `v0.1.0`. The workflow builds and publishes `colabdeploy-shared` first, then the gateway and agent using PyPI Trusted Publishing (OIDC; no PyPI API token is stored in GitHub).

The actual upload is intentionally release-triggered. PyPI releases are public and immutable; the repository owner should confirm the package names, metadata, and version before creating the GitHub Release.

## Later releases

Before each release, bump all three versions together and update the exact `colabdeploy-shared==...` dependency in the gateway and agent project metadata. Run the test and packaging CI, then publish a GitHub Release whose tag matches the package versions (`vX.Y.Z`).

## Local build check

```bash
python -m pip install build twine
python -m build packages/shared-schema --outdir dist/shared
python -m build apps/gateway --outdir dist/gateway
python -m build agent --outdir dist/agent
twine check dist/shared/* dist/gateway/* dist/agent/*
```

The CI workflow runs the same checks. For the demo recording, follow [`demo-video.md`](demo-video.md).
