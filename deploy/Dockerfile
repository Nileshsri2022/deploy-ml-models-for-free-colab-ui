# ColabDeploy control plane — one image, one process, API + UI together.
#
# Build from the repo root (the context matters, this file lives in deploy/):
#   docker build -f deploy/Dockerfile -t colabdeploy .
#   docker run --rm -p 8000:8000 -v cdp-data:/data \
#     -e CDP_ADMIN_KEY="CDP_API_$(openssl rand -hex 16)" colabdeploy
#
# Stage 1 builds the SPA so the runtime image needs no Node at all.

# --------------------------------------------------------------------------- #
FROM node:22-alpine AS ui

WORKDIR /ui
# Copy the manifests first: dependency install is the slowest layer and should be
# cached until package.json actually changes.
COPY apps/ui/package.json apps/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY apps/ui/ ./
RUN npm run build


# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

# curl for the healthcheck; build-essential is not needed (no native deps).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Shared schema first: both other packages depend on it.
COPY packages/shared-schema/ packages/shared-schema/
COPY apps/gateway/pyproject.toml apps/gateway/README.md apps/gateway/
RUN pip install --no-cache-dir ./packages/shared-schema

COPY apps/gateway/gateway/ apps/gateway/gateway/
# [redis] pulls in redis-py for the registry mirror (CDP_REGISTRY_BACKEND=redis).
RUN pip install --no-cache-dir './apps/gateway[redis]'

COPY --from=ui /ui/dist /app/apps/ui/dist

# The agent is NOT installed here: it belongs on the VM, not in the control plane.
# Nothing in this image can execute model code, which is the safety boundary.

ENV CDP_HOST=0.0.0.0 \
    CDP_PORT=8000 \
    CDP_DATA_DIR=/data \
    CDP_UI_DIST=/app/apps/ui/dist \
    CDP_CACHE_BACKEND=memory \
    CDP_CACHE_PERSIST=1 \
    CDP_REQUIRE_AUTH=1 \
    PYTHONUNBUFFERED=1

# State that must outlive a container: the registry, uploaded artifacts and the
# cache snapshot. Mount a volume here or the free tier forgets everything on redeploy.
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz || exit 1

# One worker only: the relay holds websockets in-process, so N workers would mean
# N gateways that cannot see each other's runtimes. Scale the VMs, not this.
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--ws-ping-interval", "20", "--ws-ping-timeout", "60"]
