# citecheck — single-container deploy image for HuggingFace Spaces.
#
# Layout: GROBID 0.8.2-crf provides the JVM + reference-parsing service.
# On top we install Python 3.11 (the project's `requires-python` floor),
# uv, then the citecheck package with `web` + `claims` extras.  The
# entrypoint script (`spaces/start.sh`) launches GROBID on 127.0.0.1:8070
# in the background, waits for /api/isalive, then execs uvicorn on
# 0.0.0.0:7860 — the port HuggingFace Spaces routes public traffic to.
#
# Single-container keeps the deploy simple (no docker-compose on HF
# Spaces), at the cost of a ~3 GB image.  On HF free tier (2 vCPU, 16 GB
# RAM, 50 GB disk) this fits comfortably.
#
# Local build + smoke test:
#   docker build -t citecheck:dev .
#   docker run --rm -p 7860:7860 \
#     -e CITECHECK_CONTACT_EMAIL=pbeauco@gmail.com \
#     -e CEREBRAS_API_KEY=$CEREBRAS_API_KEY \
#     citecheck:dev
#
# Then visit http://127.0.0.1:7860/.

# ---------------------------------------------------------------------------
# Stage 1 — base on the official GROBID CRF image (~2 GB, Debian-slim + JDK).
# ---------------------------------------------------------------------------
FROM lfoppiano/grobid:0.8.2-crf

# We are root in the base image; HF Spaces requires a non-root user with
# UID 1000 named `user` for write-permission compatibility on /tmp and
# /data.  Set it up before we install Python so apt cache stays clean.
USER root

# Drop the GROBID base image's ENTRYPOINT; we supply our own that boots
# both GROBID and uvicorn.
ENTRYPOINT []

# ---------------------------------------------------------------------------
# Python runtime — install uv first, then let it provision Python 3.11.
# This avoids assumptions about which Debian release the GROBID base
# image is on (bullseye ships Python 3.9, bookworm ships 3.11).  uv's
# managed Python installs are statically linked and version-pinned, so
# the runtime is deterministic regardless of the base image's apt repo.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast pip-replacement).  Binary downloaded from astral.sh.
# UV_UNMANAGED_INSTALL puts the binary at /usr/local/bin/uv without
# the standard ~/.cargo dance, so it's on PATH for every user.
RUN curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_UNMANAGED_INSTALL=/usr/local/bin sh \
    && uv --version

# Provision Python 3.11 via uv (managed install).  We pin to 3.11
# explicitly (rather than ">=3.11") so the image is reproducible.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN uv python install 3.11 \
    && ln -sf "$(uv python find 3.11)" /usr/local/bin/python \
    && python --version

# ---------------------------------------------------------------------------
# citecheck install — copy the project tree, then `uv sync` with the web +
# claims extras.  We split the COPY into two steps so source-only edits
# (templates, routes) don't bust the heavy sentence-transformers /
# pytorch layer cache.
# ---------------------------------------------------------------------------
WORKDIR /app

# 1) Manifests only — lets the dependency-install layer cache survive any
# subsequent source-only edit.
COPY pyproject.toml uv.lock README.md ./

# 2) Source needed by the wheel build (hatchling reads src/citecheck).
COPY src/ ./src/

# Sync the locked dependencies plus the two extras the web demo needs.
# --frozen guarantees we install exactly uv.lock; --no-dev skips pytest /
# ruff to keep the image lean.  --torch-backend=cpu is critical: it tells
# uv to install the CPU-only torch wheel (~200 MB) instead of the default
# CUDA-enabled wheel (~3 GB including nvidia-cu* dependencies we never
# use — HF free tier has no GPU).
ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_TORCH_BACKEND=cpu
RUN uv sync --frozen --no-dev --extra web --extra claims \
    && rm -rf /tmp/uv-cache

# 3) Static, templates, scripts — these change frequently; keep them after
# the heavy install layer so iteration is fast.
COPY spaces/start.sh ./spaces/start.sh
RUN chmod +x ./spaces/start.sh

# ---------------------------------------------------------------------------
# Runtime config — we keep running as root inside the container because
# (a) the GROBID base image owns /opt/grobid as root and the service
# expects to write logs/temp files there, and (b) HF Spaces' Docker SDK
# does not require non-root (unlike its Gradio/Streamlit SDKs).  The web
# app's writable paths (/tmp/citecheck/*) are always root-writable.
#
# Secrets (CEREBRAS_API_KEY, CITECHECK_CONTACT_EMAIL) are injected at
# runtime via the HF Space Secrets UI; nothing sensitive is baked in.
# ---------------------------------------------------------------------------
RUN mkdir -p /tmp/citecheck/uploads /tmp/citecheck/data

# GROBID HTTP endpoint that the web app talks to; bound on localhost
# inside the container so it's never exposed externally.  uvicorn listens
# on 0.0.0.0:7860 because HF routes public traffic there.
ENV GROBID_HOST=http://127.0.0.1:8070 \
    CITECHECK_WEB_UPLOAD_DIR=/tmp/citecheck/uploads \
    CITECHECK_WEB_DATA_DIR=/tmp/citecheck/data \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

EXPOSE 7860

# ---------------------------------------------------------------------------
# Entrypoint — boots GROBID (background) then uvicorn (foreground).  See
# spaces/start.sh for the wait-for-grobid logic.
# ---------------------------------------------------------------------------
CMD ["./spaces/start.sh"]
