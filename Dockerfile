# syntax=docker/dockerfile:1
#
# CPU-only image for cxr_model: a reproducible environment from uv + the
# committed lockfile, no local setup. The MC code falls back to CPU
# automatically (cupy imports but finds no GPU), so the sweeps, the `cxr` CLI,
# and the test suite all run here.
#
#   docker build -t cxr-model .
#   docker run --rm cxr-model pytest -q                                   # CPU safety net
#   docker run --rm -v "$PWD/checkpoints:/app/checkpoints" cxr-model cxr scan silicon --quick
#
# ---------------------------------------------------------------------------
# GPU image (future): everything below the base layer is shared with a GPU
# build. To make one, swap the base for an NVIDIA CUDA image matching the
# cupy-cuda13x wheel and add Python + uv onto it, e.g.
#
#   FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
#   RUN apt-get update && apt-get install -y --no-install-recommends \
#         python3 python3-venv && rm -rf /var/lib/apt/lists/*
#   COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
#
# then keep the identical "app layer" below and run with `--gpus all` (the host
# needs the NVIDIA Container Toolkit). See docs/running-on-a-cluster.md.
# ---------------------------------------------------------------------------

# ---- base layer (the only part a GPU build changes) ----
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

# ---- app layer (identical for CPU and GPU) ----
# Copy (don't hardlink) into the venv, precompile to .pyc, and never fetch a
# second interpreter (the base already ships Python 3.14).
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) Install dependencies from the lockfile first, so this heavy layer is cached
#    and only re-runs when pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 2) Add the source and install the package itself (fast; re-runs on any edit).
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# `uv run` is the entrypoint, so the container behaves like the local dev env:
#   docker run cxr-model                       -> cxr --help            (CMD below)
#   docker run cxr-model pytest -q             -> the CPU test suite
#   docker run cxr-model cxr scan silicon --quick
ENTRYPOINT ["uv", "run", "--frozen"]
CMD ["cxr", "--help"]
