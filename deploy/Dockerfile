# Serves the champion model as a REST API (see serve.py) via ONNX Runtime --
# no torch/lightning executed at inference time, and ~6-8x faster end to end
# than the PyTorch path once warmed up (docs/experiments.md). One process, no
# nginx, no auto-generated virtualenv -- see serve.py's module docstring for
# why this replaced `mlflow models build-docker` for Cloud Run specifically.
FROM python:3.13-slim

WORKDIR /app

# ffmpeg is required by PersonalityModelONNX.predict_from_audio's decode step
# (/predict_audio) -- a subprocess dependency, not a Python one.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv, not plain pip: exordium's transitive unigaze dependency hard-pins
# timm==0.3.2, which imports torch._six -- removed from current torch, an
# ImportError predict_from_media's face-detection import chain hits at
# request time, not at build time. deploy/overrides.txt (below) forces
# timm>=1.0 instead, mirroring pyproject.toml's [tool.uv]
# .override-dependencies (a uv-sync-only mechanism plain `pip install`
# cannot use at all).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install the package itself (core deps still include torch -- it is a hard
# dependency of personalitylinmult's own import chain, not something this
# image avoids) plus the serve extra (fastapi, uvicorn) and the onnx extra
# (onnx, onnxruntime, onnxscript, exordium, timm, transformers -- see
# pyproject.toml's onnx extra comments for why predict_from_media needs
# these at inference time, not just export time). Copying only what the
# build needs keeps the image cache-friendly across code changes that don't
# touch dependencies.
COPY pyproject.toml README.md deploy/overrides.txt ./
COPY personalitylinmult ./personalitylinmult
# hatch-vcs derives the package version from git tags; this build context has
# no .git directory (only the files copied above). A throwaway repo with one
# tag gives it real (if meaningless) VCS metadata to read -- the served app
# only ever calls PersonalityModelONNX.from_pretrained, which does not
# depend on __version__ being meaningful.
RUN git init -q && git config user.email a@a.a && git config user.name a \
    && git add -A && git commit -q -m init && git tag v0.0.0
RUN uv pip install --system --no-cache --override overrides.txt ".[serve,onnx]"

COPY deploy/serve.py ./deploy/serve.py

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn deploy.serve:app --host 0.0.0.0 --port ${PORT}"]
