#!/usr/bin/env bash
# Idempotent environment bootstrap for Flux.
#
# Safe to run repeatedly: it creates the virtualenv if missing, installs pinned
# dependencies (CPU torch wheels), and pre-downloads the default model into the
# Hugging Face cache so the server starts without a cold download.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
FLUX_MODEL="${FLUX_MODEL:-distilgpt2}"

echo "==> Flux setup starting (model=$FLUX_MODEL)"

# The default base image ships python3 but not the venv module. Install it once
# (idempotent: the check short-circuits on subsequent runs).
if ! "$PYTHON_BIN" -c "import ensurepip" >/dev/null 2>&1; then
  echo "==> Installing python3-venv (requires sudo)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip >/dev/null

echo "==> Installing CPU torch wheel"
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

echo "==> Installing project dependencies"
pip install -r requirements-dev.txt

echo "==> Pre-downloading model into Hugging Face cache"
FLUX_MODEL="$FLUX_MODEL" python scripts/download_model.py

echo "==> Flux setup complete"
