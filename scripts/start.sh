#!/usr/bin/env bash
# Start the Flux server. Intended to run as a long-lived foreground process
# (e.g. a Cloud Agent terminal). Assumes scripts/setup.sh already ran.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export FLUX_HOST="${FLUX_HOST:-0.0.0.0}"
export FLUX_PORT="${FLUX_PORT:-8000}"

echo "==> Starting Flux on ${FLUX_HOST}:${FLUX_PORT}"
exec python -m flux
