#!/usr/bin/env bash
# Compare bundled, sequential, and batched singleton inference on one GPU.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-10k-v0}"
if [ "$#" -gt 0 ]; then
  shift
fi
if [ ! -f "$QWENRLCD_RUN_DIR/final/decision_head.pt" ]; then
  echo "[benchmark] missing saved model in $QWENRLCD_RUN_DIR/final" >&2
  exit 2
fi

uv sync --extra train --extra dev
uv run pytest -q tests/test_benchmark.py
uv run qwenrlcd-benchmark --run-dir "$QWENRLCD_RUN_DIR" "$@"
