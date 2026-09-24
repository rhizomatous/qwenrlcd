#!/usr/bin/env bash
# Gate SDPA on real BF16 training forward/backward passes from the saved full run.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-full-v0}"
QWENRLCD_REPORT="$QWENRLCD_RUN_DIR/sdpa_training_benchmark.json"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
uv sync --extra train --extra dev
uv run pytest -q tests/test_benchmark_training.py tests/test_losses.py tests/test_model.py

uv run qwenrlcd-benchmark-training \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --output "$QWENRLCD_REPORT"
