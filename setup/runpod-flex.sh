#!/usr/bin/env bash
# Check FlexAttention semantics, then compare real training passes against SDPA.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-full-v0}"
QWENRLCD_REPORT="$QWENRLCD_RUN_DIR/flex_training_benchmark.json"
QWENRLCD_SELECTION_REPORT="$QWENRLCD_RUN_DIR/sdpa_training_probability_benchmark.json"
if [ ! -f "$QWENRLCD_SELECTION_REPORT" ]; then
  QWENRLCD_SELECTION_REPORT="$QWENRLCD_RUN_DIR/sdpa_training_benchmark.json"
fi
QWENRLCD_SELECTION_ARGS=()
if [ -f "$QWENRLCD_SELECTION_REPORT" ]; then
  QWENRLCD_SELECTION_ARGS=(--selection-report "$QWENRLCD_SELECTION_REPORT")
fi

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
uv sync --extra train --extra dev
uv run pytest -q tests/test_data.py tests/test_model.py tests/test_benchmark_training.py

uv run qwenrlcd-check-parallel \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --attn-implementation flex_attention \
  --compare-attn-implementation sdpa

uv run qwenrlcd-benchmark-training \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --reference-backend sdpa \
  --candidate-backend flex_attention \
  --output "$QWENRLCD_REPORT" \
  "${QWENRLCD_SELECTION_ARGS[@]}"
