#!/usr/bin/env bash
# Same 32-bundle Choice fit, now with isolated per-option scoring.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

QWENRLCD_CONFIG=configs/qwen3_1_7b_option_head_32.json
QWENRLCD_OUTPUT_DIR=outputs/qwen3-1.7b-option-head-32-v0

test -f artifacts/choice_capacity_32.jsonl
if [ -e "$QWENRLCD_OUTPUT_DIR" ]; then
  echo "[option-head] refusing to reuse existing output directory: $QWENRLCD_OUTPUT_DIR" >&2
  exit 2
fi

uv sync --extra train --extra dev
uv run pytest -q \
  tests/test_data.py tests/test_formatting.py tests/test_preflight.py \
  tests/test_model.py tests/test_losses.py
uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"
uv run qwenrlcd-check-parallel --config "$QWENRLCD_CONFIG"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
uv run qwenrlcd-check-parallel --run-dir "$QWENRLCD_OUTPUT_DIR"
uv run qwenrlcd-diagnose-choice \
  --run-dir "$QWENRLCD_OUTPUT_DIR" --split train --permutation-check
echo "[option-head] complete: same-fixture fit and order robustness, not holdout validation"
