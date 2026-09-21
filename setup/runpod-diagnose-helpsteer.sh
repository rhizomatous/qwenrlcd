#!/usr/bin/env bash
# Compare selected-baseline HelpSteer Score behavior on train and development validation.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-50k-v0}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
uv sync --extra train --extra dev
uv run pytest -q tests/test_diagnose_score.py tests/test_data.py

echo "[helpsteer] evaluating the selected 50k training sample"
uv run qwenrlcd-diagnose-development \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --mode score \
  --score-split train \
  --score-source helpsteer2

echo "[helpsteer] evaluating development validation"
uv run qwenrlcd-diagnose-development \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --mode score \
  --score-split validation \
  --score-source helpsteer2

echo "[helpsteer] complete; compare train and validation metrics above"
