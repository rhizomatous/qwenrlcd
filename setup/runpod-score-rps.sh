#!/usr/bin/env bash
# Controlled 50k ablation: add ordinal RPS only to Score questions.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_BASELINE_DIR="outputs/qwen3-1.7b-core-option-50k-v0"
QWENRLCD_CANDIDATE_DIR="outputs/qwen3-1.7b-core-option-50k-rps-v0"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_BASELINE_DIR/score_validation_diagnostic.json"
bash "$QWENRLCD_SETUP_DIR/runpod-pilot.sh" \
  configs/qwen3_1_7b_core_option_50k_rps.json
uv run qwenrlcd-diagnose-development \
  --run-dir "$QWENRLCD_CANDIDATE_DIR" \
  --mode score
uv run qwenrlcd-compare-score \
  --baseline "$QWENRLCD_BASELINE_DIR/score_validation_diagnostic.json" \
  --candidate "$QWENRLCD_CANDIDATE_DIR/score_validation_diagnostic.json"
