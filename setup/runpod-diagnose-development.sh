#!/usr/bin/env bash
# Read-only diagnostics for a completed core option-head run.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${1:-outputs/qwen3-1.7b-core-option-50k-v0}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
uv sync --extra train --extra dev
uv run pytest -q tests/test_diagnose_score.py tests/test_choice_set_probe.py
uv run qwenrlcd-diagnose-development --run-dir "$QWENRLCD_RUN_DIR"
