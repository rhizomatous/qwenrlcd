#!/usr/bin/env bash
# Source-stratified core pilot, with a tokenizer-only length check before GPU work.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_core_pilot.json}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
uv sync --extra train --extra dev

QWENRLCD_OUTPUT_DIR="$(uv run python - "$QWENRLCD_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["output_dir"])
PY
)"
if [ -e "$QWENRLCD_OUTPUT_DIR" ]; then
  echo "[pilot] refusing to reuse existing output directory: $QWENRLCD_OUTPUT_DIR" >&2
  exit 2
fi

uv run pytest -q tests/test_losses.py
uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"
echo "[pilot] passed"
echo "[pilot] metrics: $QWENRLCD_OUTPUT_DIR/validation_metrics.json"
