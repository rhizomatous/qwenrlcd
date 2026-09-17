#!/usr/bin/env bash
# End-to-end packed-question forward/backward/evaluate/save smoke run.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_lora.json}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

echo "[smoke] config=$QWENRLCD_CONFIG"
uv run qwenrlcd-check-parallel --config "$QWENRLCD_CONFIG"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

QWENRLCD_OUTPUT_DIR="$(uv run python - "$QWENRLCD_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["output_dir"])
PY
)"

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"

echo "[smoke] passed"
echo "[smoke] metrics: $QWENRLCD_OUTPUT_DIR/validation_metrics.json"
