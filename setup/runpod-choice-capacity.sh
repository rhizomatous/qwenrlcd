#!/usr/bin/env bash
# Fixed-order 32-bundle Choice capacity gate; validation deliberately repeats train.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_choice_capacity_32.json}"

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
QWENRLCD_TRAIN_FILE="$(uv run python - "$QWENRLCD_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
if config["train_file"] != config["validation_file"]:
    raise SystemExit("capacity gate requires identical train/validation files")
if config.get("permute_training", True):
    raise SystemExit("capacity gate requires permute_training=false")
print(config["train_file"])
PY
)"
if [ -e "$QWENRLCD_OUTPUT_DIR" ]; then
  echo "[capacity] refusing to reuse existing output directory: $QWENRLCD_OUTPUT_DIR" >&2
  exit 2
fi

uv run pytest -q tests/test_choice_capacity.py tests/test_losses.py
uv run qwenrlcd-choice-capacity \
  --pilot-config configs/qwen3_1_7b_core_pilot.json \
  --output "$QWENRLCD_TRAIN_FILE"
uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"
echo "[capacity] training and validation intentionally use the same 32 bundles"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"
uv run qwenrlcd-diagnose-choice --run-dir "$QWENRLCD_OUTPUT_DIR" --examples-per-source 0
uv run qwenrlcd-choice-capacity-report --run-dir "$QWENRLCD_OUTPUT_DIR"
