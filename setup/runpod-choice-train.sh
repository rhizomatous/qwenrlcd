#!/usr/bin/env bash
# Controlled Choice-only capacity/generalization diagnostic on the pilot sample.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_choice_diagnostic.json}"

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
  echo "[choice] refusing to reuse existing output directory: $QWENRLCD_OUTPUT_DIR" >&2
  exit 2
fi

uv run pytest -q tests/test_hf_data.py tests/test_diagnose_choice.py
uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"
uv run qwenrlcd-diagnose-choice \
  --run-dir "$QWENRLCD_OUTPUT_DIR" \
  --split train \
  --examples-per-source 0
uv run qwenrlcd-diagnose-choice \
  --run-dir "$QWENRLCD_OUTPUT_DIR" \
  --split validation
uv run python - "$QWENRLCD_OUTPUT_DIR/validation_metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    history = json.load(handle)
for metrics in history:
    print(
        "[choice] validation",
        f"epoch={metrics['epoch']}",
        f"step={metrics['step']}",
        f"accuracy={metrics['accuracy']:.4f}",
        f"brier={metrics['brier']:.4f}",
        f"kl={metrics['kl']:.4f}",
        f"ece={metrics['ece']:.4f}",
    )
PY
echo "[choice] passed"
echo "[choice] metrics: $QWENRLCD_OUTPUT_DIR/validation_metrics.json"
