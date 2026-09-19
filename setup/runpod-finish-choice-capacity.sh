#!/usr/bin/env bash
# Resume the interrupted fixed-order Choice fit from its intact step-300 checkpoint.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
export PYTHONPATH="$QWENRLCD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QWENRLCD_OUTPUT_DIR=outputs/qwen3-1.7b-choice-capacity-32-v0
QWENRLCD_CONFIG=configs/qwen3_1_7b_choice_capacity_32.json
test -f artifacts/choice_capacity_32.jsonl
test -f "$QWENRLCD_OUTPUT_DIR/checkpoint-300/trainer_state.json"
test -f "$QWENRLCD_OUTPUT_DIR/checkpoint-300/model.safetensors"
test ! -e "$QWENRLCD_OUTPUT_DIR/final"

.venv/bin/python - <<'PY'
from pathlib import Path

from qwenrlcd.checkpointing import resolve_resume_checkpoint

run_dir = Path("outputs/qwen3-1.7b-choice-capacity-32-v0")
latest = resolve_resume_checkpoint("latest", run_dir)
if latest.name != "checkpoint-300":
    raise SystemExit(f"expected checkpoint-300, found {latest}")
print(f"[capacity] resuming from {latest}")
PY

uv sync --extra train --extra dev
uv run pytest -q tests/test_compact_checkpoint.py tests/test_checkpointing.py
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG" --resume-from latest

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
uv run qwenrlcd-diagnose-choice --run-dir "$QWENRLCD_OUTPUT_DIR" --examples-per-source 0
uv run qwenrlcd-choice-capacity-report --run-dir "$QWENRLCD_OUTPUT_DIR"

# The final adapter, metrics, and diagnostic have passed; the old 3.7 GB
# training-state checkpoint is no longer needed for this capacity test.
.venv/bin/python -m qwenrlcd.prune_checkpoints \
  --run-dir "$QWENRLCD_OUTPUT_DIR" --keep 0 --apply
du -sh /workspace
