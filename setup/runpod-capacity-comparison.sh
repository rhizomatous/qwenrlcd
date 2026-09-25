#!/usr/bin/env bash
# Compare adaptation capacity without opening core test or test_ood.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONTROL_RUN="${QWENRLCD_CONTROL_RUN:-outputs/qwen3-1.7b-core-option-10k-v0}"
QWENRLCD_FULL_CONFIG="${QWENRLCD_FULL_CONFIG:-configs/qwen3_1_7b_core_option_10k_full.json}"
QWENRLCD_LARGE_CONFIG="${QWENRLCD_LARGE_CONFIG:-configs/qwen3_4b_core_option_10k_lora.json}"
QWENRLCD_REPORT="${QWENRLCD_REPORT:-outputs/qwen3-capacity-comparison-10k-v0.json}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
uv sync --extra train --extra dev

config_output_dir() {
  uv run python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["output_dir"])
PY
}

run_branch() {
  local label="$1"
  local config="$2"
  local output_dir
  local -a train_args
  output_dir="$(config_output_dir "$config")"
  train_args=(--config "$config")
  if [ -f "$output_dir/.capacity-complete" ]; then
    echo "[capacity] $label already complete: $output_dir"
    uv run python - "$output_dir" <<'PY'
import sys

from qwenrlcd.checkpointing import prune_checkpoints

removed = prune_checkpoints(sys.argv[1], keep_last=0, include_incomplete=True)
if removed:
    print("[capacity] removed completed-run checkpoints:", ", ".join(p.name for p in removed))
PY
    return
  fi
  if [ -e "$output_dir" ]; then
    if compgen -G "$output_dir/checkpoint-*" >/dev/null; then
      echo "[capacity] resuming $label from its latest checkpoint"
      train_args+=(--resume-from latest)
    else
      echo "[capacity] refusing non-resumable partial output: $output_dir" >&2
      exit 2
    fi
  else
    echo "[capacity] preflight $label"
    uv run qwenrlcd-preflight --config "$config"
  fi
  echo "[capacity] train $label"
  uv run qwenrlcd-train "${train_args[@]}"
  test -f "$output_dir/final/decision_head.pt"
  test -f "$output_dir/final/decision_config.json"
  test -f "$output_dir/validation_metrics.json"
  touch "$output_dir/.capacity-complete"
  uv run python - "$output_dir" <<'PY'
import sys

from qwenrlcd.checkpointing import prune_checkpoints

removed = prune_checkpoints(sys.argv[1], keep_last=0, include_incomplete=True)
if removed:
    print("[capacity] removed completed-run checkpoints:", ", ".join(p.name for p in removed))
PY
}

if [ ! -f "$QWENRLCD_CONTROL_RUN/final/decision_head.pt" ] \
  || [ ! -f "$QWENRLCD_CONTROL_RUN/validation_metrics.json" ]; then
  echo "[capacity] missing completed 1.7B LoRA control: $QWENRLCD_CONTROL_RUN" >&2
  exit 2
fi

uv run python - \
  "$QWENRLCD_CONTROL_RUN/training_config.json" \
  "$QWENRLCD_FULL_CONFIG" \
  "$QWENRLCD_LARGE_CONFIG" <<'PY'
import json
import sys

from qwenrlcd.compare_capacity import COHORT_KEYS

with open(sys.argv[1], encoding="utf-8") as handle:
    control = json.load(handle)
for path in sys.argv[2:]:
    with open(path, encoding="utf-8") as handle:
        candidate = json.load(handle)
    differences = {
        key: (control.get(key), candidate.get(key))
        for key in COHORT_KEYS
        if control.get(key) != candidate.get(key)
    }
    if differences:
        raise SystemExit(f"capacity cohort mismatch in {path}: {differences}")
print("[capacity] configs share the control's train/development cohort and schedule")
PY

uv run pytest -q \
  tests/test_model.py \
  tests/test_losses.py \
  tests/test_metrics.py \
  tests/test_preflight.py \
  tests/test_compare_capacity.py

run_branch "1.7B full fine-tune" "$QWENRLCD_FULL_CONFIG"
run_branch "4B LoRA" "$QWENRLCD_LARGE_CONFIG"

QWENRLCD_FULL_RUN="$(config_output_dir "$QWENRLCD_FULL_CONFIG")"
QWENRLCD_LARGE_RUN="$(config_output_dir "$QWENRLCD_LARGE_CONFIG")"
uv run qwenrlcd-compare-capacity \
  --control-run "$QWENRLCD_CONTROL_RUN" \
  --full-run "$QWENRLCD_FULL_RUN" \
  --large-lora-run "$QWENRLCD_LARGE_RUN" \
  --output "$QWENRLCD_REPORT"

echo "[capacity] complete; test and test_ood remain sealed"
