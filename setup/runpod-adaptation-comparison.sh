#!/usr/bin/env bash
# Tune full-fine-tune LR, then compare 1.7B LoRA/full across fixed-data training seeds.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_EXPERIMENT_DIR="${QWENRLCD_EXPERIMENT_DIR:-outputs/qwen3-1.7b-adaptation-comparison-v0}"
QWENRLCD_EXISTING_FULL_RUN="${QWENRLCD_EXISTING_FULL_RUN:-outputs/qwen3-1.7b-core-option-10k-full-v0}"
QWENRLCD_LORA_CONFIGS=(
  configs/qwen3_1_7b_adaptation_lora_lr1e4.json
  configs/qwen3_1_7b_adaptation_lora_sdpa.json
  configs/qwen3_1_7b_adaptation_lora_lr4e4.json
)
QWENRLCD_FULL_CONFIGS=(
  configs/qwen3_1_7b_adaptation_full_lr5e6.json
  configs/qwen3_1_7b_adaptation_full_lr1e5.json
  configs/qwen3_1_7b_adaptation_full_lr4e5.json
)

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

cleanup_completed_checkpoints() {
  uv run python - "$1" <<'PY'
import sys

from qwenrlcd.checkpointing import prune_checkpoints

removed = prune_checkpoints(sys.argv[1], keep_last=0, include_incomplete=True)
if removed:
    print("[adaptation] removed completed-run checkpoints:", ", ".join(p.name for p in removed))
PY
}

run_branch() {
  local label="$1"
  local config="$2"
  local output_dir
  local -a train_args
  output_dir="$(config_output_dir "$config")"
  train_args=(--config "$config")
  if [ -f "$output_dir/.adaptation-complete" ]; then
    echo "[adaptation] $label already complete: $output_dir"
    cleanup_completed_checkpoints "$output_dir"
    return
  fi
  if [ -e "$output_dir" ]; then
    if compgen -G "$output_dir/checkpoint-*" >/dev/null; then
      echo "[adaptation] resuming $label from its latest checkpoint"
      train_args+=(--resume-from latest)
    else
      echo "[adaptation] refusing non-resumable partial output: $output_dir" >&2
      exit 2
    fi
  else
    echo "[adaptation] preflight $label"
    uv run qwenrlcd-preflight --config "$config"
  fi
  echo "[adaptation] train $label"
  uv run qwenrlcd-train "${train_args[@]}"
  test -f "$output_dir/final/decision_head.pt"
  test -f "$output_dir/final/decision_config.json"
  test -f "$output_dir/validation_metrics.json"
  touch "$output_dir/.adaptation-complete"
  cleanup_completed_checkpoints "$output_dir"
}

if [ ! -f "$QWENRLCD_EXISTING_FULL_RUN/final/decision_head.pt" ] \
  || [ ! -f "$QWENRLCD_EXISTING_FULL_RUN/validation_metrics.json" ]; then
  echo "[adaptation] missing completed full-FT lr=2e-5 run: $QWENRLCD_EXISTING_FULL_RUN" >&2
  exit 2
fi

uv run pytest -q \
  tests/test_hf_data.py \
  tests/test_compare_adaptation.py \
  tests/test_model.py \
  tests/test_losses.py \
  tests/test_metrics.py \
  tests/test_preflight.py

uv run python - \
  "${QWENRLCD_LORA_CONFIGS[@]}" \
  "$QWENRLCD_EXISTING_FULL_RUN/training_config.json" \
  "${QWENRLCD_FULL_CONFIGS[@]}" <<'PY'
import json
import sys

from qwenrlcd.compare_adaptation import fixed_sample_cohort

configs = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        configs.append((path, json.load(handle)))
reference = fixed_sample_cohort(configs[0][1])
for path, config in configs:
    if config.get("attn_implementation") != "sdpa":
        raise SystemExit(f"adaptation comparison requires SDPA: {path}")
for path, config in configs[1:]:
    candidate = fixed_sample_cohort(config)
    differences = {
        key: (reference[key], candidate[key])
        for key in reference
        if reference[key] != candidate[key]
    }
    if differences:
        raise SystemExit(f"adaptation cohort mismatch in {path}: {differences}")
print("[adaptation] configs share one fixed data cohort and training schedule")
PY

for QWENRLCD_CONFIG in "${QWENRLCD_LORA_CONFIGS[@]}"; do
  run_branch "LoRA seed=17 SDPA" "$QWENRLCD_CONFIG"
done
for QWENRLCD_CONFIG in "${QWENRLCD_FULL_CONFIGS[@]}"; do
  run_branch "full fine-tune seed=17" "$QWENRLCD_CONFIG"
done

QWENRLCD_LORA_SWEEP_RUNS=()
for QWENRLCD_CONFIG in "${QWENRLCD_LORA_CONFIGS[@]}"; do
  QWENRLCD_LORA_SWEEP_RUNS+=("$(config_output_dir "$QWENRLCD_CONFIG")")
done
QWENRLCD_FULL_SWEEP_RUNS=("$QWENRLCD_EXISTING_FULL_RUN")
for QWENRLCD_CONFIG in "${QWENRLCD_FULL_CONFIGS[@]}"; do
  QWENRLCD_FULL_SWEEP_RUNS+=("$(config_output_dir "$QWENRLCD_CONFIG")")
done

QWENRLCD_WINNING_LORA_RUN="$(uv run python - "${QWENRLCD_LORA_SWEEP_RUNS[@]}" <<'PY'
import sys

from qwenrlcd.compare_adaptation import select_best_lora_run
from qwenrlcd.compare_capacity import load_run

winner = select_best_lora_run([load_run(path) for path in sys.argv[1:]])
print(winner["run_dir"])
PY
)"
QWENRLCD_WINNING_FULL_RUN="$(uv run python - "${QWENRLCD_FULL_SWEEP_RUNS[@]}" <<'PY'
import sys

from qwenrlcd.compare_adaptation import select_best_full_run
from qwenrlcd.compare_capacity import load_run

winner = select_best_full_run([load_run(path) for path in sys.argv[1:]])
print(winner["run_dir"])
PY
)"
echo "[adaptation] selected LoRA: $QWENRLCD_WINNING_LORA_RUN"
echo "[adaptation] selected full fine-tune: $QWENRLCD_WINNING_FULL_RUN"

QWENRLCD_GENERATED_CONFIG_DIR="$QWENRLCD_EXPERIMENT_DIR/generated-configs"
mkdir -p "$QWENRLCD_GENERATED_CONFIG_DIR"
uv run python - \
  "$QWENRLCD_WINNING_LORA_RUN/training_config.json" \
  "$QWENRLCD_WINNING_FULL_RUN/training_config.json" \
  "$QWENRLCD_GENERATED_CONFIG_DIR" \
  "$QWENRLCD_EXPERIMENT_DIR" <<'PY'
import json
import sys
from pathlib import Path

lora_path, full_path, config_dir, experiment_dir = map(Path, sys.argv[1:])
config_dir.mkdir(parents=True, exist_ok=True)
for source_path, adaptation in ((lora_path, "lora"), (full_path, "full")):
    with source_path.open(encoding="utf-8") as handle:
        base = json.load(handle)
    for seed in (29, 43):
        config = {**base, "seed": seed, "dataset_seed": 17}
        config["output_dir"] = str(experiment_dir / f"{adaptation}-selected-seed{seed}")
        destination = config_dir / f"{adaptation}-seed{seed}.json"
        destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

QWENRLCD_LORA_RUNS=("$QWENRLCD_WINNING_LORA_RUN")
QWENRLCD_FULL_RUNS=("$QWENRLCD_WINNING_FULL_RUN")
for QWENRLCD_SEED in 29 43; do
  QWENRLCD_CONFIG="$QWENRLCD_GENERATED_CONFIG_DIR/lora-seed${QWENRLCD_SEED}.json"
  run_branch "LoRA selected recipe seed=$QWENRLCD_SEED" "$QWENRLCD_CONFIG"
  QWENRLCD_LORA_RUNS+=("$(config_output_dir "$QWENRLCD_CONFIG")")

  QWENRLCD_CONFIG="$QWENRLCD_GENERATED_CONFIG_DIR/full-seed${QWENRLCD_SEED}.json"
  run_branch "full selected recipe seed=$QWENRLCD_SEED" "$QWENRLCD_CONFIG"
  QWENRLCD_FULL_RUNS+=("$(config_output_dir "$QWENRLCD_CONFIG")")
done

QWENRLCD_COMPARE_ARGS=(--output "$QWENRLCD_EXPERIMENT_DIR/final-report.json")
for QWENRLCD_RUN in "${QWENRLCD_LORA_RUNS[@]}"; do
  QWENRLCD_COMPARE_ARGS+=(--lora-run "$QWENRLCD_RUN")
done
for QWENRLCD_RUN in "${QWENRLCD_FULL_RUNS[@]}"; do
  QWENRLCD_COMPARE_ARGS+=(--full-run "$QWENRLCD_RUN")
done
for QWENRLCD_RUN in "${QWENRLCD_LORA_SWEEP_RUNS[@]}"; do
  QWENRLCD_COMPARE_ARGS+=(--lora-sweep-run "$QWENRLCD_RUN")
done
for QWENRLCD_RUN in "${QWENRLCD_FULL_SWEEP_RUNS[@]}"; do
  QWENRLCD_COMPARE_ARGS+=(--full-sweep-run "$QWENRLCD_RUN")
done
uv run qwenrlcd-compare-adaptation "${QWENRLCD_COMPARE_ARGS[@]}"

echo "[adaptation] complete; test and test_ood remain sealed"
