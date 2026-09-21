#!/usr/bin/env bash
# Full core training. Optionally stop at a real resumable step for timing calibration.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${QWENRLCD_CONFIG:-configs/qwen3_1_7b_core_option_full.json}"
QWENRLCD_STOP_AFTER_STEP="${QWENRLCD_STOP_AFTER_STEP:-}"

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

QWENRLCD_TRAIN_ARGS=(--config "$QWENRLCD_CONFIG")
if [ -e "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt" ]; then
  echo "[full] final model already exists: $QWENRLCD_OUTPUT_DIR" >&2
  exit 2
elif [ -d "$QWENRLCD_OUTPUT_DIR" ]; then
  if compgen -G "$QWENRLCD_OUTPUT_DIR/checkpoint-*" >/dev/null; then
    echo "[full] resuming latest checkpoint in $QWENRLCD_OUTPUT_DIR"
    QWENRLCD_TRAIN_ARGS+=(--resume-from latest)
  else
    echo "[full] output exists without a resumable checkpoint: $QWENRLCD_OUTPUT_DIR" >&2
    exit 2
  fi
else
  uv run pytest -q \
    tests/test_data.py \
    tests/test_model.py \
    tests/test_losses.py \
    tests/test_metrics.py \
    tests/test_preflight.py \
    tests/test_train_validation.py
  uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"
fi

if [ -n "$QWENRLCD_STOP_AFTER_STEP" ]; then
  QWENRLCD_TRAIN_ARGS+=(--stop-after-step "$QWENRLCD_STOP_AFTER_STEP")
fi
uv run qwenrlcd-train "${QWENRLCD_TRAIN_ARGS[@]}"

test -f "$QWENRLCD_OUTPUT_DIR/training_timing.json"
uv run python - "$QWENRLCD_OUTPUT_DIR/training_timing.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    timing = json.load(handle)
print(
    "[full] timing",
    f"step={timing['current_step']}/{timing['total_steps']}",
    f"seconds_per_step={timing['seconds_per_optimizer_step']:.3f}",
    f"remaining_training_hours={timing['remaining_training_seconds'] / 3600:.2f}",
)
PY

if [ -n "$QWENRLCD_STOP_AFTER_STEP" ]; then
  echo "[full] calibration checkpoint saved; rerun without QWENRLCD_STOP_AFTER_STEP"
  exit 0
fi

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"
uv run python - "$QWENRLCD_OUTPUT_DIR/validation_metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metrics = json.load(handle)[-1]
print(
    "[full] development validation",
    f"step={metrics['step']}",
    f"bundles={metrics['bundles']}",
    f"questions={metrics['questions']}",
    f"brier={metrics['brier']:.4f}",
    f"kl={metrics['kl']:.4f}",
    f"ece={metrics['ece']:.4f}",
)
for name, values in metrics["slices"]["by_type"].items():
    print(
        f"[full] {name}",
        f"n={values['count']}",
        f"brier={values['brier']:.4f}",
        f"kl={values['kl_divergence']:.4f}",
    )
PY
echo "[full] complete: $QWENRLCD_OUTPUT_DIR"
