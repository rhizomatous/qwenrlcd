#!/usr/bin/env bash
# Train on the existing 32-bundle Choice fixture with question/option permutation.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

QWENRLCD_CONFIG=configs/qwen3_1_7b_choice_permutation_32.json
QWENRLCD_BASELINE=outputs/qwen3-1.7b-choice-capacity-32-v0
QWENRLCD_RUN=outputs/qwen3-1.7b-choice-permutation-32-v0

test -f artifacts/choice_capacity_32.jsonl
test -f "$QWENRLCD_BASELINE/final/decision_head.pt"
if [ -e "$QWENRLCD_RUN" ]; then
  echo "[permutation] refusing to reuse existing output directory: $QWENRLCD_RUN" >&2
  exit 2
fi

uv sync --extra train --extra dev
uv run pytest -q tests/test_diagnose_choice.py tests/test_choice_capacity.py
uv run python - "$QWENRLCD_CONFIG" "$QWENRLCD_BASELINE/training_config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    permutation = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    baseline = json.load(handle)
for key in (
    "model_id", "train_file", "validation_file", "seed", "max_length",
    "max_questions", "max_choices", "epochs", "micro_batch_size",
    "gradient_accumulation_steps", "learning_rate", "head_learning_rate",
    "weight_decay", "warmup_ratio", "max_grad_norm", "ce_weight",
    "brier_weight", "lora",
):
    if permutation[key] != baseline[key]:
        raise SystemExit(f"[permutation] mismatched control setting: {key}")
if not permutation["permute_training"] or baseline["permute_training"]:
    raise SystemExit("[permutation] expected permutation on only for the new run")
if permutation["train_file"] != permutation["validation_file"]:
    raise SystemExit("[permutation] fit experiment requires identical train/validation files")
print("[permutation] matched baseline settings and fixture")
PY
uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"

# Measure order sensitivity in the already-fitted fixed-order control.
uv run qwenrlcd-diagnose-choice \
  --run-dir "$QWENRLCD_BASELINE" --split train --permutation-check

uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"
test -f "$QWENRLCD_RUN/final/decision_head.pt"
uv run qwenrlcd-diagnose-choice \
  --run-dir "$QWENRLCD_RUN" --split train --permutation-check

uv run python - "$QWENRLCD_BASELINE" "$QWENRLCD_RUN" <<'PY'
import json
import sys
from pathlib import Path

for run in (Path(sys.argv[1]), Path(sys.argv[2])):
    with (run / "choice_permutation_diagnostic.json").open(encoding="utf-8") as handle:
        report = json.load(handle)
    print(f"[permutation] {run.name}")
    for view, summary in report["views"].items():
        overall = summary["overall"]
        model = overall["model"]
        uniform = overall["uniform"]
        print(
            f"  {view}: n={overall['count']} "
            f"brier={model['brier']:.4f} (uniform {uniform['brier']:.4f}) "
            f"kl={model['kl_divergence']:.4f} "
            f"(uniform {uniform['kl_divergence']:.4f})"
        )
        for source, metrics in summary["by_source"].items():
            print(
                f"    {source}: brier={metrics['model']['brier']:.4f} "
                f"kl={metrics['model']['kl_divergence']:.4f}"
            )
    consistency = report["canonical_vs_fresh"]["overall"]
    print(
        f"  fresh order: mean TV={consistency['mean_total_variation']:.4f} "
        f"argmax agreement={consistency['argmax_agreement']:.3f} "
        f"options reordered={consistency['option_order_changed']}/"
        f"{consistency['questions']}"
    )
PY
echo "[permutation] experiment complete: fit and order-robustness metrics only"
echo "[permutation] this is not holdout validation"
