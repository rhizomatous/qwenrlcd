#!/usr/bin/env bash
# One-shot test/test_ood evaluation of the frozen development-selected model.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_RUN_DIR="${QWENRLCD_RUN_DIR:-outputs/qwen3-1.7b-core-option-full-v0}"
QWENRLCD_REPORT="${QWENRLCD_REPORT:-$QWENRLCD_RUN_DIR/sealed_evaluation.json}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
uv sync --extra train --extra dev

test -f "$QWENRLCD_RUN_DIR/final/decision_head.pt"
test -f "$QWENRLCD_RUN_DIR/validation_metrics.json"
uv run pytest -q \
  tests/test_evaluate_sealed.py \
  tests/test_data.py \
  tests/test_losses.py \
  tests/test_metrics.py

uv run qwenrlcd-evaluate-sealed \
  --run-dir "$QWENRLCD_RUN_DIR" \
  --output "$QWENRLCD_REPORT" \
  --attention-backend sdpa

uv run python - "$QWENRLCD_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if not report.get("selection_frozen_before_evaluation"):
    raise SystemExit("model selection was not frozen before sealed evaluation")
if not report.get("complete") or set(report.get("splits", {})) != {"test", "test_ood"}:
    raise SystemExit("sealed evaluation is incomplete")
print("[sealed] complete; model selection remains frozen")
PY
