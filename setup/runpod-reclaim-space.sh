#!/usr/bin/env bash
# Retain completed model artifacts; preview first, pass --apply to remove their checkpoints.
set -euo pipefail

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--apply" ]; }; then
  echo "usage: bash setup/runpod-reclaim-space.sh [--apply]" >&2
  exit 2
fi

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
export PYTHONPATH="$QWENRLCD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
QWENRLCD_COMPLETED_RUNS=()
for QWENRLCD_RUN in outputs/*; do
  if [ ! -d "$QWENRLCD_RUN" ] || [ -L "$QWENRLCD_RUN" ]; then
    continue
  fi
  if [ ! -f "$QWENRLCD_RUN/training_config.json" ] \
    || [ ! -f "$QWENRLCD_RUN/validation_metrics.json" ] \
    || [ ! -f "$QWENRLCD_RUN/final/decision_head.pt" ] \
    || [ ! -f "$QWENRLCD_RUN/final/decision_config.json" ]; then
    continue
  fi
  if compgen -G "$QWENRLCD_RUN/checkpoint-*" >/dev/null \
    || compgen -G "$QWENRLCD_RUN/.checkpoint-*.incomplete" >/dev/null; then
    QWENRLCD_COMPLETED_RUNS+=("$QWENRLCD_RUN")
  fi
done

echo "[reclaim] previewing checkpoints from ${#QWENRLCD_COMPLETED_RUNS[@]} completed runs"
for QWENRLCD_RUN in "${QWENRLCD_COMPLETED_RUNS[@]}"; do
  .venv/bin/python -m qwenrlcd.prune_checkpoints \
    --run-dir "$QWENRLCD_RUN" --keep 0 --include-incomplete
done

if [ "${1:-}" = "--apply" ]; then
  echo "[reclaim] applying only the previewed checkpoint cleanup"
  for QWENRLCD_RUN in "${QWENRLCD_COMPLETED_RUNS[@]}"; do
    .venv/bin/python -m qwenrlcd.prune_checkpoints \
      --run-dir "$QWENRLCD_RUN" --keep 0 --include-incomplete --apply
  done
fi

du -sh /workspace
