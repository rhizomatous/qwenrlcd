#!/usr/bin/env bash
# Retain completed run artifacts; preview first, pass --apply to remove old checkpoints.
set -euo pipefail

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--apply" ]; }; then
  echo "usage: bash setup/runpod-reclaim-space.sh [--apply]" >&2
  exit 2
fi

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
export PYTHONPATH="$QWENRLCD_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

QWENRLCD_COMPLETED_RUNS=(
  outputs/qwen3-1.7b-resume-test
  outputs/qwen3-1.7b-resume-test-reference
  outputs/qwen3-1.7b-choice-diagnostic-v0
  outputs/qwen3-1.7b-core-pilot-v0
)

echo "[reclaim] previewing all targets"
for QWENRLCD_RUN in "${QWENRLCD_COMPLETED_RUNS[@]}"; do
  .venv/bin/python -m qwenrlcd.prune_checkpoints \
    --run-dir "$QWENRLCD_RUN" --keep 0
done
.venv/bin/python -m qwenrlcd.prune_checkpoints \
  --run-dir outputs/qwen3-1.7b-choice-capacity-32-v0 \
  --keep 1 --include-incomplete

if [ "${1:-}" = "--apply" ]; then
  echo "[reclaim] applying only the previewed checkpoint cleanup"
  for QWENRLCD_RUN in "${QWENRLCD_COMPLETED_RUNS[@]}"; do
    .venv/bin/python -m qwenrlcd.prune_checkpoints \
      --run-dir "$QWENRLCD_RUN" --keep 0 --apply
  done
  .venv/bin/python -m qwenrlcd.prune_checkpoints \
    --run-dir outputs/qwen3-1.7b-choice-capacity-32-v0 \
    --keep 1 --include-incomplete --apply
fi

df -h /workspace
