#!/usr/bin/env bash
# One-time, idempotent Runpod setup for qwenrlcd.
#
# From the repository root:
#   bash setup/runpod-init.sh
#   bash setup/runpod-init.sh Qwen/Qwen3-1.7B-Base
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_REPO_ROOT="$(cd "$QWENRLCD_SETUP_DIR/.." && pwd)"
QWENRLCD_MODEL_ID="${1:-Qwen/Qwen3-1.7B-Base}"

if [ "$#" -gt 1 ]; then
  echo "usage: bash setup/runpod-init.sh [MODEL_ID]" >&2
  exit 2
fi

cd "$QWENRLCD_REPO_ROOT"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "[init] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

echo "[init] installing Python 3.12 and project dependencies"
uv python install 3.12
uv sync --python 3.12 --extra train --extra dev

echo "[init] checking the package and CUDA runtime"
uv run python - <<'PY'
import sys

import torch
import transformers

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable; do not start training on CPU")

properties = torch.cuda.get_device_properties(0)
print("device", properties.name)
print("vram_gib", round(properties.total_memory / 2**30, 1))
print("bf16_supported", torch.cuda.is_bf16_supported())
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: this project is configured for BF16 training")
PY

echo "[init] running dependency-light tests"
uv run pytest -q

echo "[init] caching $QWENRLCD_MODEL_ID in $HF_HOME"
uv run hf download "$QWENRLCD_MODEL_ID" \
  --exclude "*.gguf" "*.onnx" "*.msgpack" "*.h5" "*.ot"

cat <<'EOF'

[init] ready.

In each new shell:
  source setup/runpod-activate.sh

Then run the end-to-end GPU smoke test inside tmux:
  tmux new -s qwenrlcd
  bash setup/runpod-smoke.sh

Detach with Ctrl-b then d; reattach with: tmux attach -t qwenrlcd
EOF
