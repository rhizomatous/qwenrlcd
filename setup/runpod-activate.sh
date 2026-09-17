#!/usr/bin/env bash
# Per-shell environment for qwenrlcd on Runpod.
#
# Source this file; executing it cannot modify the parent shell:
#   source setup/runpod-activate.sh

_qwenrlcd_source="${BASH_SOURCE[0]:-$0}"
if [ "$_qwenrlcd_source" = "$0" ]; then
  echo "source this script instead: source setup/runpod-activate.sh" >&2
  exit 2
fi

QWENRLCD_ROOT="$(cd "$(dirname "$_qwenrlcd_source")/.." && pwd)"
export QWENRLCD_ROOT

# Runpod mounts persistent storage at /workspace. Fall back to repo-local caches
# when this is sourced elsewhere. All values remain user-overridable.
if [ -d /workspace ] && [ -w /workspace ]; then
  _qwenrlcd_cache_root=/workspace/.cache/qwenrlcd
else
  _qwenrlcd_cache_root="$QWENRLCD_ROOT/.cache"
fi

: "${HF_HOME:=$_qwenrlcd_cache_root/huggingface}"
: "${UV_CACHE_DIR:=$_qwenrlcd_cache_root/uv}"
export HF_HOME UV_CACHE_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The uv installer places its executables here.
if [ -f "$HOME/.local/bin/env" ]; then
  # shellcheck source=/dev/null
  . "$HOME/.local/bin/env"
fi
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

cd "$QWENRLCD_ROOT" || return 1
echo "activated: root=$QWENRLCD_ROOT hf_cache=$HF_HOME uv=$(command -v uv || echo MISSING)"

unset _qwenrlcd_source _qwenrlcd_cache_root
