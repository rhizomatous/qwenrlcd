#!/usr/bin/env bash
# Full-core Qwen3-4B LoRA training using the selected capacity-study recipe.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export QWENRLCD_CONFIG="${QWENRLCD_CONFIG:-configs/qwen3_4b_core_option_full_lora.json}"

exec "$QWENRLCD_SETUP_DIR/runpod-full.sh"
