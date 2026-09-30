#!/usr/bin/env bash
# Package durable RunPod results without caches, checkpoints, or disposable runs.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

QWENRLCD_EXPORT_ROOT="${QWENRLCD_EXPORT_ROOT:-$QWENRLCD_ROOT/artifacts}"
QWENRLCD_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
QWENRLCD_EXPORT_PATH="${QWENRLCD_EXPORT_PATH:-$QWENRLCD_EXPORT_ROOT/qwenrlcd-runpod-$QWENRLCD_TIMESTAMP.tar.gz}"
QWENRLCD_SELECTED_RUN="outputs/qwen3-1.7b-core-option-full-v0"
QWENRLCD_DIAGNOSTIC_RUNS=(
  "outputs/qwen3-4b-core-option-10k-lora-v0"
  "outputs/qwen3-4b-core-option-full-lora-v0"
)

test -f "$QWENRLCD_SELECTED_RUN/final/decision_head.pt"
test -f "$QWENRLCD_SELECTED_RUN/sealed_evaluation.json"

mkdir -p "$QWENRLCD_EXPORT_ROOT"
mkdir -p "$(dirname "$QWENRLCD_EXPORT_PATH")"
QWENRLCD_STAGING="$(mktemp -d "$QWENRLCD_EXPORT_ROOT/.runpod-export.XXXXXX")"
cleanup() {
  rm -rf -- "$QWENRLCD_STAGING"
}
trap cleanup EXIT

# Keep every compact result/config/benchmark record while preserving its output path.
while IFS= read -r -d '' QWENRLCD_FILE; do
  QWENRLCD_DESTINATION="$QWENRLCD_STAGING/$QWENRLCD_FILE"
  mkdir -p "$(dirname "$QWENRLCD_DESTINATION")"
  cp -p "$QWENRLCD_FILE" "$QWENRLCD_DESTINATION"
done < <(find outputs -type f -name '*.json' -print0)

copy_model_artifacts() {
  local run_dir="$1"
  local destination="$QWENRLCD_STAGING/$run_dir"
  if [ ! -d "$run_dir/final" ]; then
    echo "[export] model artifacts absent, skipping: $run_dir"
    return
  fi
  mkdir -p "$destination"
  cp -a "$run_dir/final" "$destination/"
  if [ -d "$run_dir/tokenizer" ]; then
    cp -a "$run_dir/tokenizer" "$destination/"
  fi
  echo "[export] retained model artifacts: $run_dir"
}

copy_model_artifacts "$QWENRLCD_SELECTED_RUN"
for QWENRLCD_RUN in "${QWENRLCD_DIAGNOSTIC_RUNS[@]}"; do
  copy_model_artifacts "$QWENRLCD_RUN"
done

git rev-parse HEAD > "$QWENRLCD_STAGING/repository-commit.txt"
git status --short > "$QWENRLCD_STAGING/repository-status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$QWENRLCD_STAGING/exported-at.txt"
printf '%s\n' \
  "Frozen selected model: $QWENRLCD_SELECTED_RUN" \
  "All outputs/**/*.json result records" \
  "Diagnostic model: ${QWENRLCD_DIAGNOSTIC_RUNS[0]}" \
  "Diagnostic model: ${QWENRLCD_DIAGNOSTIC_RUNS[1]}" \
  "Excluded: checkpoint-*, caches, virtual environments, and other model binaries" \
  > "$QWENRLCD_STAGING/CONTENTS.txt"

(
  cd "$QWENRLCD_STAGING"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

tar -czf "$QWENRLCD_EXPORT_PATH" -C "$QWENRLCD_STAGING" .
(
  cd "$(dirname "$QWENRLCD_EXPORT_PATH")"
  sha256sum "$(basename "$QWENRLCD_EXPORT_PATH")" \
    > "$(basename "$QWENRLCD_EXPORT_PATH").sha256"
)

echo "[export] archive: $QWENRLCD_EXPORT_PATH"
echo "[export] archive checksum: $QWENRLCD_EXPORT_PATH.sha256"
du -h "$QWENRLCD_EXPORT_PATH" "$QWENRLCD_EXPORT_PATH.sha256"
