#!/usr/bin/env bash
# Stop at step 7, resume from the newest complete checkpoint, and finish at step 20.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_resume_test.json}"

# shellcheck source=/dev/null
source "$QWENRLCD_SETUP_DIR/runpod-activate.sh"

QWENRLCD_OUTPUT_DIR="$(uv run python - "$QWENRLCD_CONFIG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["output_dir"])
PY
)"
QWENRLCD_REFERENCE_DIR="$QWENRLCD_OUTPUT_DIR-reference"

if [ -e "$QWENRLCD_OUTPUT_DIR" ] || [ -e "$QWENRLCD_REFERENCE_DIR" ]; then
  echo "[resume-test] refusing to reuse existing output directories:" >&2
  echo "  $QWENRLCD_OUTPUT_DIR" >&2
  echo "  $QWENRLCD_REFERENCE_DIR" >&2
  echo "[resume-test] move them aside or choose a config with a fresh output_dir" >&2
  exit 2
fi

echo "[resume-test] phase 1: train through step 7"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG" --stop-after-step 7
test -f "$QWENRLCD_OUTPUT_DIR/checkpoint-7/trainer_state.json"
test -f "$QWENRLCD_OUTPUT_DIR/checkpoint-7/trainable_model.safetensors"
test ! -e "$QWENRLCD_OUTPUT_DIR/checkpoint-7/model.safetensors"
test ! -e "$QWENRLCD_OUTPUT_DIR/final"

echo "[resume-test] phase 2: resume latest and finish"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG" --resume-from latest

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"
uv run python - "$QWENRLCD_OUTPUT_DIR/validation_metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    final = json.load(handle)[-1]
if final["step"] != 20:
    raise SystemExit(f"expected final step 20, got {final['step']}")
print("resume final metrics", json.dumps(final, sort_keys=True))
PY

echo "[resume-test] phase 3: run an uninterrupted reference"
uv run qwenrlcd-train \
  --config "$QWENRLCD_CONFIG" \
  --output-dir "$QWENRLCD_REFERENCE_DIR"

uv run qwenrlcd-predict \
  --run-dir "$QWENRLCD_OUTPUT_DIR" \
  --input data/overfit.jsonl \
  --output "$QWENRLCD_OUTPUT_DIR/resume_predictions.jsonl"
uv run qwenrlcd-predict \
  --run-dir "$QWENRLCD_REFERENCE_DIR" \
  --input data/overfit.jsonl \
  --output "$QWENRLCD_REFERENCE_DIR/reference_predictions.jsonl"

uv run python - \
  "$QWENRLCD_OUTPUT_DIR/resume_predictions.jsonl" \
  "$QWENRLCD_REFERENCE_DIR/reference_predictions.jsonl" <<'PY'
import json
import sys

def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


resumed = read_jsonl(sys.argv[1])
reference = read_jsonl(sys.argv[2])
max_delta = 0.0
for resumed_bundle, reference_bundle in zip(resumed, reference, strict=True):
    if resumed_bundle["id"] != reference_bundle["id"]:
        raise SystemExit("prediction bundle order differs")
    for question_id, resumed_question in resumed_bundle["questions"].items():
        reference_question = reference_bundle["questions"][question_id]
        for key, probability in resumed_question["probabilities"].items():
            delta = abs(probability - reference_question["probabilities"][key])
            max_delta = max(max_delta, delta)

print(f"resume/reference max probability delta={max_delta:.8f}")
if max_delta > 0.00001:
    raise SystemExit("resumed training diverged from uninterrupted reference")
PY

echo "[resume-test] passed"
