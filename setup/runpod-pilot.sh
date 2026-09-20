#!/usr/bin/env bash
# Source-stratified core pilot, with a tokenizer-only length check before GPU work.
set -euo pipefail

QWENRLCD_SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
QWENRLCD_CONFIG="${1:-configs/qwen3_1_7b_core_option_pilot.json}"

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
if [ -e "$QWENRLCD_OUTPUT_DIR" ]; then
  echo "[pilot] refusing to reuse existing output directory: $QWENRLCD_OUTPUT_DIR" >&2
  exit 2
fi

uv run pytest -q tests/test_data.py tests/test_model.py tests/test_losses.py tests/test_metrics.py
uv run qwenrlcd-preflight --config "$QWENRLCD_CONFIG"
uv run qwenrlcd-train --config "$QWENRLCD_CONFIG"

test -f "$QWENRLCD_OUTPUT_DIR/final/decision_head.pt"
test -f "$QWENRLCD_OUTPUT_DIR/validation_metrics.json"
uv run python - "$QWENRLCD_OUTPUT_DIR/validation_metrics.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metrics = json.load(handle)[-1]
print(
    "[pilot] held-out validation",
    f"step={metrics['step']}",
    f"questions={metrics['questions']}",
    f"brier={metrics['brier']:.4f}",
    f"uniform_brier={metrics['uniform_brier']:.4f}",
    f"kl={metrics['kl']:.4f}",
    f"uniform_kl={metrics['uniform_kl']:.4f}",
)
if "reference_validation" in metrics:
    reference = metrics["reference_validation"]
    print(
        "[pilot] reference validation",
        f"bundles={reference['bundles']}",
        f"questions={reference['questions']}",
        f"brier={reference['brier']:.4f}",
        f"kl={reference['kl']:.4f}",
    )
for slice_name in ("by_source", "by_type"):
    weighting = "bundle-macro" if slice_name == "by_source" else "question-micro"
    print(f"[pilot] {slice_name} ({weighting})")
    for name, values in metrics["slices"][slice_name].items():
        print(
            f"  {name}: n={values['questions'] if slice_name == 'by_source' else values['count']} "
            f"brier={values['brier']:.4f} (uniform {values['uniform_brier']:.4f}) "
            f"kl={values['kl_divergence']:.4f} "
            f"(uniform {values['uniform_kl_divergence']:.4f})"
        )
PY
echo "[pilot] complete (inspect held-out metrics above)"
echo "[pilot] metrics: $QWENRLCD_OUTPUT_DIR/validation_metrics.json"
