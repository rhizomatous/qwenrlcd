# qwenrlcd

An experimental non-autoregressive decision model built on `Qwen/Qwen3-1.7B-Base`. One request contains a shared string or structured state
and many typed questions. One model forward returns every question's probability distribution.

## Architecture

The physical input packs a shared state, question prefixes, and option branches into one sequence. A tree attention mask lets each question see the state, and each option see the state, its question, and itself—but no sibling options or other questions. Logical positions reset for each question and option branch.

Qwen returns a hidden state at each option's decision marker. The same scalar head scores every option; softmax within each question produces its probability distribution. Invalid padded slots are masked. Reordering options reorders their scores by key without changing what any option branch can see. All questions and options are evaluated in one forward pass.

This replaces the earlier positional-head prototype; its saved model components are not loaded by the current implementation.

The initial implementation uses eager attention and a dense tree mask. FlashAttention cannot express this topology directly. Sparse/FlexAttention optimization may be possible optimizations but have not been tried yet.

## Typed questions

The query schema is borrowed from [TypeSafe Primitives](https://docs.typesafe.ai/primitives):

- `choice`: an unordered distribution over caller-defined options.
- `score`: a distribution over ordered levels. the score is its expected level.
- `noul`: a binary false/true distribution reported as the probability of true.

## Canonical JSONL

Each line is one shared state with a map of labeled questions:

```json
{
  "id": "ticket-001",
  "state": {
    "message": "My order arrived broken and I need a replacement.",
    "account_balance_usd": -12
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle `message`?",
      "criteria": {
        "returns": "Exchanges, refunds, or damaged items",
        "shipping": "Late or missing deliveries",
        "billing": "Charges and payment problems"
      },
      "target": {"returns": 0.9, "shipping": 0.05, "billing": 0.05}
    },
    "negative_balance": {
      "type": "noul",
      "instructions": "Is `account_balance_usd` negative?",
      "target": {"false": 0.0, "true": 1.0}
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this request?",
      "criteria": ["routine", "time-sensitive", "urgent"],
      "target": {"1": 0.75, "2": 0.25}
    }
  }
}
```

Question IDs are response-routing keys and are deliberately excluded from model input.
Noul `criteria` is optional, matching the System One API. When omitted, the model sees
the question type, instructions, and the `false`/`true` option keys, without
fabricated criterion descriptions.

## Training data and metrics

The trainer accepts the JSONL fixtures via `train_file` and `validation_file`, or a
standard Hugging Face dataset via `dataset_path` and `dataset_config` (for example,
`"dataset_path": "../qwenrlcd-data", "dataset_config": "core"`). A Hub dataset ID
can replace the local path. The dataset adapter reads decisions and their source
label; source provenance details are not model inputs.

For a bounded pilot, `train_bundle_limit` and `validation_bundle_limit` select a
deterministic source-stratified subset of whole bundles. `qwenrlcd-preflight --config
configs/qwen3_1_7b_core_option_pilot.json` checks packed token lengths without loading the
model; the pilot config also forbids state truncation during training.

`question_type_filter` can retain only one decision type from those same sampled
bundles for a controlled training run. `qwenrlcd-diagnose-choice --run-dir
outputs/your-run` reloads a saved model and reports Choice metrics
by source and option count against a uniform baseline, with example predictions.
For a run trained with option permutation, add `--split train
--training-epoch-view 3` to evaluate the exact packed inputs seen in epoch 3
instead of the canonical train order. Epoch numbers are zero-based.

Loss is averaged over questions **within each bundle**, then over bundles. Validation
reports bundle-macro and source-macro metrics alongside source, question-type,
choice-count, question-count, and target-entropy slices. `accuracy` is argmax-label
agreement; `expected_accuracy` and ECE use the probability assigned by the target
distribution to the model's predicted class. NLL, Brier, target entropy, KL, and JS
divergence retain the full soft target.
The pilot reports Brier and KL against a uniform distribution over each question's
options, both overall and by source and question type.

## Tests

The test suite covers schema validation, soft targets, formatting, tree-attention isolation, logical position reset, deterministic fixture generation, and calibration
metrics.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

## Inference

A saved run contains the tokenizer, LoRA adapter, decision head, and training configuration. It can be used like so:

```bash
qwenrlcd-predict --run-dir outputs/your-run --input data/smoke_inference.jsonl --output predictions.jsonl
```

The output is keyed by bundle and question IDs.
