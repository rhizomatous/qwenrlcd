# qwenrlcd

An experimental non-autoregressive decision model built on `Qwen/Qwen3-1.7B-Base`. One request contains a shared string or structured state
and many typed questions. One model forward returns every question's probability distribution.

## Architecture

The physical input packs the state and all question branches into one sequence. Each branch uses the same logical positions immediately after the state. A tree attention mask permits a branch to see the state and itself, but never another question. Consequently adding, removing, or reordering an unrelated question cannot change the information available to any other branch.

Qwen returns a hidden state at every decision marker. A shared 255-slot linear head turns all markers into logits shaped `[batch, questions, choices]`. Invalid question
and choice slots are masked.

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
only the question type and instructions; the false/true output ordering remains fixed.

## Training data and metrics

The trainer accepts the JSONL fixtures via `train_file` and `validation_file`, or a
standard Hugging Face dataset via `dataset_path` and `dataset_config` (for example,
`"dataset_path": "../qwenrlcd-data", "dataset_config": "core"`). A Hub dataset ID
can replace the local path. The dataset adapter reads decisions and their source
label; source provenance details are not model inputs.

Loss is averaged over questions **within each bundle**, then over bundles. Validation
reports bundle-macro and source-macro metrics alongside source, question-type,
choice-count, question-count, and target-entropy slices. `accuracy` is argmax-label
agreement; `expected_accuracy` and ECE use the probability assigned by the target
distribution to the model's predicted class. NLL, Brier, target entropy, KL, and JS
divergence retain the full soft target.

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
qwenrlcd-predict --run-dir outputs/qwen3-1.7b-parallel-smoke --input data/smoke_inference.jsonl --output predictions.jsonl
```

The output is keyed by bundle and question IDs.
