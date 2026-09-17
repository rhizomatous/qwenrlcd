# qwenrlcd

An experimental calibrated decision model built on `Qwen/Qwen3.5-2B-Base`.
The model consumes unstructured state plus a typed question and returns a
probability distribution over declared options.

## Initial design

```text
state + question + option descriptions
                 |
                 v
        Qwen text backbone
                 |
       final <|decision|> state
                 |
       shared 255-slot linear head
                 |
     masked categorical distribution
```

The first milestone supports:

- `choice`: an unordered categorical decision;
- `noul`: a binary false/true probability;
- `score`: an ordered categorical distribution whose expected index is the score;
- hard or soft target distributions;
- option permutation during training;
- cross-entropy and multiclass Brier losses;
- LoRA over Qwen's attention, linear-attention, and MLP projections.

## Local setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

## Canonical JSONL format

Each line represents one independently evaluated question:

```json
{
  "id": "ticket-001",
  "state": "My shoes arrived in the wrong size.",
  "question": {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "options": [
      {"key": "returns", "description": "Exchanges, refunds, or damaged items"},
      {"key": "shipping", "description": "Delivery status or lost packages"},
      {"key": "billing", "description": "Charges, invoices, or payment problems"}
    ]
  },
  "target": {"returns": 1.0}
}
```

`target` may contain a soft distribution, such as annotator vote fractions. Targets must be non-negative, refer to declared option keys, and sum to one.
