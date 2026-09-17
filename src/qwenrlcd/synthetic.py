from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from .data import write_jsonl
from .schema import DecisionBundle

DEPARTMENTS = {
    "returns": "Exchanges, refunds, wrong sizes, or damaged items",
    "shipping": "Late, missing, or misdirected deliveries",
    "billing": "Charges, invoices, cards, or payment failures",
}

TICKET_TEMPLATES = {
    "returns": [
        "The {item} arrived in the wrong size and I need to exchange it.",
        "My {item} is damaged. How can I get a replacement?",
        "I changed my mind about the {item} and would like a refund.",
    ],
    "shipping": [
        "Tracking says delivered, but the {item} is not here.",
        "The {item} was due {days} days ago and has not arrived.",
        "My {item} was sent to the wrong address.",
    ],
    "billing": [
        "I was charged twice for the {item}.",
        "My card payment for the {item} keeps failing.",
        "The invoice for the {item} has the wrong amount.",
    ],
}

ITEMS = ["running shoes", "coffee grinder", "winter coat", "desk lamp", "headphones"]


def _categorical_target(answer: str, keys: list[str], confidence: float = 0.9) -> dict[str, float]:
    remainder = (1.0 - confidence) / (len(keys) - 1)
    return {key: confidence if key == answer else remainder for key in keys}


def _noul_target(probability_true: float) -> dict[str, float]:
    return {"false": 1.0 - probability_true, "true": probability_true}


def _score_target(value: float, levels: int) -> dict[str, float]:
    value = min(max(value, 0.0), levels - 1)
    lower = math.floor(value)
    upper = math.ceil(value)
    if lower == upper:
        return {str(lower): 1.0}
    return {str(lower): upper - value, str(upper): value - lower}


def generate_bundles(count: int, seed: int) -> list[DecisionBundle]:
    """Generate small mixed-question fixtures; not a substantive training corpus."""
    rng = random.Random(seed)
    bundles: list[DecisionBundle] = []
    department_keys = list(DEPARTMENTS)
    for index in range(count):
        department = rng.choice(department_keys)
        days_waiting = rng.randint(0, 8)
        balance = rng.randint(-100, 100)
        item = rng.choice(ITEMS)
        message = rng.choice(TICKET_TEMPLATES[department]).format(
            item=item, days=days_waiting
        )
        urgent_signal = min(2.0, days_waiting / 4 + (0.5 if "keeps failing" in message else 0))
        refund_probability = 0.9 if department == "returns" else 0.05

        bundles.append(
            DecisionBundle.from_dict(
                {
                    "id": f"support-{seed}-{index:05d}",
                    "state": {
                        "message": message,
                        "account_balance_usd": balance,
                        "days_waiting": days_waiting,
                    },
                    "questions": {
                        "department": {
                            "type": "choice",
                            "instructions": "Which team should handle `message`?",
                            "criteria": DEPARTMENTS,
                            "target": _categorical_target(department, department_keys),
                        },
                        "refund_requested": {
                            "type": "noul",
                            "instructions": (
                                "Does `message` request a refund, exchange, or replacement?"
                            ),
                            "target": _noul_target(refund_probability),
                        },
                        "urgency": {
                            "type": "score",
                            "instructions": "How urgent is the request given the message and wait?",
                            "criteria": [
                                "Routine; no meaningful delay or urgency.",
                                "Time-sensitive or moderately delayed.",
                                "Urgent; severe delay or explicit business impact.",
                            ],
                            "target": _score_target(urgent_signal, 3),
                        },
                        "negative_balance": {
                            "type": "noul",
                            "instructions": "Is `account_balance_usd` negative?",
                            "target": _noul_target(float(balance < 0)),
                        },
                    },
                }
            )
        )
    return bundles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic smoke-test bundles")
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    write_jsonl(
        generate_bundles(args.train, args.seed), args.output_dir / "smoke_train.jsonl"
    )
    write_jsonl(
        generate_bundles(args.validation, args.seed + 1),
        args.output_dir / "smoke_validation.jsonl",
    )


if __name__ == "__main__":
    main()
