from __future__ import annotations

import argparse
import random
from pathlib import Path

from .data import write_jsonl
from .schema import DecisionRecord, Option, Question, QuestionType


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
        "The {item} was due four days ago and has not arrived.",
        "My {item} was sent to the wrong address.",
    ],
    "billing": [
        "I was charged twice for the {item}.",
        "My card payment for the {item} keeps failing.",
        "The invoice for the {item} has the wrong amount.",
    ],
}

ITEMS = ["running shoes", "coffee grinder", "winter coat", "desk lamp", "headphones"]


def generate_records(count: int, seed: int) -> list[DecisionRecord]:
    rng = random.Random(seed)
    records: list[DecisionRecord] = []
    for index in range(count):
        if index % 3:
            department = rng.choice(list(DEPARTMENTS))
            state = rng.choice(TICKET_TEMPLATES[department]).format(item=rng.choice(ITEMS))
            records.append(
                DecisionRecord(
                    id=f"ticket-{seed}-{index:05d}",
                    state=state,
                    question=Question(
                        type=QuestionType.CHOICE,
                        instructions="Which team should handle this customer request?",
                        options=tuple(
                            Option(key=key, description=description)
                            for key, description in DEPARTMENTS.items()
                        ),
                    ),
                    target={department: 1.0},
                )
            )
        else:
            balance = rng.randint(-100, 100)
            records.append(
                DecisionRecord(
                    id=f"balance-{seed}-{index:05d}",
                    state={"account_balance_usd": balance},
                    question=Question(
                        type=QuestionType.NOUL,
                        instructions="The account balance is negative.",
                        options=(
                            Option("false", "The statement is false"),
                            Option("true", "The statement is true"),
                        ),
                    ),
                    target={"true" if balance < 0 else "false": 1.0},
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic smoke-test data")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    write_jsonl(generate_records(args.train, args.seed), args.output_dir / "smoke_train.jsonl")
    write_jsonl(
        generate_records(args.validation, args.seed + 1),
        args.output_dir / "smoke_validation.jsonl",
    )


if __name__ == "__main__":
    main()
