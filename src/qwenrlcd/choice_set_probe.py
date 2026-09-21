from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from .metrics import calibration_metrics
from .schema import DecisionBundle, Option, Question, QuestionType

COLORS = ("red", "blue", "green", "yellow")
URN_COUNTS = (
    (7, 2, 1, 0),
    (1, 5, 3, 1),
    (2, 1, 1, 6),
    (3, 4, 2, 1),
    (0, 1, 7, 2),
    (4, 0, 1, 5),
)
COLOR_SUBSETS = (("red", "blue"), ("red", "green"), ("blue", "yellow"))


def _color_question(
    question_id: str, counts: dict[str, int], included: Sequence[str], *,
    remainder: str | None,
) -> Question:
    options = [Option(color, f"{color} marble") for color in included]
    target = {color: counts[color] / sum(counts.values()) for color in included}
    if remainder is not None:
        omitted = [color for color in COLORS if color not in included]
        key = "none_of_the_above" if remainder == "none" else "other_colors"
        description = (
            "None of the listed colors"
            if remainder == "none" else " or ".join(f"{color} marble" for color in omitted)
        )
        options.append(Option(key, description))
        target[key] = sum(counts[color] for color in omitted) / sum(counts.values())
    return Question(
        id=question_id,
        type=QuestionType.CHOICE,
        instructions="What color is one marble drawn uniformly at random from this urn?",
        options=tuple(options),
        target=target,
    )


def _range_question(
    question_id: str, value: int, ranges: Sequence[tuple[str, int, int]],
) -> Question:
    options = tuple(
        [Option(key, f"Inclusive range {lower} through {upper}")
         for key, lower, upper in ranges]
        + [Option("none_of_the_above", "None of the listed ranges contains the value")]
    )
    matches = [(key, upper - lower) for key, lower, upper in ranges if lower <= value <= upper]
    winner = min(matches, key=lambda item: item[1])[0] if matches else "none_of_the_above"
    return Question(
        id=question_id,
        type=QuestionType.CHOICE,
        instructions="Which offered range is the narrowest one containing the value?",
        options=options,
        target={winner: 1.0},
    )


def build_choice_set_probe() -> list[DecisionBundle]:
    """Small, fixed development probe; never used as training or sealed-test data."""
    bundles = []
    for index, amounts in enumerate(URN_COUNTS):
        counts = dict(zip(COLORS, amounts, strict=True))
        questions = [_color_question("full", counts, COLORS, remainder=None)]
        for subset_index, included in enumerate(COLOR_SUBSETS):
            questions.extend((
                _color_question(f"subset{subset_index}_none", counts, included, remainder="none"),
                _color_question(
                    f"subset{subset_index}_explicit", counts, included, remainder="explicit"
                ),
            ))
        bundles.append(DecisionBundle(
            id=f"choice-probe-urn-{index}",
            state={"urn_marble_counts": counts},
            questions=tuple(questions),
            source="choice_probe_urn",
        ))

    for value in (3, 7):
        narrow = ("narrow", value - 1, value + 1)
        broad = ("broad", 0, 10)
        distractor = ("outside", 11, 20)
        questions = (
            _range_question("overlap_broad_only", value, (broad, distractor)),
            _range_question("overlap_narrow_added", value, (broad, narrow, distractor)),
            _range_question("overlap_none", value, (("low", 0, 2), ("high", 9, 10))),
        )
        bundles.append(DecisionBundle(
            id=f"choice-probe-overlap-{value}",
            state={"value": value},
            questions=questions,
            source="choice_probe_overlap",
        ))
    return bundles


def summarize_choice_set_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("choice-set probe needs predictions")
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            len(row["options"]) != len(row["probabilities"])
            or len(row["options"]) != len(row["target"])
        ):
            raise ValueError("choice-set prediction, target, and options must align")
        question_id = row["question_id"]
        if question_id == "full":
            variant = "urn_full"
        elif question_id.endswith("_none") and question_id.startswith("subset"):
            variant = "urn_none_of_above"
        elif question_id.endswith("_explicit"):
            variant = "urn_explicit_complement"
        elif question_id.startswith("overlap_"):
            variant = question_id
        else:
            raise ValueError(f"unrecognized choice-set probe question: {question_id}")
        by_variant[variant].append(row)

    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        predictions = [row["probabilities"] for row in group]
        targets = [row["target"] for row in group]
        uniform = [[1 / len(target)] * len(target) for target in targets]
        return {
            "count": len(group),
            "model": asdict(calibration_metrics(predictions, targets)),
            "uniform": asdict(calibration_metrics(uniform, targets)),
        }

    pairs = []
    indexed = {(row["bundle_id"], row["question_id"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate probe question identity")
    for (bundle_id, question_id), none_row in indexed.items():
        if not question_id.startswith("subset") or not question_id.endswith("_none"):
            continue
        explicit = indexed[(bundle_id, question_id.removesuffix("_none") + "_explicit")]
        none_index = next(
            i for i, option in enumerate(none_row["options"])
            if option["key"] == "none_of_the_above"
        )
        explicit_index = next(
            i for i, option in enumerate(explicit["options"])
            if option["key"] == "other_colors"
        )
        target = none_row["target"][none_index]
        if abs(target - explicit["target"][explicit_index]) > 1e-8:
            raise ValueError("none and explicit-complement targets differ")
        pairs.append({
            "bundle_id": bundle_id,
            "subset": question_id.removesuffix("_none"),
            "target_remainder": target,
            "predicted_none": none_row["probabilities"][none_index],
            "predicted_explicit": explicit["probabilities"][explicit_index],
        })

    return {
        "note": (
            "Exploratory, hand-built development probe—not a held-out dataset result. "
            "The explicit-complement variant lists omitted colors; the none-of-above "
            "variant requires interpreting the offered set."
        ),
        "overall": metrics(rows),
        "by_variant": {key: metrics(group) for key, group in sorted(by_variant.items())},
        "remainder_pairs": pairs,
        "mean_none_absolute_error": sum(
            abs(pair["predicted_none"] - pair["target_remainder"]) for pair in pairs
        ) / len(pairs),
        "mean_explicit_absolute_error": sum(
            abs(pair["predicted_explicit"] - pair["target_remainder"]) for pair in pairs
        ) / len(pairs),
        "predictions": rows,
    }
