from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .schema import DecisionBundle, Question, QuestionType


def format_question_prediction(
    question: Question, probabilities: Sequence[float]
) -> dict[str, Any]:
    if len(probabilities) != len(question.options):
        raise ValueError(
            f"question {question.id} expected {len(question.options)} probabilities, "
            f"got {len(probabilities)}"
        )

    values = [float(probability) for probability in probabilities]
    distribution = {
        option.key: probability
        for option, probability in zip(question.options, values, strict=True)
    }
    best_index = max(range(len(values)), key=values.__getitem__)

    if question.type is QuestionType.CHOICE:
        return {
            "type": "choice",
            "choice": question.options[best_index].key,
            "confidence": values[best_index],
            "probabilities": distribution,
        }
    if question.type is QuestionType.SCORE:
        return {
            "type": "score",
            "score": sum(index * probability for index, probability in enumerate(values)),
            "confidence": values[best_index],
            "probabilities": distribution,
            "legend": {
                option.key: option.description for option in question.options
            },
        }
    return {
        "type": "noul",
        "noul": distribution["true"],
        "probabilities": distribution,
    }


def format_bundle_prediction(
    bundle: DecisionBundle, probability_rows: Sequence[Sequence[float]]
) -> dict[str, Any]:
    if len(probability_rows) != len(bundle.questions):
        raise ValueError(
            f"bundle {bundle.id} expected {len(bundle.questions)} probability rows, "
            f"got {len(probability_rows)}"
        )
    return {
        "id": bundle.id,
        "questions": {
            question.id: format_question_prediction(question, probabilities)
            for question, probabilities in zip(
                bundle.questions, probability_rows, strict=True
            )
        },
    }
