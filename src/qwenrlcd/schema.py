from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

MAX_CHOICES = 255
MAX_QUESTIONS = 64


class QuestionType(StrEnum):
    CHOICE = "choice"
    NOUL = "noul"
    SCORE = "score"


def _require_jsonlike(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc


@dataclass(frozen=True, slots=True)
class Option:
    key: str
    description: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("option key must be a non-empty string")
        _require_jsonlike(self.description, "option description")


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    type: QuestionType
    instructions: Any
    options: tuple[Option, ...]
    target: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("question id must be a non-empty string")
        _require_jsonlike(self.instructions, "question instructions")

        if not 2 <= len(self.options) <= MAX_CHOICES:
            raise ValueError(f"questions require 2 to {MAX_CHOICES} options")
        keys = [option.key for option in self.options]
        if len(keys) != len(set(keys)):
            raise ValueError("option keys must be unique within a question")
        if self.type is QuestionType.NOUL and keys != ["false", "true"]:
            raise ValueError("noul options must be ordered as false, true")
        if self.type is QuestionType.SCORE and keys != [str(i) for i in range(len(keys))]:
            raise ValueError("score option keys must be consecutive levels starting at 0")

        if self.target is None:
            return

        unknown = set(self.target) - set(keys)
        if unknown:
            raise ValueError(f"target refers to unknown options: {sorted(unknown)}")
        probabilities = list(self.target.values())
        if not probabilities:
            raise ValueError("target distribution cannot be empty")
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise ValueError("target probabilities must be finite and non-negative")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("target probabilities must sum to one")

    @classmethod
    def from_entry(cls, question_id: str, value: Mapping[str, Any]) -> Question:
        question_type = QuestionType(value["type"])
        criteria = value.get("criteria")

        if question_type is QuestionType.CHOICE:
            if not isinstance(criteria, Mapping):
                raise ValueError("choice criteria must be an option-to-description map")
            options = tuple(Option(str(key), description) for key, description in criteria.items())
        elif question_type is QuestionType.SCORE:
            if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
                raise ValueError("score criteria must be an ordered list of levels")
            options = tuple(
                Option(str(index), description)
                for index, description in enumerate(criteria)
            )
        else:
            if criteria is not None and not isinstance(criteria, Mapping):
                raise ValueError("noul criteria must be an optional true/false map")
            criteria = criteria or {}
            options = (
                Option("false", criteria.get("false")),
                Option("true", criteria.get("true")),
            )

        raw_target = value.get("target")
        if raw_target is not None and not isinstance(raw_target, Mapping):
            raise ValueError("target must be an option-to-probability map")
        target = (
            {
                str(key): float(probability)
                for key, probability in raw_target.items()
            }
            if raw_target is not None
            else None
        )
        return cls(
            id=question_id,
            type=question_type,
            instructions=value["instructions"],
            options=options,
            target=target,
        )

    def to_entry(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.type.value,
            "instructions": self.instructions,
        }
        if self.type is QuestionType.CHOICE:
            value["criteria"] = {option.key: option.description for option in self.options}
        elif self.type is QuestionType.SCORE:
            value["criteria"] = [option.description for option in self.options]
        else:
            if any(option.description is not None for option in self.options):
                value["criteria"] = {
                    option.key: option.description for option in self.options
                }
        if self.target is not None:
            value["target"] = dict(self.target)
        return value

    def target_vector(self) -> list[float]:
        if self.target is None:
            raise ValueError(f"question {self.id} has no training target")
        return [float(self.target.get(option.key, 0.0)) for option in self.options]

    def expected_score(self) -> float:
        if self.type is not QuestionType.SCORE:
            raise ValueError("expected_score is only defined for score questions")
        return sum(index * probability for index, probability in enumerate(self.target_vector()))

    def permuted_options(self, rng: random.Random) -> Question:
        if self.type is not QuestionType.CHOICE:
            return self
        options = list(self.options)
        rng.shuffle(options)
        return Question(self.id, self.type, self.instructions, tuple(options), self.target)


@dataclass(frozen=True, slots=True)
class DecisionBundle:
    id: str
    state: Any
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("bundle id must be a non-empty string")
        _require_jsonlike(self.state, "state")
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ValueError(f"bundles require 1 to {MAX_QUESTIONS} questions")
        question_ids = [question.id for question in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("question ids must be unique within a bundle")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionBundle:
        questions = value.get("questions")
        if not isinstance(questions, Mapping):
            raise ValueError("questions must be a map keyed by question id")
        return cls(
            id=value["id"],
            state=value["state"],
            questions=tuple(
                Question.from_entry(str(question_id), question)
                for question_id, question in questions.items()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "questions": {question.id: question.to_entry() for question in self.questions},
        }

    def permuted(self, rng: random.Random) -> DecisionBundle:
        questions = [question.permuted_options(rng) for question in self.questions]
        rng.shuffle(questions)
        return DecisionBundle(self.id, self.state, tuple(questions))
