from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

MAX_CHOICES = 255
MAX_SCORE_LEVELS = 10


class QuestionType(StrEnum):
    CHOICE = "choice"
    NOUL = "noul"
    SCORE = "score"


@dataclass(frozen=True, slots=True)
class Option:
    key: str
    description: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("option key must be a non-empty string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Option:
        return cls(key=value["key"], description=value.get("description"))

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "description": self.description}


@dataclass(frozen=True, slots=True)
class Question:
    type: QuestionType
    instructions: Any
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, (str, Mapping, Sequence)):
            raise ValueError("question instructions must be JSON-like text or structure")

        option_count = len(self.options)
        if not 2 <= option_count <= MAX_CHOICES:
            raise ValueError(f"questions require 2 to {MAX_CHOICES} options")

        keys = [option.key for option in self.options]
        if len(keys) != len(set(keys)):
            raise ValueError("option keys must be unique")

        if self.type is QuestionType.NOUL and option_count != 2:
            raise ValueError("noul questions require exactly two options")
        if self.type is QuestionType.SCORE and option_count > MAX_SCORE_LEVELS:
            raise ValueError(f"score questions support at most {MAX_SCORE_LEVELS} levels")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Question:
        return cls(
            type=QuestionType(value["type"]),
            instructions=value["instructions"],
            options=tuple(Option.from_dict(option) for option in value["options"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "instructions": self.instructions,
            "options": [option.to_dict() for option in self.options],
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    id: str
    state: Any
    question: Question
    target: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("record id must be a non-empty string")

        option_keys = {option.key for option in self.question.options}
        target_keys = set(self.target)
        unknown = target_keys - option_keys
        if unknown:
            raise ValueError(f"target refers to unknown options: {sorted(unknown)}")

        values = list(self.target.values())
        if not values:
            raise ValueError("target distribution cannot be empty")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("target probabilities must be finite and non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("target probabilities must sum to one")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DecisionRecord:
        return cls(
            id=value["id"],
            state=value["state"],
            question=Question.from_dict(value["question"]),
            target={key: float(probability) for key, probability in value["target"].items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "question": self.question.to_dict(),
            "target": dict(self.target),
        }

    def target_vector(self) -> list[float]:
        return [float(self.target.get(option.key, 0.0)) for option in self.question.options]

    def answer_index(self) -> int:
        vector = self.target_vector()
        return max(range(len(vector)), key=vector.__getitem__)

    def permuted(self, rng: random.Random) -> DecisionRecord:
        options = list(self.question.options)
        rng.shuffle(options)
        return DecisionRecord(
            id=self.id,
            state=self.state,
            question=Question(
                type=self.question.type,
                instructions=self.question.instructions,
                options=tuple(options),
            ),
            target=self.target,
        )
