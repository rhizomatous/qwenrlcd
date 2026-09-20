from __future__ import annotations

import json
from typing import Any

from .schema import DecisionBundle, Option, Question

DECISION_MARKER = "<|decision|>"


def render_jsonlike(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_state(bundle: DecisionBundle) -> str:
    return "\n".join(("<|decision_state|>", render_jsonlike(bundle.state)))


def render_question_prefix(question: Question) -> str:
    """Common prefix shared by isolated option branches of one question."""
    return "\n".join((
        "<|decision_question|>",
        f"type: {question.type.value}",
        f"instructions: {render_jsonlike(question.instructions)}",
    ))


def render_option(option: Option) -> str:
    """Index-free option branch, ending at its own decision marker."""
    lines = ["<|decision_option|>", f"key: {option.key}"]
    if option.description is not None:
        lines.append(f"description: {render_jsonlike(option.description)}")
    lines.append(DECISION_MARKER)
    return "\n".join(lines)


def render_bundle(bundle: DecisionBundle) -> str:
    """Human-readable packed representation; model attention is supplied separately."""
    parts = [render_state(bundle)]
    for question in bundle.questions:
        parts.append(render_question_prefix(question))
        parts.extend(render_option(option) for option in question.options)
    return "\n".join(parts)
