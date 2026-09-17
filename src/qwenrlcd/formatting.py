from __future__ import annotations

import json
from typing import Any

from .schema import DecisionBundle, Question

DECISION_MARKER = "<|decision|>"


def render_jsonlike(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_state(bundle: DecisionBundle) -> str:
    return "\n".join(("<|decision_state|>", render_jsonlike(bundle.state)))


def render_question(question: Question) -> str:
    """Render one branch without its routing id or training target."""
    lines = [
        "<|decision_question|>",
        f"type: {question.type.value}",
        f"instructions: {render_jsonlike(question.instructions)}",
        "<|decision_criteria|>",
    ]
    for index, option in enumerate(question.options):
        description = render_jsonlike(option.description)
        lines.append(f"[{index}] {option.key}: {description}")
    lines.append(DECISION_MARKER)
    return "\n".join(lines)


def render_bundle(bundle: DecisionBundle) -> str:
    """Human-readable packed representation; model attention is supplied separately."""
    return "\n".join([render_state(bundle), *(render_question(q) for q in bundle.questions)])
