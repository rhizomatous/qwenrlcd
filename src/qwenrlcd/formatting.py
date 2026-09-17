from __future__ import annotations

import json
from typing import Any

from .schema import DecisionRecord


def render_jsonlike(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_record(record: DecisionRecord) -> str:
    """Serialize a record without leaking its training target into the prompt."""
    lines = [
        "<|decision_state|>",
        render_jsonlike(record.state),
        "<|decision_question|>",
        f"type: {record.question.type.value}",
        f"instructions: {render_jsonlike(record.question.instructions)}",
        "<|decision_options|>",
    ]
    for index, option in enumerate(record.question.options):
        description = render_jsonlike(option.description) if option.description is not None else ""
        lines.append(f"[{index}] {option.key}: {description}")
    lines.append("<|decision|>")
    return "\n".join(lines)
