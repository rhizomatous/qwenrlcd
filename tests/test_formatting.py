from __future__ import annotations

from qwenrlcd.formatting import render_record
from qwenrlcd.schema import DecisionRecord


def test_render_record_contains_schema_but_not_target() -> None:
    record = DecisionRecord.from_dict(
        {
            "id": "ticket",
            "state": {"message": "wrong size"},
            "question": {
                "type": "choice",
                "instructions": "Pick a team",
                "options": [
                    {"key": "returns", "description": "exchanges"},
                    {"key": "billing", "description": "payments"},
                ],
            },
            "target": {"returns": 1.0},
        }
    )
    rendered = render_record(record)
    assert "<|decision_state|>" in rendered
    assert "[0] returns: exchanges" in rendered
    assert "<|decision|>" in rendered
    assert '"target"' not in rendered
