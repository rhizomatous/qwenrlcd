from __future__ import annotations

from qwenrlcd.formatting import DECISION_MARKER, render_bundle, render_question
from qwenrlcd.schema import DecisionBundle

from .test_schema import example_dict


def test_render_bundle_contains_state_and_every_decision_slot() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    rendered = render_bundle(bundle)

    assert rendered.count("<|decision_state|>") == 1
    assert rendered.count(DECISION_MARKER) == 3
    assert "[0] returns: Damaged items" in rendered
    assert "private_route_key" not in rendered
    assert '"target"' not in rendered


def test_question_id_is_not_model_input() -> None:
    value = example_dict()
    value["questions"]["private_route_key"] = value["questions"].pop("route")
    question = DecisionBundle.from_dict(value).questions[0]

    assert "private_route_key" not in render_question(question)
