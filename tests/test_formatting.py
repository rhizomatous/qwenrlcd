from __future__ import annotations

from qwenrlcd.formatting import (
    DECISION_MARKER,
    render_bundle,
    render_option,
    render_question_prefix,
)
from qwenrlcd.schema import DecisionBundle

from .test_schema import example_dict


def test_render_bundle_contains_state_and_every_decision_slot() -> None:
    bundle = DecisionBundle.from_dict(example_dict())
    rendered = render_bundle(bundle)

    assert rendered.count("<|decision_state|>") == 1
    assert rendered.count(DECISION_MARKER) == 7
    assert "key: returns\ndescription: Damaged items" in rendered
    assert "private_route_key" not in rendered
    assert '"target"' not in rendered


def test_question_id_is_not_model_input() -> None:
    value = example_dict()
    value["questions"]["private_route_key"] = value["questions"].pop("route")
    question = DecisionBundle.from_dict(value).questions[0]

    assert "private_route_key" not in render_question_prefix(question)


def test_noul_renders_implicit_keys_and_optional_descriptions() -> None:
    value = example_dict()
    bare = DecisionBundle.from_dict(value).questions[1]
    assert render_option(bare.options[1]) == (
        "<|decision_option|>\nkey: true\n<|decision|>"
    )

    value["questions"]["urgent"]["criteria"] = {
        "false": "No immediate action is needed",
        "true": "Immediate action is needed",
    }
    explicit = DecisionBundle.from_dict(value).questions[1]
    rendered = render_option(explicit.options[1])
    assert "key: true\ndescription: Immediate action is needed" in rendered
