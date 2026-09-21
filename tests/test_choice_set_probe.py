from __future__ import annotations

import pytest

from qwenrlcd.choice_set_probe import build_choice_set_probe, summarize_choice_set_probe
from qwenrlcd.data import DecisionDataset

from .test_data import WhitespaceTokenizer


def _target_rows() -> list[dict]:
    rows = []
    for bundle in build_choice_set_probe():
        for question in bundle.questions:
            target = question.target_vector()
            rows.append({
                "source": bundle.source,
                "bundle_id": bundle.id,
                "question_id": question.id,
                "state_excerpt": str(bundle.state),
                "instructions": question.instructions,
                "options": [{"key": option.key} for option in question.options],
                "target": target,
                "probabilities": target,
            })
    return rows


def test_choice_set_probe_has_exact_labels_and_paired_complements() -> None:
    bundles = build_choice_set_probe()
    assert len(bundles) == 8
    assert sum(len(bundle.questions) for bundle in bundles) == 48
    first = bundles[0]
    assert first.questions[0].target_vector() == pytest.approx([0.7, 0.2, 0.1, 0])
    none = first.questions[1]
    explicit = first.questions[2]
    assert none.target_vector() == pytest.approx([0.7, 0.2, 0.1])
    assert explicit.target_vector() == pytest.approx(none.target_vector())
    assert none.options[-1].key == "none_of_the_above"
    assert explicit.options[-1].key == "other_colors"
    overlap = bundles[-2]
    assert [max(q.target, key=q.target.get) for q in overlap.questions] == [
        "broad", "narrow", "none_of_the_above"
    ]


def test_choice_set_probe_report_pairs_same_target_variants() -> None:
    report = summarize_choice_set_probe(_target_rows())
    assert report["overall"]["count"] == 48
    assert report["by_variant"]["urn_none_of_above"]["count"] == 18
    assert report["by_variant"]["urn_explicit_complement"]["count"] == 18
    assert len(report["remainder_pairs"]) == 18
    assert report["mean_none_absolute_error"] == pytest.approx(0)
    assert report["mean_explicit_absolute_error"] == pytest.approx(0)
    assert report["overall"]["model"]["brier"] == pytest.approx(0)


def test_choice_set_probe_packs_with_training_limits() -> None:
    bundles = build_choice_set_probe()
    dataset = DecisionDataset(
        bundles, WhitespaceTokenizer(), max_length=4096, max_choices=255,
        max_questions=32, shuffle=False, seed=17, require_no_state_truncation=True,
    )
    assert len(dataset) == len(bundles)
    assert all(len(dataset[index]["question_types"]) <= 32 for index in range(len(dataset)))
