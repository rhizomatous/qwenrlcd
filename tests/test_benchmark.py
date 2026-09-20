from __future__ import annotations

import pytest

from qwenrlcd.benchmark import (
    benchmark_bundles,
    benchmark_state,
    max_probability_delta,
    parse_positive_ints,
    probabilities,
    timing_summary,
)
from qwenrlcd.data import DecisionDataset

from .test_data import WhitespaceTokenizer


def test_benchmark_bundles_share_state_and_questions() -> None:
    bundled, singletons = benchmark_bundles("same state", 28)
    assert len(bundled.questions) == len(singletons) == 28
    assert all(singleton.state == bundled.state for singleton in singletons)
    assert [singleton.questions[0] for singleton in singletons] == list(bundled.questions)
    assert len({question.id for question in bundled.questions}) == 28
    with pytest.raises(ValueError, match="question_count"):
        benchmark_bundles("same state", 29)


def test_benchmark_state_reaches_target_length() -> None:
    tokenizer = WhitespaceTokenizer()
    state, count = benchmark_state(tokenizer, 100)
    assert state
    assert count >= 100


def test_benchmark_bundled_and_singleton_inputs_pack_without_targets() -> None:
    bundled, singletons = benchmark_bundles("same state", 4)
    dataset = DecisionDataset(
        [bundled, *singletons], WhitespaceTokenizer(),
        max_length=512, max_choices=255, max_questions=32,
        shuffle=False, seed=17, require_targets=False,
        require_no_state_truncation=True,
    )
    assert len(dataset[0]["decision_indices"]) == 4
    assert all(len(dataset[index]["decision_indices"]) == 1 for index in range(1, 5))


def test_benchmark_summaries() -> None:
    assert parse_positive_ints("1,4,28") == (1, 4, 28)
    with pytest.raises(ValueError, match="distinct positive"):
        parse_positive_ints("1,1")
    assert timing_summary([4.0, 1.0, 3.0, 2.0, 5.0]) == {
        "p50_ms": 3.0, "p95_ms": 5.0, "min_ms": 1.0, "max_ms": 5.0,
    }
    assert probabilities([[0.0, 0.0]]) == [[0.5, 0.5]]
    assert max_probability_delta([[0.25, 0.75]], [[0.5, 0.5]]) == 0.25
