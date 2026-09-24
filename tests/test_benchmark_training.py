from __future__ import annotations

import pytest

from qwenrlcd.benchmark_training import (
    evenly_spaced_names,
    quantile_rank,
    reused_length_cases,
    select_length_cases,
)


def test_quantile_rank_uses_nearest_observed_rank() -> None:
    assert quantile_rank(11, 0.0) == 0
    assert quantile_rank(11, 0.5) == 5
    assert quantile_rank(11, 0.95) == 10
    assert quantile_rank(11, 1.0) == 10
    with pytest.raises(ValueError, match="positive"):
        quantile_rank(0, 0.5)
    with pytest.raises(ValueError, match="between"):
        quantile_rank(10, 1.1)


def test_select_length_cases_uses_distinct_rows_and_reserves_maximum() -> None:
    lengths = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    cases = select_length_cases(lengths)
    assert [case["name"] for case in cases] == ["median", "p95", "maximum"]
    assert cases[0]["target_length"] == 50
    assert [lengths[index] for index in cases[0]["indices"]] == [50, 40]
    assert cases[1]["target_length"] == 100
    assert [lengths[index] for index in cases[1]["indices"]] == [90, 80]
    assert cases[2]["indices"] == [9]
    selected = [index for case in cases for index in case["indices"]]
    assert len(selected) == len(set(selected)) == 5


def test_evenly_spaced_names_includes_endpoints() -> None:
    assert evenly_spaced_names(["d", "a", "c", "b"], 3) == ["a", "c", "d"]
    assert evenly_spaced_names(["b", "a"], 8) == ["a", "b"]
    with pytest.raises(ValueError, match="positive"):
        evenly_spaced_names(["a"], 0)


def test_reused_length_cases_validates_saved_indices() -> None:
    report = {
        "selection": [
            {
                "name": "median",
                "quantile": 0.5,
                "target_length": 50,
                "indices": [4, 3],
                "ids": ["median-a", "median-b"],
                "individual_lengths": [50, 40],
            },
            {
                "name": "p95",
                "quantile": 0.95,
                "target_length": 100,
                "indices": [8, 7],
            },
            {
                "name": "maximum",
                "quantile": 1.0,
                "target_length": 100,
                "indices": [9],
            },
        ]
    }
    cases = reused_length_cases(report, 10)
    assert [case["indices"] for case in cases] == [[4, 3], [8, 7], [9]]
    assert cases[0]["expected_ids"] == ["median-a", "median-b"]
    assert cases[0]["expected_lengths"] == [50, 40]

    report["selection"][1]["indices"] = [9, 7]
    with pytest.raises(ValueError, match="reuses"):
        reused_length_cases(report, 10)
