from __future__ import annotations

import pytest


def test_loss_averages_questions_within_each_bundle_first() -> None:
    torch = pytest.importorskip("torch")
    from qwenrlcd.losses import decision_loss

    logits = torch.tensor(
        [[[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
         [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]]
    )
    targets = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
         [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )
    num_choices = torch.tensor([[2, 0, 0], [2, 2, 2]])
    mask = torch.tensor([[True, False, False], [True, True, True]])

    def loss(row: slice, questions: slice) -> object:
        return decision_loss(
            logits[row, questions], targets[row, questions],
            num_choices[row, questions], mask[row, questions],
            ce_weight=1.0, brier_weight=0.25,
        )

    combined = loss(slice(None), slice(None))
    short = loss(slice(0, 1), slice(0, 1))
    long = loss(slice(1, 2), slice(None))
    torch.testing.assert_close(combined.total, (short.total + long.total) / 2)
    torch.testing.assert_close(combined.per_bundle_total, torch.stack(
        [short.total, long.total]
    ))
    assert combined.total > (short.total + 3 * long.total) / 4
