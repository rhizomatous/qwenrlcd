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
            torch.zeros_like(mask[row, questions]),
            ce_weight=1.0, brier_weight=0.25, ordinal_rps_weight=0.0,
        )

    combined = loss(slice(None), slice(None))
    short = loss(slice(0, 1), slice(0, 1))
    long = loss(slice(1, 2), slice(None))
    torch.testing.assert_close(combined.total, (short.total + long.total) / 2)
    torch.testing.assert_close(combined.per_bundle_total, torch.stack(
        [short.total, long.total]
    ))
    assert combined.total > (short.total + 3 * long.total) / 4


def test_score_rps_penalizes_distant_ordinal_misses_more() -> None:
    torch = pytest.importorskip("torch")
    from qwenrlcd.losses import decision_loss

    logits = torch.tensor([[[0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]])
    targets = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    num_choices = torch.tensor([[3, 3]])
    question_mask = torch.tensor([[True, True]])
    score_mask = torch.tensor([[True, True]])

    losses = decision_loss(
        logits, targets, num_choices, question_mask, score_mask,
        ce_weight=0.0, brier_weight=0.0, ordinal_rps_weight=2.0,
    )

    # A 0→1 miss has RPS 0.5 and a 0→2 miss has RPS 1.0.
    assert losses.ordinal_rps.item() == pytest.approx(0.75)
    assert losses.total.item() == pytest.approx(1.5)


def test_score_rps_ignores_non_score_questions_and_rejects_padding() -> None:
    torch = pytest.importorskip("torch")
    from qwenrlcd.losses import decision_loss

    logits = torch.tensor([[[0.0, 0.0, 20.0], [0.0, 0.0, 0.0]]])
    targets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    num_choices = torch.tensor([[3, 0]])
    question_mask = torch.tensor([[True, False]])
    losses = decision_loss(
        logits, targets, num_choices, question_mask,
        torch.tensor([[False, False]]),
        ce_weight=0.0, brier_weight=0.0, ordinal_rps_weight=1.0,
    )
    assert losses.ordinal_rps.item() == pytest.approx(0)
    assert losses.total.item() == pytest.approx(0)

    with pytest.raises(ValueError, match="padded"):
        decision_loss(
            logits, targets, num_choices, question_mask,
            torch.tensor([[False, True]]),
            ce_weight=0.0, brier_weight=0.0, ordinal_rps_weight=1.0,
        )
