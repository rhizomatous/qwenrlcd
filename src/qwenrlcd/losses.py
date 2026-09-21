from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DecisionLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor
    ordinal_rps: torch.Tensor
    probabilities: torch.Tensor
    per_bundle_total: torch.Tensor


def mask_invalid_choices(logits: torch.Tensor, num_choices: torch.Tensor) -> torch.Tensor:
    slots = torch.arange(logits.shape[-1], device=logits.device).view(1, 1, -1)
    invalid = slots >= num_choices.unsqueeze(-1)
    return logits.masked_fill(invalid, torch.finfo(logits.dtype).min)


def decision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_choices: torch.Tensor,
    question_mask: torch.Tensor,
    score_mask: torch.Tensor,
    *,
    ce_weight: float,
    brier_weight: float,
    ordinal_rps_weight: float,
) -> DecisionLoss:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, questions, choices]")
    masked_logits = mask_invalid_choices(logits.float(), num_choices)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    probabilities = log_probabilities.exp()

    valid = question_mask.float()
    if score_mask.shape != question_mask.shape:
        raise ValueError("score_mask and question_mask must have the same shape")
    if torch.any(score_mask & ~question_mask):
        raise ValueError("score_mask cannot select padded questions")
    questions_per_bundle = valid.sum(dim=-1)
    if torch.any(questions_per_bundle == 0):
        raise ValueError("every bundle must contain at least one valid question")
    per_question_ce = -(targets * log_probabilities).sum(dim=-1)
    per_question_brier = ((probabilities - targets) ** 2).sum(dim=-1)
    slots = torch.arange(logits.shape[-1], device=logits.device).view(1, 1, -1)
    ordinal_boundaries = slots < (num_choices - 1).clamp_min(0).unsqueeze(-1)
    cumulative_error = (probabilities.cumsum(dim=-1) - targets.cumsum(dim=-1)) ** 2
    per_question_rps = (
        (cumulative_error * ordinal_boundaries).sum(dim=-1)
        / (num_choices - 1).clamp_min(1)
    )
    per_bundle_ce = (per_question_ce * valid).sum(dim=-1) / questions_per_bundle
    per_bundle_brier = (per_question_brier * valid).sum(dim=-1) / questions_per_bundle
    per_bundle_rps = (
        per_question_rps * score_mask.float()
    ).sum(dim=-1) / questions_per_bundle
    per_bundle_total = (
        ce_weight * per_bundle_ce
        + brier_weight * per_bundle_brier
        + ordinal_rps_weight * per_bundle_rps
    )
    return DecisionLoss(
        per_bundle_total.mean(),
        per_bundle_ce.mean(),
        per_bundle_brier.mean(),
        per_bundle_rps.mean(),
        probabilities,
        per_bundle_total,
    )
