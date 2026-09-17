from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class DecisionLoss:
    total: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor
    probabilities: torch.Tensor


def mask_invalid_choices(logits: torch.Tensor, num_choices: torch.Tensor) -> torch.Tensor:
    slots = torch.arange(logits.shape[-1], device=logits.device).view(1, 1, -1)
    invalid = slots >= num_choices.unsqueeze(-1)
    return logits.masked_fill(invalid, torch.finfo(logits.dtype).min)


def decision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_choices: torch.Tensor,
    question_mask: torch.Tensor,
    *,
    ce_weight: float,
    brier_weight: float,
) -> DecisionLoss:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, questions, choices]")
    masked_logits = mask_invalid_choices(logits.float(), num_choices)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    probabilities = log_probabilities.exp()

    valid = question_mask.float()
    denominator = valid.sum().clamp_min(1.0)
    per_question_ce = -(targets * log_probabilities).sum(dim=-1)
    per_question_brier = ((probabilities - targets) ** 2).sum(dim=-1)
    cross_entropy = (per_question_ce * valid).sum() / denominator
    brier = (per_question_brier * valid).sum() / denominator
    total = ce_weight * cross_entropy + brier_weight * brier
    return DecisionLoss(total, cross_entropy, brier, probabilities)
