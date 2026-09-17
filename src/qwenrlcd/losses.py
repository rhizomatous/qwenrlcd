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
    slots = torch.arange(logits.shape[-1], device=logits.device).unsqueeze(0)
    invalid = slots >= num_choices.unsqueeze(1)
    return logits.masked_fill(invalid, torch.finfo(logits.dtype).min)


def decision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_choices: torch.Tensor,
    *,
    ce_weight: float,
    brier_weight: float,
) -> DecisionLoss:
    masked_logits = mask_invalid_choices(logits.float(), num_choices)
    log_probabilities = F.log_softmax(masked_logits, dim=-1)
    probabilities = log_probabilities.exp()

    cross_entropy = -(targets * log_probabilities).sum(dim=-1).mean()
    brier = ((probabilities - targets) ** 2).sum(dim=-1).mean()
    total = ce_weight * cross_entropy + brier_weight * brier
    return DecisionLoss(total, cross_entropy, brier, probabilities)
