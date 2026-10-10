"""Loss functions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on logits (Lin et al., 2017): alpha * (1 - p_t)^gamma * BCE."""

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(name: str = "focal", **kwargs) -> nn.Module:
    name = name.lower()
    if name == "focal":
        return FocalLoss(**kwargs)
    if name == "bce":
        return nn.BCEWithLogitsLoss(**kwargs)
    raise ValueError(f"Unknown loss: {name}")
