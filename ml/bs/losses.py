"""Losses for imbalanced four-class Burn Severity segmentation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ml._shared import IGNORE_INDEX


class WeightedCrossEntropyDiceLoss(nn.Module):
    """Equal mixture of weighted CE and foreground multiclass Dice loss."""

    def __init__(self, class_weights: torch.Tensor, dice_weight: float = 0.5):
        super().__init__()
        if not 0 <= dice_weight <= 1:
            raise ValueError("dice_weight must be between zero and one")
        self.register_buffer("class_weights", class_weights.float())
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=IGNORE_INDEX)
        valid = target != IGNORE_INDEX
        safe_target = target.masked_fill(~valid, 0)
        probabilities = torch.softmax(logits, dim=1)
        encoded = F.one_hot(safe_target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        valid_channels = valid[:, None].float()
        probabilities = probabilities * valid_channels
        encoded = encoded * valid_channels
        # Foreground-only Dice prevents the plentiful background from dominating.
        intersection = (probabilities[:, 1:] * encoded[:, 1:]).sum(dim=(0, 2, 3))
        denominator = probabilities[:, 1:].sum(dim=(0, 2, 3)) + encoded[:, 1:].sum(dim=(0, 2, 3))
        dice_loss = 1 - ((2 * intersection + 1.0) / (denominator + 1.0)).mean()
        return (1 - self.dice_weight) * cross_entropy + self.dice_weight * dice_loss
