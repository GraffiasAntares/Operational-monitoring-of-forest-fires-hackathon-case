"""Compact GroupNorm U-Net sized for 512px BS chips and modest GPUs."""
from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int, preferred: int = 8) -> int:
    return next(groups for groups in range(min(preferred, channels), 0, -1) if channels % groups == 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout else nn.Identity(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class CompactUNet(nn.Module):
    """Four-stage U-Net with 4 logits per pixel and no pretrained encoder."""

    def __init__(self, in_channels: int, base_channels: int = 24, dropout: float = 0.08, classes: int = 4):
        super().__init__()
        if in_channels < 1 or base_channels < 4:
            raise ValueError("in_channels must be positive and base_channels must be at least four")
        widths = [base_channels * (2**level) for level in range(4)]
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        previous = in_channels
        for width in widths:
            self.encoder.append(ConvBlock(previous, width, dropout))
            previous = width
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2, dropout)
        self.upconvolutions = nn.ModuleList()
        self.decoder = nn.ModuleList()
        previous = widths[-1] * 2
        for width in reversed(widths):
            self.upconvolutions.append(nn.ConvTranspose2d(previous, width, kernel_size=2, stride=2))
            self.decoder.append(ConvBlock(width * 2, width, dropout))
            previous = width
        self.head = nn.Conv2d(widths[0], classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        current = image
        for block in self.encoder:
            current = block(current)
            skips.append(current)
            current = self.pool(current)
        current = self.bottleneck(current)
        for upsample, block, skip in zip(self.upconvolutions, self.decoder, reversed(skips), strict=True):
            current = upsample(current)
            if current.shape[-2:] != skip.shape[-2:]:
                current = nn.functional.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = block(torch.cat((skip, current), dim=1))
        return self.head(current)
