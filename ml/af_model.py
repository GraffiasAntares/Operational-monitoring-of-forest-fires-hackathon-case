"""Small U-Net used for the AF baseline on a 4 GB laptop GPU."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as functional


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = self.up(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, value), dim=1))


class AfUNet(nn.Module):
    """Four-downsampling U-Net; base=16 is about 2 million parameters."""

    def __init__(self, in_channels: int = 15, base_channels: int = 16, dropout: float = .05, out_channels: int = 1) -> None:
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        b = base_channels
        self.stem = DoubleConv(in_channels, b)
        self.down1 = Down(b, b * 2)
        self.down2 = Down(b * 2, b * 4)
        self.down3 = Down(b * 4, b * 8)
        self.down4 = Down(b * 8, b * 16)
        self.dropout = nn.Dropout2d(dropout)
        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(value)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.dropout(self.down4(x4))
        value = self.up1(x5, x4)
        value = self.up2(value, x3)
        value = self.up3(value, x2)
        value = self.up4(value, x1)
        return self.head(value)
