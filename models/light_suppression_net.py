from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock


class LightSuppressionNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(6, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, reflectance: torch.Tensor, o_hat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([reflectance, o_hat], dim=1)
        return self.net(x)
