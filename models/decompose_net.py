from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBlock


class DecomposeNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(6, base_channels),
            ConvBlock(base_channels, base_channels),
            ConvBlock(base_channels, base_channels),
        )
        self.o_head = nn.Sequential(nn.Conv2d(base_channels, 3, 3, 1, 1), nn.Sigmoid())
        self.l_head = nn.Sequential(nn.Conv2d(base_channels, 1, 3, 1, 1), nn.Sigmoid())
        self.r_head = nn.Sequential(nn.Conv2d(base_channels, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, background: torch.Tensor, o_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([background, o_hat], dim=1)
        feat = self.stem(x)
        o = self.o_head(feat)
        l = self.l_head(feat)
        r = self.r_head(feat)
        return o, l, r
