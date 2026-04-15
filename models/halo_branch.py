from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import CAB, ConvBlock


class HaloBranch(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, s=2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, s=2)

        self.cab = CAB(base_channels * 4)

        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 4, base_channels * 2),
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(base_channels * 2, base_channels),
        )
        self.out = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.enc1(image)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x = self.cab(x3)
        x = self.dec2(x)
        x = self.dec1(x)

        halo = self.tanh(self.out(x))
        background = torch.clamp(image - halo, 0.0, 1.0)
        return halo, background
