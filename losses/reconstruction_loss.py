from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        image: torch.Tensor,
        halo: torch.Tensor,
        light_effect: torch.Tensor,
        shadow: torch.Tensor,
        reflectance: torch.Tensor,
    ) -> torch.Tensor:
        shadow_rgb = shadow.repeat(1, 3, 1, 1)
        recon = torch.clamp(reflectance * shadow_rgb + light_effect + halo, 0.0, 1.0)
        return self.l1(recon, image)
