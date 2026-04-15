from __future__ import annotations

import torch
import torch.nn as nn


class GradientExclusionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, light_effect: torch.Tensor, reflectance: torch.Tensor) -> torch.Tensor:
        dx1, dy1 = self._grad(light_effect)
        dx2, dy2 = self._grad(reflectance)
        loss_x = (dx1.abs() * dx2.abs()).mean()
        loss_y = (dy1.abs() * dy2.abs()).mean()
        return loss_x + loss_y
