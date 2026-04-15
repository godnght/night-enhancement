from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureTextureConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def _sobel_grad(self, x: torch.Tensor) -> torch.Tensor:
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device
        ).view(1, 1, 3, 3)
        gx = F.conv2d(x, kernel_x, padding=1)
        gy = F.conv2d(x, kernel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def _high_pass(self, x: torch.Tensor) -> torch.Tensor:
        blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - blur

    def forward(self, bg_shadow: torch.Tensor, out_shadow: torch.Tensor) -> torch.Tensor:
        g1 = self._sobel_grad(bg_shadow)
        g2 = self._sobel_grad(out_shadow)
        h1 = self._high_pass(bg_shadow)
        h2 = self._high_pass(out_shadow)
        return self.l1(g1, g2) + self.l1(h1, h2)
