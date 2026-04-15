from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaloSeparationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _edge(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        return torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]).mean() + torch.abs(
            gray[:, :, 1:, :] - gray[:, :, :-1, :]
        ).mean()

    def forward(self, halo: torch.Tensor, image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        edge_refine = self._edge(halo)
        preserve = F.l1_loss(torch.clamp(background + halo, 0.0, 1.0), image)
        return edge_refine + preserve
