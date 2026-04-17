from __future__ import annotations

import torch
import torch.nn as nn


class PairedEnhanceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        if target is None:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        return self.l1(pred, target)
