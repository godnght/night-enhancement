from __future__ import annotations

import torch
import torch.nn as nn


class ColorConstancyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, background: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        def channel_balance(x: torch.Tensor) -> torch.Tensor:
            m = x.mean(dim=(2, 3))
            rg = (m[:, 0] - m[:, 1]).pow(2)
            rb = (m[:, 0] - m[:, 2]).pow(2)
            gb = (m[:, 1] - m[:, 2]).pow(2)
            return (rg + rb + gb).mean()

        return 0.5 * channel_balance(background) + 0.5 * channel_balance(output)
