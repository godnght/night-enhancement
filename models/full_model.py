from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .decompose_net import DecomposeNet
from .halo_branch import HaloBranch
from .light_suppression_net import LightSuppressionNet
from .vitlight_estimator import ViTLightEstimator


@dataclass
class FullModelConfig:
    halo_base: int = 32
    decompose_base: int = 32
    suppress_base: int = 32
    vit_embed: int = 128
    vit_heads: int = 4
    vit_layers: int = 4
    sh_dim: int = 27


class FullModel(nn.Module):
    def __init__(self, cfg: FullModelConfig) -> None:
        super().__init__()
        self.halo_branch = HaloBranch(cfg.halo_base)
        self.vitlight = ViTLightEstimator(
            embed_dim=cfg.vit_embed,
            num_heads=cfg.vit_heads,
            num_layers=cfg.vit_layers,
            sh_dim=cfg.sh_dim,
        )
        self.decompose = DecomposeNet(cfg.decompose_base)
        self.suppress = LightSuppressionNet(cfg.suppress_base)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        g, b = self.halo_branch(image)
        feat, sh, o_hat = self.vitlight(b)
        o, l, r = self.decompose(b, o_hat)
        y = self.suppress(r, o_hat)
        return {
            "G": g,
            "B": b,
            "feat": feat,
            "sh_coeff": sh,
            "O_hat": o_hat,
            "O": o,
            "L": l,
            "R": r,
            "Y": y,
        }
