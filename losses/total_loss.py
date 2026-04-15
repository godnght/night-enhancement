from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .color_constancy_loss import ColorConstancyLoss
from .gradient_exclusion_loss import GradientExclusionLoss
from .halo_losses import HaloSeparationLoss
from .reconstruction_loss import ReconstructionLoss
from .structure_texture_loss import StructureTextureConsistencyLoss


class TotalLoss(nn.Module):
    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self.weights = weights
        self.recon = ReconstructionLoss()
        self.color = ColorConstancyLoss()
        self.structure = StructureTextureConsistencyLoss()
        self.halo = HaloSeparationLoss()
        self.grad_excl = GradientExclusionLoss()

    def forward(self, image: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        bg_shadow = outputs["B"].max(dim=1, keepdim=True).values
        out_shadow = outputs["Y"].max(dim=1, keepdim=True).values

        l_recon = self.recon(image, outputs["G"], outputs["O"], outputs["L"], outputs["R"])
        l_color = self.color(outputs["B"], outputs["Y"])
        l_struct = self.structure(bg_shadow, out_shadow)
        l_halo = self.halo(outputs["G"], image, outputs["B"])
        l_grad = self.grad_excl(outputs["O"], outputs["R"])

        total = (
            self.weights.get("recon", 1.0) * l_recon
            + self.weights.get("color", 0.0) * l_color
            + self.weights.get("structure", 0.0) * l_struct
            + self.weights.get("halo", 0.0) * l_halo
            + self.weights.get("grad_excl", 0.0) * l_grad
        )

        stats = {
            "recon": float(l_recon.detach().item()),
            "color": float(l_color.detach().item()),
            "structure": float(l_struct.detach().item()),
            "halo": float(l_halo.detach().item()),
            "grad_excl": float(l_grad.detach().item()),
            "total": float(total.detach().item()),
        }
        return total, stats
