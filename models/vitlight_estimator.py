from __future__ import annotations

import torch
import torch.nn as nn


class ViTLightEstimator(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        sh_dim: int = 27,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.sh_head = nn.Linear(embed_dim, sh_dim)
        self.map_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, background: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = background.shape
        patches = self.patch(background)
        ph, pw = patches.shape[-2:]
        tokens = patches.flatten(2).transpose(1, 2)
        encoded = self.encoder(tokens)

        feat = encoded.mean(dim=1)
        sh_coeff = self.sh_head(feat)

        maps = encoded.transpose(1, 2).reshape(b, -1, ph, pw)
        o_hat = self.map_head(maps)
        o_hat = nn.functional.interpolate(o_hat, size=(h, w), mode="bilinear", align_corners=False)
        return feat, sh_coeff, o_hat
