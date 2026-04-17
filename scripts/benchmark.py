from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.full_model import FullModel, FullModelConfig
from utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    cfg = load_yaml(args.model_cfg)
    model = FullModel(
        FullModelConfig(
            halo_base=cfg["halo_branch"]["base_channels"],
            decompose_base=cfg["decompose_net"]["base_channels"],
            suppress_base=cfg["light_suppression"]["base_channels"],
            vit_embed=cfg["vitlight"]["embed_dim"],
            vit_heads=cfg["vitlight"]["num_heads"],
            vit_layers=cfg["vitlight"]["num_layers"],
            sh_dim=cfg["vitlight"]["sh_dim"],
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    x = torch.randn(1, 3, args.size, args.size, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.runs):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    ms = dt * 1000.0 / args.runs
    fps = 1000.0 / ms
    print(f"avg latency: {ms:.3f} ms")
    print(f"fps: {fps:.2f}")


if __name__ == "__main__":
    main()
