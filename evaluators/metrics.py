from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    m = mse(pred, target)
    if m <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val * max_val) / m)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    # Expected shape: [B, C, H, W], value range [0, 1].
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    b = pred_np.shape[0]
    scores = []
    for i in range(b):
        x = np.transpose(pred_np[i], (1, 2, 0))
        y = np.transpose(target_np[i], (1, 2, 0))
        scores.append(ssim_fn(x, y, data_range=1.0, channel_axis=2))
    return float(np.mean(scores))


def information_entropy(img: torch.Tensor) -> float:
    arr = (img.detach().cpu().numpy().clip(0, 1) * 255.0).astype(np.uint8)
    hist, _ = np.histogram(arr, bins=256, range=(0, 255), density=True)
    hist = hist + 1e-12
    return float(-(hist * np.log2(hist)).sum())


def evaluate_pair(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "mse": mse(pred, target),
        "psnr": psnr(pred, target),
        "ssim": ssim(pred, target),
        "ie": information_entropy(pred),
    }
