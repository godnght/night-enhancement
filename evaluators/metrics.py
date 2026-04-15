from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target) ** 2).item())


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    m = mse(pred, target)
    if m <= 1e-12:
        return 99.0
    return 10.0 * math.log10((max_val * max_val) / m)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean().item()
    mu_y = target.mean().item()
    var_x = pred.var().item()
    var_y = target.var().item()
    cov = ((pred - mu_x) * (target - mu_y)).mean().item()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(num / (den + 1e-8))


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
