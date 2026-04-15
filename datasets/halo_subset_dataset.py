from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HaloSubsetDataset(Dataset):
    def __init__(self, root: str, list_file: str, image_size: int = 256) -> None:
        self.root = Path(root)
        with open(list_file, "r", encoding="utf-8") as f:
            self.items: List[str] = [line.strip() for line in f if line.strip()]
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        rel = self.items[idx]
        img = Image.open(self.root / rel).convert("RGB")
        return {"low": self.tf(img), "low_path": rel}
