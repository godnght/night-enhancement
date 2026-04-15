from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def _read_split_file(split_path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1:
                pairs.append((parts[0], ""))
            else:
                pairs.append((parts[0], parts[1]))
    return pairs


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _replace_part(parts: List[str], old: str, new: str) -> List[str]:
    out = parts[:]
    for i, p in enumerate(out):
        if p.lower() == old:
            out[i] = new
    return out


def _candidate_high_paths(low_path: Path) -> List[Path]:
    parts = list(low_path.parts)
    candidates: List[Path] = []
    for low_token in ("low", "input", "underexposed"):
        if any(p.lower() == low_token for p in parts):
            for high_token in ("high", "normal", "gt"):
                replaced = _replace_part(parts, low_token, high_token)
                candidates.append(Path(*replaced))
                # Keep original case variants commonly used in public datasets.
                candidates.append(Path(*_replace_part(parts, low_token, high_token.capitalize())))
                candidates.append(Path(*_replace_part(parts, low_token, high_token.upper())))
    return candidates


def _discover_pairs(scan_root: Path, rel_base: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for p in scan_root.rglob("*"):
        if not p.is_file() or not _is_image(p):
            continue
        if not any(token in str(part).lower() for token in ("low", "input", "underexposed") for part in p.parts):
            continue

        for cand in _candidate_high_paths(p):
            if cand.exists() and _is_image(cand):
                low_rel = str(p.relative_to(rel_base).as_posix())
                high_rel = str(cand.relative_to(rel_base).as_posix())
                pairs.append((low_rel, high_rel))
                break

    pairs = sorted(set(pairs))
    return pairs


class LOLDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: str | None = None,
        split: str = "train",
        image_size: int = 256,
        paired: bool = True,
        seed: int = 42,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> None:
        self.root = Path(root)
        self.paired = paired

        if split_file and Path(split_file).exists():
            self.pairs = _read_split_file(split_file)
        else:
            self.pairs = self._build_pairs_by_structure(
                split=split,
                seed=seed,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

        self.paired = paired
        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _build_pairs_by_structure(
        self,
        split: str,
        seed: int,
        val_ratio: float,
        test_ratio: float,
    ) -> List[Tuple[str, str]]:
        # Prefer official LOL split semantics when our485/eval15 are available.
        if (self.root / "our485").exists() and (self.root / "eval15").exists():
            train_pairs = _discover_pairs(self.root / "our485", self.root)
            eval_pairs = _discover_pairs(self.root / "eval15", self.root)
            if split == "train":
                return train_pairs
            return eval_pairs

        all_pairs = _discover_pairs(self.root, self.root)
        if not all_pairs:
            raise RuntimeError(
                f"No low/high pairs found under dataset root: {self.root}. "
                "Check directory structure or provide an explicit split file."
            )

        random.Random(seed).shuffle(all_pairs)
        n = len(all_pairs)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        n_train = max(n - n_val - n_test, 1)

        train_pairs = all_pairs[:n_train]
        val_pairs = all_pairs[n_train : n_train + n_val]
        test_pairs = all_pairs[n_train + n_val :]

        if split == "train":
            return train_pairs
        if split == "val":
            return val_pairs if val_pairs else test_pairs
        return test_pairs if test_pairs else val_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.root / rel_path).convert("RGB")
        return self.tf(img)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        low_rel, high_rel = self.pairs[idx]
        low = self._load(low_rel)
        sample: Dict[str, torch.Tensor | str] = {"low": low, "low_path": low_rel}
        if self.paired and high_rel:
            high = self._load(high_rel)
            sample["high"] = high
            sample["high_path"] = high_rel
        return sample
