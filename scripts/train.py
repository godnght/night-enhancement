from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from datasets.lol_dataset import LOLDataset
from models.full_model import FullModel, FullModelConfig
from trainers.trainer_unsupervised import UnsupervisedTrainer
from utils.config import load_yaml
from utils.runtime import resolve_cfg_path, resolve_project_path
from utils.seed import set_seed


def build_model_cfg(cfg: dict) -> FullModelConfig:
    return FullModelConfig(
        halo_base=cfg["halo_branch"]["base_channels"],
        decompose_base=cfg["decompose_net"]["base_channels"],
        suppress_base=cfg["light_suppression"]["base_channels"],
        vit_embed=cfg["vitlight"]["embed_dim"],
        vit_heads=cfg["vitlight"]["num_heads"],
        vit_layers=cfg["vitlight"]["num_layers"],
        sh_dim=cfg["vitlight"]["sh_dim"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfg", default="configs/dataset.yaml")
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--train_cfg", default="configs/train.yaml")
    args = parser.parse_args()

    dataset_cfg = load_yaml(resolve_cfg_path(args.dataset_cfg, ROOT))
    model_cfg = load_yaml(resolve_cfg_path(args.model_cfg, ROOT))
    train_cfg = load_yaml(resolve_cfg_path(args.train_cfg, ROOT))

    dataset_cfg["root"] = resolve_project_path(dataset_cfg["root"], ROOT)
    for key in ("train_split", "val_split", "test_split"):
        value = str(dataset_cfg.get(key, "")).strip()
        if value:
            dataset_cfg[key] = resolve_project_path(value, ROOT)

    train_cfg["output_dir"] = resolve_project_path(str(train_cfg.get("output_dir", "outputs")), ROOT)

    set_seed(int(train_cfg.get("seed", 42)))

    train_ds = LOLDataset(
        root=dataset_cfg["root"],
        split_file=dataset_cfg.get("train_split", ""),
        split="train",
        image_size=int(dataset_cfg.get("image_size", 256)),
        paired=bool(dataset_cfg.get("paired", True)),
        seed=int(dataset_cfg.get("split_seed", 42)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.1)),
    )
    val_ds = LOLDataset(
        root=dataset_cfg["root"],
        split_file=dataset_cfg.get("val_split", ""),
        split="val",
        image_size=int(dataset_cfg.get("image_size", 256)),
        paired=bool(dataset_cfg.get("paired", True)),
        seed=int(dataset_cfg.get("split_seed", 42)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.1)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=bool(dataset_cfg.get("pin_memory", True)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 4)),
        pin_memory=bool(dataset_cfg.get("pin_memory", True)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(build_model_cfg(model_cfg)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    trainer = UnsupervisedTrainer(model, optimizer, train_loader, val_loader, train_cfg, device)
    trainer.train(int(train_cfg.get("epochs", 80)))


if __name__ == "__main__":
    main()
