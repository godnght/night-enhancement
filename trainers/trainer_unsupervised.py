from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import torch
from torch.cuda.amp import GradScaler as LegacyGradScaler, autocast as legacy_autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from datasets.lol_dataset import LOLDataset
from losses.total_loss import TotalLoss
from models.full_model import FullModel, FullModelConfig
from utils.logger import JsonlLogger
from utils.config import load_yaml
from utils.runtime import resolve_cfg_path, resolve_project_path
from utils.seed import set_seed


class UnsupervisedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        train_cfg: Dict,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.device = device

        self.criterion = TotalLoss(train_cfg.get("lambda", {}))
        self.amp = bool(train_cfg.get("amp", True))
        amp_device = "cuda" if torch.cuda.is_available() else "cpu"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(amp_device, enabled=self.amp)
            self._autocast = lambda: torch.amp.autocast(device_type=amp_device, enabled=self.amp)
        else:
            self.scaler = LegacyGradScaler(enabled=self.amp)
            self._autocast = lambda: legacy_autocast(enabled=self.amp)
        self.grad_clip = float(train_cfg.get("grad_clip", 0.0))

        out_dir = Path(train_cfg.get("output_dir", "./outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = out_dir
        self.logger = JsonlLogger(str(out_dir / "train_log.jsonl"))

    def train(self, epochs: int) -> None:
        best = float("inf")
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)
            self.logger.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best:
                best = val_loss
                self._save("best.pt", epoch)
            if epoch % int(self.cfg.get("save_every", 5)) == 0:
                self._save(f"epoch_{epoch}.pt", epoch)

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        loop = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for batch in loop:
            image = batch["low"].to(self.device)
            target_high = batch.get("high")
            if isinstance(target_high, torch.Tensor):
                target_high = target_high.to(self.device)
            else:
                target_high = None
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                outputs = self.model(image)
                loss, stats = self.criterion(image, outputs, target_high)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += float(loss.item())
            loop.set_postfix(total=f"{stats['total']:.4f}")

        return running / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        running = 0.0
        loop = tqdm(self.val_loader, desc=f"val {epoch}", leave=False)
        for batch in loop:
            image = batch["low"].to(self.device)
            target_high = batch.get("high")
            if isinstance(target_high, torch.Tensor):
                target_high = target_high.to(self.device)
            else:
                target_high = None
            outputs = self.model(image)
            loss, _ = self.criterion(image, outputs, target_high)
            running += float(loss.item())
        return running / max(len(self.val_loader), 1)

    def _save(self, name: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg,
            },
            self.ckpt_dir / name,
        )


def _build_model_cfg(cfg: dict) -> FullModelConfig:
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
    model = FullModel(_build_model_cfg(model_cfg)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    trainer = UnsupervisedTrainer(model, optimizer, train_loader, val_loader, train_cfg, device)
    trainer.train(int(train_cfg.get("epochs", 80)))


if __name__ == "__main__":
    main()
