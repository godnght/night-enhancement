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
from evaluators.metrics import evaluate_pair
from models.full_model import FullModel, FullModelConfig
from utils.config import load_yaml
from utils.runtime import resolve_cfg_path, resolve_project_path


def build_model(cfg: dict, ckpt_path: str, device: torch.device) -> FullModel:
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
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.to(device).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_cfg", default="configs/dataset.yaml")
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    dcfg = load_yaml(resolve_cfg_path(args.dataset_cfg, ROOT))
    mcfg = load_yaml(resolve_cfg_path(args.model_cfg, ROOT))
    ckpt_path = resolve_project_path(args.checkpoint, ROOT)
    if not Path(ckpt_path).exists():
        alt = ROOT / "outputs" / Path(args.checkpoint).name
        if alt.exists():
            ckpt_path = str(alt)
        else:
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path}. "
                f"Also tried: {alt}"
            )

    dcfg["root"] = resolve_project_path(dcfg["root"], ROOT)
    for key in ("train_split", "val_split", "test_split"):
        value = str(dcfg.get(key, "")).strip()
        if value:
            dcfg[key] = resolve_project_path(value, ROOT)

    ds = LOLDataset(
        root=dcfg["root"],
        split_file=dcfg.get("test_split", ""),
        split="test",
        image_size=int(dcfg.get("image_size", 256)),
        paired=True,
        seed=int(dcfg.get("split_seed", 42)),
        val_ratio=float(dcfg.get("val_ratio", 0.1)),
        test_ratio=float(dcfg.get("test_ratio", 0.1)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(mcfg, ckpt_path, device)

    sums = {"mse": 0.0, "psnr": 0.0, "ssim": 0.0, "ie": 0.0}
    count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["low"].to(device)
            y = model(x)["Y"].cpu()
            gt = batch["high"]
            m = evaluate_pair(y, gt)
            for k, v in m.items():
                sums[k] += v
            count += 1

    count = max(count, 1)
    print("Evaluation Results")
    for k in sums:
        print(f"{k}: {sums[k] / count:.6f}")


if __name__ == "__main__":
    main()
