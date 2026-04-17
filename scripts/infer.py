from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from models.full_model import FullModel, FullModelConfig
from utils.config import load_yaml
from utils.runtime import resolve_cfg_path, resolve_project_path


def load_model(model_cfg_path: str, ckpt_path: str, device: torch.device) -> FullModel:
    cfg = load_yaml(resolve_cfg_path(model_cfg_path, ROOT))
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
    parser.add_argument("--model_cfg", default="configs/model.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/infer")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_cfg, resolve_project_path(args.checkpoint, ROOT), device)

    tf = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    out_dir = Path(resolve_project_path(args.output_dir, ROOT))
    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(resolve_project_path(args.input_dir, ROOT))
    for path in sorted(input_dir.glob("*")):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        image = Image.open(path).convert("RGB")
        x = tf(image).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x)["Y"].squeeze(0).cpu().clamp(0, 1)
        out = transforms.ToPILImage()(y)
        out.save(out_dir / path.name)


if __name__ == "__main__":
    main()
