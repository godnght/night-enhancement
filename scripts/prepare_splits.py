from __future__ import annotations

import argparse
from pathlib import Path
import random


def list_images(folder: Path) -> list[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    return sorted([str(p.as_posix()) for p in files])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="LOL root")
    parser.add_argument("--low_dir", default="low")
    parser.add_argument("--high_dir", default="high")
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root)
    low_list = list_images(root / args.low_dir)
    high_set = {p.replace(f"/{args.high_dir}/", "/") for p in list_images(root / args.high_dir)}

    pairs: list[tuple[str, str]] = []
    for low_abs in low_list:
        rel_low = str(Path(low_abs).relative_to(root).as_posix())
        candidate_high = rel_low.replace(f"{args.low_dir}/", f"{args.high_dir}/", 1)
        if str(Path(candidate_high).as_posix().replace(f"{args.high_dir}/", "")) in high_set:
            pairs.append((rel_low, candidate_high))
        elif (root / candidate_high).exists():
            pairs.append((rel_low, candidate_high))

    random.seed(args.seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * args.train_ratio)
    train = pairs[:n_train]
    val = pairs[n_train:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "lol_train.txt", "w", encoding="utf-8") as f:
        for a, b in train:
            f.write(f"{a} {b}\n")

    with open(out / "lol_val.txt", "w", encoding="utf-8") as f:
        for a, b in val:
            f.write(f"{a} {b}\n")

    with open(out / "lol_test.txt", "w", encoding="utf-8") as f:
        for a, b in val:
            f.write(f"{a} {b}\n")

    print(f"pairs={n}, train={len(train)}, val={len(val)}, test={len(val)}")


if __name__ == "__main__":
    main()
