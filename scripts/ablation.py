from __future__ import annotations

import argparse
import subprocess


ABLATIONS = {
    "w_o_halo_branch": ["lambda.halo=0.0"],
    "w_o_structure": ["lambda.structure=0.0"],
    "w_o_color": ["lambda.color=0.0"],
    "w_o_grad_excl": ["lambda.grad_excl=0.0"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for name, overrides in ABLATIONS.items():
        cmd = ["python", "scripts/train.py", "--train_cfg", "configs/train.yaml"]
        print(f"[{name}] {' '.join(cmd)} # overrides: {overrides}")
        if not args.dry_run:
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
