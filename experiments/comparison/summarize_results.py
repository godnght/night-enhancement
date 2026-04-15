from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="experiments/comparison/sota_table_template.csv")
    args = parser.parse_args()

    by_dataset = defaultdict(list)
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_dataset[row["dataset"]].append(row)

    for ds, rows in by_dataset.items():
        print(f"\n=== {ds} ===")
        scored = []
        for r in rows:
            try:
                score = float(r["psnr"])
            except ValueError:
                continue
            scored.append((score, r["method"], r))
        scored.sort(reverse=True)
        for s, m, r in scored[:5]:
            print(f"{m:15s} psnr={s:.3f} ssim={r['ssim']} niqe={r['niqe']}")


if __name__ == "__main__":
    main()
