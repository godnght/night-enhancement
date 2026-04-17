from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tqdm import tqdm


@dataclass
class Asset:
    name: str
    urls: List[str]
    gdrive_id: str = ""


DATASET_ASSETS: Dict[str, List[Asset]] = {
    "lol": [
        Asset(
            name="LOL.zip",
            urls=[
                "https://drive.google.com/uc?export=download&id=157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB",
            ],
            gdrive_id="157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB",
        ),
    ],
    "lolv2": [
        Asset(
            name="LOLv2.zip",
            urls=[
                "https://drive.google.com/uc?export=download&id=1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC",
            ],
            gdrive_id="1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC",
        ),
    ],
}


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HIGH_TOKENS = ("high", "normal", "gt")
LOW_TOKENS = ("low", "input", "underexposed")


def _is_valid_archive(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 4096:
        return False
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip"):
        return zipfile.is_zipfile(path)
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        return tarfile.is_tarfile(path)
    return False


def _read_prefix(path: Path, n: int = 2048) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _looks_like_html(path: Path) -> bool:
    head = _read_prefix(path).lower()
    return b"<html" in head or b"<!doctype html" in head or b"google drive" in head


def _validate_archive_or_raise(path: Path) -> None:
    if _is_valid_archive(path):
        return
    if _looks_like_html(path):
        raise RuntimeError(
            f"downloaded file is HTML page instead of archive: {path}. "
            "Likely Google Drive quota/confirmation page."
        )
    raise RuntimeError(f"downloaded file is not a valid archive: {path}")


def _download_from_url(url: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(out_file, "wb") as f:
        total = int(resp.headers.get("Content-Length", "0"))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"download {out_file.name}")
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()


def _extract_confirm_token(html: str) -> str:
    patterns = [
        r"confirm=([0-9A-Za-z_]+)",
        r'"confirm"\s*:\s*"([0-9A-Za-z_]+)"',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return ""


def _download_google_drive(file_id: str, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    base = "https://drive.google.com/uc"
    headers = {"User-Agent": "Mozilla/5.0"}

    first_url = f"{base}?{urlencode({'export': 'download', 'id': file_id})}"
    req1 = Request(first_url, headers=headers)

    with urlopen(req1) as resp1:
        content_type = resp1.headers.get("Content-Type", "")
        content_disp = resp1.headers.get("Content-Disposition", "")

        if "application" in content_type or content_disp:
            total = int(resp1.headers.get("Content-Length", "0"))
            with open(out_file, "wb") as f:
                pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"download {out_file.name}")
                while True:
                    chunk = resp1.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
                pbar.close()
            return

        html = resp1.read().decode("utf-8", errors="ignore")

    token = _extract_confirm_token(html)
    if not token:
        raise RuntimeError(
            "Google Drive download confirmation token not found. "
            "This may be caused by quota limits or anti-bot page."
        )

    second_url = f"{base}?{urlencode({'export': 'download', 'confirm': token, 'id': file_id})}"
    req2 = Request(second_url, headers=headers)
    with urlopen(req2) as resp2, open(out_file, "wb") as f:
        total = int(resp2.headers.get("Content-Length", "0"))
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"download {out_file.name}")
        while True:
            chunk = resp2.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()


def _download(asset: Asset, out_file: Path) -> None:
    errors: List[str] = []

    for url in asset.urls:
        try:
            _download_from_url(url, out_file)
            _validate_archive_or_raise(out_file)
            return
        except Exception as exc:
            errors.append(f"URL failed: {url} -> {exc}")

    if asset.gdrive_id:
        try:
            _download_google_drive(asset.gdrive_id, out_file)
            _validate_archive_or_raise(out_file)
            return
        except Exception as exc:
            errors.append(f"Google Drive failed: id={asset.gdrive_id} -> {exc}")

    raise RuntimeError("; ".join(errors))


def _extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes).lower()

    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
        return

    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(out_dir)
        return

    raise ValueError(f"unsupported archive type: {archive}")


def _collect_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _looks_like_low(path: Path) -> bool:
    text = str(path.as_posix()).lower()
    return any(t in text for t in LOW_TOKENS)


def _looks_like_high(path: Path) -> bool:
    text = str(path.as_posix()).lower()
    return any(t in text for t in HIGH_TOKENS)


def _discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    images = _collect_images(root)
    low_map: Dict[str, Path] = {}
    high_map: Dict[str, Path] = {}

    for p in images:
        key = p.name
        if _looks_like_low(p):
            low_map[key] = p
        elif _looks_like_high(p):
            high_map[key] = p

    keys = sorted(set(low_map.keys()) & set(high_map.keys()))
    pairs = [(low_map[k], high_map[k]) for k in keys]
    return pairs


def _copy_pairs(
    pairs: List[Tuple[Path, Path]],
    dst_root: Path,
    split_file: Path,
    prefix: str,
) -> int:
    low_dir = dst_root / "low"
    high_dir = dst_root / "high"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)
    split_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(split_file, "a", encoding="utf-8") as f:
        for i, (low_src, high_src) in enumerate(pairs):
            base = f"{prefix}_{i:06d}{low_src.suffix.lower()}"
            low_rel = f"low/{base}"
            high_rel = f"high/{base}"
            shutil.copy2(low_src, dst_root / low_rel)
            shutil.copy2(high_src, dst_root / high_rel)
            f.write(f"{low_rel} {high_rel}\n")
            count += 1

    return count


def _append_split_from_pairs(
    pairs: List[Tuple[Path, Path]],
    split_file: Path,
    prefix: str,
) -> int:
    split_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(split_file, "a", encoding="utf-8") as f:
        for i, (low_src, _high_src) in enumerate(pairs):
            base = f"{prefix}_{i:06d}{low_src.suffix.lower()}"
            low_rel = f"low/{base}"
            high_rel = f"high/{base}"
            f.write(f"{low_rel} {high_rel}\n")
            count += 1
    return count


def _clear_split_files(split_dir: Path, stem: str) -> None:
    for name in (f"{stem}_train.txt", f"{stem}_val.txt", f"{stem}_test.txt"):
        p = split_dir / name
        if p.exists():
            p.unlink()


def _prepare_lol_layout(data_root: Path, extract_root: Path, split_dir: Path) -> None:
    dataset_root = data_root / "LOL"
    dataset_root.mkdir(parents=True, exist_ok=True)
    _clear_split_files(split_dir, "lol")

    train_split = split_dir / "lol_train.txt"
    val_split = split_dir / "lol_val.txt"
    test_split = split_dir / "lol_test.txt"

    total_train = 0
    total_eval = 0

    candidate_dirs = [
        extract_root,
        extract_root / "LOLdataset",
        extract_root / "LOL-v2",
        extract_root / "low",
    ]

    for sub in ("our485", "eval15"):
        sub_dir = extract_root / sub
        if not sub_dir.exists():
            for cand in candidate_dirs:
                maybe = cand / sub
                if maybe.exists():
                    sub_dir = maybe
                    break
        if not sub_dir.exists():
            continue
        pairs = _discover_pairs(sub_dir)
        if not pairs:
            continue

        if sub == "our485":
            total_train += _copy_pairs(pairs, dataset_root, train_split, prefix="our485")
        else:
            total_eval += _copy_pairs(pairs, dataset_root, val_split, prefix="eval15")
            _append_split_from_pairs(pairs, test_split, prefix="eval15")

    if total_train == 0 and total_eval == 0:
        raise RuntimeError("LOL pairing failed. Please check downloaded files and folder names.")

    print(f"LOL prepared: train={total_train}, val/test={total_eval}")
    print(f"dataset root: {dataset_root}")


def _prepare_lolv2_layout(data_root: Path, extract_root: Path, split_dir: Path) -> None:
    dataset_root = data_root / "LOLv2"
    dataset_root.mkdir(parents=True, exist_ok=True)
    _clear_split_files(split_dir, "lolv2")

    train_split = split_dir / "lolv2_train.txt"
    val_split = split_dir / "lolv2_val.txt"
    test_split = split_dir / "lolv2_test.txt"

    groups = [
        ("Real_captured", "real"),
        ("Synthetic", "syn"),
    ]

    search_roots = [
        extract_root,
        extract_root / "LOL-v2",
        extract_root / "LOLv2",
    ]

    total = 0
    for folder_name, prefix in groups:
        sub_dir = extract_root / folder_name
        if not sub_dir.exists():
            for sr in search_roots:
                maybe = sr / folder_name
                if maybe.exists():
                    sub_dir = maybe
                    break
        if not sub_dir.exists():
            continue
        pairs = _discover_pairs(sub_dir)
        if not pairs:
            continue

        n = len(pairs)
        n_train = int(n * 0.8)
        train_pairs = pairs[:n_train]
        eval_pairs = pairs[n_train:]

        total += _copy_pairs(train_pairs, dataset_root, train_split, prefix=f"{prefix}_train")
        _copy_pairs(eval_pairs, dataset_root, val_split, prefix=f"{prefix}_val")
        _copy_pairs(eval_pairs, dataset_root, test_split, prefix=f"{prefix}_test")

    if total == 0:
        raise RuntimeError("LOL-v2 pairing failed. Please check downloaded files and folder names.")

    print(f"LOL-v2 prepared: train={total}")
    print(f"dataset root: {dataset_root}")


def _resolve_local_extracted_source(name: str, local_archive_dir: Path | None) -> Path | None:
    if local_archive_dir is None:
        return None
    if not local_archive_dir.exists():
        return None

    candidates: List[Path] = []
    if name == "lol":
        candidates = [
            local_archive_dir / "LOL",
            local_archive_dir / "lol",
            local_archive_dir,
        ]
    elif name == "lolv2":
        candidates = [
            local_archive_dir / "LOLv2",
            local_archive_dir / "LOL-v2",
            local_archive_dir / "lolv2",
            local_archive_dir,
        ]

    for c in candidates:
        if not c.exists():
            continue
        if name == "lol":
            if (c / "our485").exists() or (c / "eval15").exists():
                return c
        if name == "lolv2":
            if (c / "Real_captured").exists() or (c / "Synthetic").exists():
                return c
    return None


def download_dataset(
    name: str,
    data_root: Path,
    cache_root: Path,
    split_dir: Path,
    skip_existing: bool,
    local_archive_dir: Path | None,
) -> None:
    assets = DATASET_ASSETS[name]
    extract_root = cache_root / name / "extracted"
    archive_root = cache_root / name / "archives"

    local_extracted_source = _resolve_local_extracted_source(name, local_archive_dir)
    if local_extracted_source is not None:
        print(f"use local extracted source: {local_extracted_source}")
        if name == "lol":
            _prepare_lol_layout(data_root, local_extracted_source, split_dir)
        elif name == "lolv2":
            _prepare_lolv2_layout(data_root, local_extracted_source, split_dir)
        else:
            raise ValueError(f"unsupported dataset: {name}")
        return

    for asset in assets:
        archive_path = archive_root / asset.name
        local_archive = (local_archive_dir / asset.name) if local_archive_dir else None

        if local_archive and local_archive.exists() and _is_valid_archive(local_archive):
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_archive, archive_path)
            print(f"use local archive: {local_archive}")

        if archive_path.exists() and not _is_valid_archive(archive_path):
            print(f"invalid cached archive detected, removing: {archive_path}")
            archive_path.unlink()

        if archive_path.exists() and skip_existing:
            print(f"skip existing archive: {archive_path}")
        else:
            print(f"downloading {name}: {asset.name}")
            _download(asset, archive_path)

        target_extract = extract_root
        target_extract.mkdir(parents=True, exist_ok=True)
        marker = target_extract / f".{asset.name}.done"
        if marker.exists() and skip_existing:
            print(f"skip existing extract: {asset.name}")
        else:
            print(f"extracting {asset.name}")
            _extract(archive_path, target_extract)
            marker.touch()

    if name == "lol":
        _prepare_lol_layout(data_root, extract_root, split_dir)
    elif name == "lolv2":
        _prepare_lolv2_layout(data_root, extract_root, split_dir)
    else:
        raise ValueError(f"unsupported dataset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["lol"],
        choices=sorted(DATASET_ASSETS.keys()),
        help="datasets to download",
    )
    parser.add_argument("--data_root", default="data", help="final dataset root")
    parser.add_argument("--cache_root", default="data/_downloads", help="download cache")
    parser.add_argument("--split_dir", default="data/splits", help="split file output")
    parser.add_argument(
        "--local_archive_dir",
        default="",
        help="optional folder containing manually downloaded archives (LOL.zip/LOLv2.zip)",
    )
    parser.add_argument("--no_skip", action="store_true", help="re-download and re-extract even if exists")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    split_dir = Path(args.split_dir)
    local_archive_dir = Path(args.local_archive_dir) if args.local_archive_dir else None
    skip_existing = not args.no_skip

    for dataset in args.datasets:
        print(f"\n=== processing {dataset} ===")
        download_dataset(dataset, data_root, cache_root, split_dir, skip_existing, local_archive_dir)

    print("\nAll requested datasets are prepared.")


if __name__ == "__main__":
    main()
