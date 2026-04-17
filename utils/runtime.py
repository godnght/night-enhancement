from __future__ import annotations

import sys
from pathlib import Path


def project_root_from(file_path: str) -> Path:
    return Path(file_path).resolve().parents[1]


def ensure_root_on_syspath(root: Path) -> None:
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)


def resolve_cfg_path(path_or_name: str, root: Path) -> Path:
    raw = Path(path_or_name)
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        candidates.append(root / raw)
        candidates.append(root / "configs" / raw)
        if raw.suffix == "":
            candidates.append(root / "configs" / f"{raw.name}.yaml")
            candidates.append(root / f"{raw.name}.yaml")

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Config file not found: {path_or_name}. Tried: " + ", ".join(str(c) for c in candidates)
    )


def resolve_project_path(path_value: str, root: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())
