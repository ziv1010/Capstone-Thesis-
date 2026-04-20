from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import yaml


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = _ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Any, path: str | Path) -> None:
    p = _ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def save_joblib(obj: Any, path: str | Path) -> None:
    p = _ensure_parent(path)
    joblib.dump(obj, p)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


def save_torch(obj: Any, path: str | Path) -> None:
    p = _ensure_parent(path)
    import torch

    torch.save(obj, p)


def load_torch(path: str | Path) -> Any:
    import torch

    return torch.load(Path(path), map_location="cpu")
