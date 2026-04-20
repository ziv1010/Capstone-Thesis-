from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import load_json


def load_case_json(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Case JSON at {path} must be an object.")
    return payload
