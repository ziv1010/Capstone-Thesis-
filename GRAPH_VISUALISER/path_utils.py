"""Path helpers for GRAPH_VISUALISER scripts.

All relative paths in config.yaml are interpreted relative to the config file,
not the shell's current working directory.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parent


def resolve_path(value: str | Path, base_dir: Path = APP_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def resolve_config_path(path: str | Path) -> Path:
    config_path = Path(path).expanduser()
    if config_path.is_absolute():
        return config_path

    cwd_path = (Path.cwd() / config_path).resolve()
    if cwd_path.exists():
        return cwd_path

    return (APP_ROOT / config_path).resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = resolve_config_path(path)
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    return resolve_config_paths(config, config_path.parent), config_path


def resolve_config_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(config)

    for bucket in resolved.get("buckets", []) or []:
        data_dir = bucket.get("data_dir")
        if data_dir:
            bucket["data_dir"] = str(resolve_path(data_dir, base_dir))

    output_dir = resolved.get("output_dir")
    if output_dir:
        resolved["output_dir"] = str(resolve_path(output_dir, base_dir))

    output = resolved.get("output")
    if isinstance(output, dict) and output.get("dir"):
        output["dir"] = str(resolve_path(output["dir"], base_dir))

    return resolved


def resolve_output_arg(value: str | Path, config_path: Path) -> Path:
    return resolve_path(value, config_path.parent)
