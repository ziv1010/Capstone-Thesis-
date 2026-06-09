from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return project_root().parent


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    expanded = Path(os.path.expandvars(str(path))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return ((Path(base) if base is not None else project_root()) / expanded).resolve()


def resolve_config_paths(paths_cfg: dict[str, Any], base: str | Path | None = None) -> dict[str, Any]:
    return {
        key: str(resolve_path(value, base=base)) if isinstance(value, str) else value
        for key, value in paths_cfg.items()
    }


def _resolve_known_config_paths(cfg: dict[str, Any]) -> None:
    if isinstance(cfg.get("paths"), dict):
        cfg["paths"] = resolve_config_paths(cfg["paths"])

    inference_cfg = cfg.get("inference")
    if isinstance(inference_cfg, dict):
        for key in ("trained_model_run_dir", "trained_graph_cache"):
            value = inference_cfg.get(key)
            if isinstance(value, str):
                inference_cfg[key] = str(resolve_path(value))


def _portable_path_string(value: str) -> str:
    expanded = Path(value)
    if not expanded.is_absolute():
        return value

    section_root = project_root()
    repository_root = repo_root()
    try:
        return expanded.relative_to(section_root).as_posix()
    except ValueError:
        pass

    try:
        expanded.relative_to(repository_root)
    except ValueError:
        return value

    return os.path.relpath(expanded, section_root).replace(os.sep, "/")


def _make_paths_portable(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _make_paths_portable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_make_paths_portable(value) for value in payload]
    if isinstance(payload, str):
        return _portable_path_string(payload)
    return payload


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(payload: Any, path: str | Path, indent: int = 2) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(_make_paths_portable(payload), handle, indent=indent, ensure_ascii=False)
        handle.write("\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML config at {path} must parse to a mapping.")
    _resolve_known_config_paths(data)
    return data


def dump_yaml(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_make_paths_portable(payload), handle, sort_keys=False)


def list_json_files(root: str | Path, pattern: str = "*.json") -> list[Path]:
    return sorted(Path(root).glob(pattern))


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged
