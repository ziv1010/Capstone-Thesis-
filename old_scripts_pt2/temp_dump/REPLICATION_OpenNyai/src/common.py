"""Shared utilities for the replication workspace."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def configure_logger(name: str, log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    logger: logging.Logger | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if logger is not None:
        logger.info("Running command: %s", shlex.join(command))
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def capture_command_output(command: Sequence[str], cwd: Path | None = None) -> str:
    result = run_subprocess(command, cwd=cwd, capture_output=True)
    return result.stdout.strip()


def git_commit(repo_path: Path) -> str:
    return capture_command_output(["git", "-C", str(repo_path), "rev-parse", "HEAD"])


def export_pip_freeze(env_name: str, output_path: Path) -> None:
    result = run_subprocess(
        ["micromamba", "run", "-n", env_name, "python", "-m", "pip", "freeze"],
        capture_output=True,
    )
    write_text(output_path, result.stdout)


def collect_machine_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "hostname": platform.node(),
    }


def collect_gpu_info() -> dict[str, Any]:
    try:
        result = run_subprocess(["nvidia-smi"], capture_output=True)
        return {
            "available": True,
            "nvidia_smi": result.stdout,
        }
    except Exception as exc:  # pragma: no cover - depends on runtime machine
        return {
            "available": False,
            "error": str(exc),
        }


def flatten_confusion_matrix(
    labels: Sequence[str],
    matrix: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gold_label, values in zip(labels, matrix):
        row = {"gold_label": gold_label}
        for pred_label, value in zip(labels, values):
            row[pred_label] = int(value)
        rows.append(row)
    return rows
