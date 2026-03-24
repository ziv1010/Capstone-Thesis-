"""File-system and logging helpers."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Tuple


@dataclass(frozen=True)
class DocumentInput:
    """One input text document plus the identifiers used by the runner."""

    source_path: Path
    file_id: str
    internal_id: str
    text: str


def configure_logging(
    log_dir: Path,
    *,
    logger_name: str = "opennyai_pipeline",
    filename_suffix: str = "",
    include_stdout: bool = True,
) -> Tuple[logging.Logger, Path]:
    """Create a console + file logger for the current run."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}{filename_suffix}.log"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if include_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger, log_path


def discover_text_files(input_dir: Path, glob_pattern: str) -> List[Path]:
    """Return all matching text files in deterministic order."""
    return sorted(path for path in input_dir.glob(glob_pattern) if path.is_file())


def read_text_file(path: Path) -> Tuple[str, str]:
    """Read a text file with UTF-8 first and a safe replacement fallback."""
    raw_bytes = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace"), "utf-8-replace"


def write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_text(path: Path, payload: str) -> None:
    """Write plain text using UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def build_tree_string(root: Path) -> str:
    """Render a small ASCII tree for a directory."""
    if not root.exists():
        return f"{root}\n`-- <missing>"

    lines = [str(root)]

    def walk(directory: Path, prefix: str) -> None:
        entries = sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
        for index, entry in enumerate(entries):
            connector = "`-- " if index == len(entries) - 1 else "|-- "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if index == len(entries) - 1 else "|   "
                walk(entry, prefix + extension)

    walk(root, "")
    return "\n".join(lines)


def compact_tree_report(paths: Iterable[Path]) -> str:
    """Build a combined tree report for multiple roots."""
    trees = [build_tree_string(path) for path in paths]
    return "\n\n".join(tree for tree in trees if tree)
