from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def make_run_name(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def setup_logger(name: str, log_dir: str | Path, run_name: str, level: str = "INFO") -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    numeric_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path / f"{run_name}.log", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
