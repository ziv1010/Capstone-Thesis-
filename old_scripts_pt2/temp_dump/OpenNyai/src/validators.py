"""Validation helpers for runtime configuration and documents."""

from __future__ import annotations

import logging

from .config import PipelineConfig, SHORT_TEXT_WARNING_CHARS


def validate_pipeline_config(config: PipelineConfig) -> None:
    """Raise early for invalid user inputs."""
    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {config.input_dir}")
    if not config.input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {config.input_dir}")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if config.pipeline_batch_size <= 0:
        raise ValueError("pipeline_batch_size must be a positive integer.")
    if config.worker_processes <= 0:
        raise ValueError("worker_processes must be a positive integer.")


def normalize_summary_length(summary_length: float, logger: logging.Logger) -> float:
    """Clamp unsupported summary-length values back to adaptive mode."""
    if 0.0 <= summary_length <= 1.0:
        return summary_length
    logger.warning(
        "summary_length=%s is outside the supported 0.0-1.0 range. Resetting to 0.0 (adaptive mode).",
        summary_length,
    )
    return 0.0


def validate_gpu_request(use_gpu: bool, logger: logging.Logger) -> None:
    """Warn when a GPU run was requested but CUDA is unavailable."""
    if not use_gpu:
        return
    try:
        import torch
    except ImportError:
        logger.warning("GPU mode was requested, but torch is unavailable. OpenNyAI will fall back to CPU.")
        return

    if not torch.cuda.is_available():
        logger.warning("GPU mode was requested, but CUDA is not available. OpenNyAI will fall back to CPU.")


def warn_if_short_text(file_id: str, text: str, logger: logging.Logger) -> None:
    """Short documents are still processed, but they merit a warning."""
    if len(text.strip()) < SHORT_TEXT_WARNING_CHARS:
        logger.warning(
            "Document %s is short (%s chars). Processing will still be attempted.",
            file_id,
            len(text.strip()),
        )
