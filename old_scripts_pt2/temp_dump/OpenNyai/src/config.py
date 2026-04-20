"""Configuration helpers for the local OpenNyAI pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOB_PATTERN = "*.txt"
DEFAULT_BATCH_SIZE = 40000
DEFAULT_SUMMARY_LENGTH = 0.0
DEFAULT_PREPROCESSING_MODEL = "en_core_web_trf"
DEFAULT_NER_MODEL = "en_legal_ner_trf"
DEFAULT_PIPELINE_BATCH_SIZE = 1
DEFAULT_WORKER_PROCESSES = 1
SHORT_TEXT_WARNING_CHARS = 200


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(value: str | None, default: Path) -> Path:
    if value is None or not str(value).strip():
        return default.resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def load_project_environment(project_root: Path = PROJECT_ROOT) -> Path:
    """Load .env values when a project-local file is present."""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    return env_path


def configure_local_runtime(project_root: Path = PROJECT_ROOT) -> Dict[str, Path]:
    """Redirect model and Hugging Face caches into the project workspace."""
    runtime_home = Path(os.getenv("OPENNYAI_RUNTIME_HOME", project_root / ".runtime_home")).resolve()
    cache_root = Path(os.getenv("OPENNYAI_CACHE_ROOT", project_root / ".cache")).resolve()
    huggingface_root = cache_root / "huggingface"
    torch_root = cache_root / "torch"

    for path in (
        runtime_home,
        cache_root,
        huggingface_root,
        huggingface_root / "hub",
        huggingface_root / "transformers",
        torch_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(runtime_home)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["HF_HOME"] = str(huggingface_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(huggingface_root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(huggingface_root / "transformers")
    os.environ["TORCH_HOME"] = str(torch_root)

    return {
        "runtime_home": runtime_home,
        "cache_root": cache_root,
        "huggingface_root": huggingface_root,
        "torch_root": torch_root,
    }


@dataclass(frozen=True)
class PipelineConfig:
    """User-facing configuration for one pipeline run."""

    input_dir: Path
    output_dir: Path
    use_gpu: bool
    glob_pattern: str
    overwrite: bool
    batch_size: int
    pipeline_batch_size: int
    worker_processes: int
    gpu_devices: str
    summary_length: float
    preprocessing_model: str
    ner_model_name: str = DEFAULT_NER_MODEL
    ner_do_sentence_level: bool = True
    ner_do_postprocess: bool = True
    verbose: bool = True

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        use_gpu = getattr(args, "use_gpu", None)
        if use_gpu is None:
            use_gpu = _env_bool("OPENNYAI_USE_GPU", False)

        overwrite = getattr(args, "overwrite", None)
        if overwrite is None:
            overwrite = _env_bool("OPENNYAI_OVERWRITE", False)

        return cls(
            input_dir=_resolve_path(getattr(args, "input_dir", None), PROJECT_ROOT / "input_txt"),
            output_dir=_resolve_path(getattr(args, "output_dir", None), PROJECT_ROOT / "outputs"),
            use_gpu=use_gpu,
            glob_pattern=getattr(args, "glob_pattern", None) or os.getenv("OPENNYAI_GLOB_PATTERN", DEFAULT_GLOB_PATTERN),
            overwrite=overwrite,
            batch_size=int(getattr(args, "batch_size", None) or os.getenv("OPENNYAI_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
            pipeline_batch_size=int(
                getattr(args, "pipeline_batch_size", None)
                or os.getenv("OPENNYAI_PIPELINE_BATCH_SIZE", DEFAULT_PIPELINE_BATCH_SIZE)
            ),
            worker_processes=int(
                getattr(args, "worker_processes", None)
                or os.getenv("OPENNYAI_WORKER_PROCESSES", DEFAULT_WORKER_PROCESSES)
            ),
            gpu_devices=str(
                getattr(args, "gpu_devices", None)
                or os.getenv("OPENNYAI_GPU_DEVICES", "")
            ).strip(),
            summary_length=float(
                getattr(args, "summary_length", None) if getattr(args, "summary_length", None) is not None
                else os.getenv("OPENNYAI_SUMMARY_LENGTH", DEFAULT_SUMMARY_LENGTH)
            ),
            preprocessing_model=(
                getattr(args, "preprocessing_model", None)
                or os.getenv("OPENNYAI_PREPROCESSING_MODEL", DEFAULT_PREPROCESSING_MODEL)
            ),
        )


@dataclass(frozen=True)
class OutputLayout:
    """All output directories for the pipeline."""

    base: Path
    combined: Path
    annotations: Path
    ner: Path
    rhetorical_roles: Path
    summaries: Path
    logs: Path


def build_output_layout(output_dir: Path) -> OutputLayout:
    base = output_dir.resolve()
    return OutputLayout(
        base=base,
        combined=base / "combined",
        annotations=base / "annotations",
        ner=base / "ner",
        rhetorical_roles=base / "rhetorical_roles",
        summaries=base / "summaries",
        logs=base / "logs",
    )


def ensure_output_layout(layout: OutputLayout) -> None:
    for directory in (
        layout.base,
        layout.combined,
        layout.annotations,
        layout.ner,
        layout.rhetorical_roles,
        layout.summaries,
        layout.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def document_output_paths(layout: OutputLayout, file_id: str) -> Dict[str, Path]:
    return {
        "combined": layout.combined / f"{file_id}.json",
        "annotations": layout.annotations / f"{file_id}.json",
        "ner": layout.ner / f"{file_id}.json",
        "rhetorical_roles": layout.rhetorical_roles / f"{file_id}.json",
        "summary_json": layout.summaries / f"{file_id}.json",
        "summary_text": layout.summaries / f"{file_id}.txt",
    }
