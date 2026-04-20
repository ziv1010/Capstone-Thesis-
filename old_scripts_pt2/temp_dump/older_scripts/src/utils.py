from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LEAKAGE_PHRASES = [
    # Specific outcome phrases only — NOT bare words like "order" or "allowed"
    # that appear innocuously throughout the body text.
    "the following order",
    "following order is passed",
    "hereby ordered",
    "is hereby allowed",
    "is hereby dismissed",
    "is hereby rejected",
    "is hereby quashed",
    "is hereby granted",
    "is hereby refused",
    "petition is allowed",
    "petition is dismissed",
    "petition succeeds",
    "petition fails",
    "prayer is rejected",
    "prayer is granted",
    "prayer is dismissed",
    "rule is made absolute",
    "rule is discharged",
    "be released on bail",
    "released on bail",
    "anticipatory bail is granted",
    "anticipatory bail is rejected",
    "anticipatory bail is refused",
    "interim protection is granted",
    "not inclined to grant",
    "bail is rejected",
    "bail is granted",
    "proceedings are quashed",
    "proceedings stand quashed",
    "appeal is dismissed",
    "appeal is allowed",
    "writ is allowed",
    "writ is dismissed",
]

DEFAULT_CONFIG: dict[str, Any] = {
    "spacy_model": "en_core_web_sm",
    "hf_home": "/scratch/ziv_baretto/Thesis_Ziv/hf_cache",
    "llm_model_name": "Qwen/Qwen2.5-32B-Instruct",
    "llm_temperature": 0.0,
    "llm_max_new_tokens": 4096,
    "llm_device_map": "auto",
    "llm_torch_dtype": "bfloat16",
    "llm_trust_remote_code": False,
    "llm_cuda_visible_devices": "6,7",
    "llm_full_document_chunking": True,
    "llm_chunk_chars": 6000,
    "llm_text_field_max_chars": 6000,
    "llm_attempts_per_chunk": 6,
    "max_chars_to_send": 60000,
    "ner_max_chars": 0,
    "pdf_extraction_mode": "pymupdf",
    "leakage_phrases": DEFAULT_LEAKAGE_PHRASES,
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = dict(DEFAULT_CONFIG)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config at {path} must be a mapping")
        config.update(loaded)

    if not isinstance(config.get("leakage_phrases"), list):
        config["leakage_phrases"] = list(DEFAULT_LEAKAGE_PHRASES)

    return config


def ensure_output_dirs(out_dir: str | Path) -> tuple[Path, Path]:
    root = Path(out_dir)
    json_dir = root / "json"
    root.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    return root, json_dir


def stable_case_id(file_name: str) -> str:
    stem = Path(file_name).stem
    slug = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return slug or "case"


def dedupe_str_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def list_pdf_files(pdf_dir: str | Path) -> list[Path]:
    root = Path(pdf_dir)
    if not root.exists():
        return []
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() == ".pdf"]
    return files


def json_string(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
