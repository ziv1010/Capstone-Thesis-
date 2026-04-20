#!/usr/bin/env python3
"""Fetch datasets and pretrained assets for the replication workspace."""

from __future__ import annotations

import argparse
import email
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import configure_logger, ensure_dir, git_commit, now_utc_iso, sha256sum, write_json


NER_MODEL_URL = "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl"
PREAMBLE_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.2.0/"
    "en_core_web_sm-3.2.0-py3-none-any.whl"
)
RR_MODEL_URL = (
    "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/"
    "Rhetorical_Role_Benchmark/Model/model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copytree_if_needed(source: Path, destination: Path, force: bool, logger: object) -> None:
    if destination.exists() and not force:
        return
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    logger.info("Copied %s -> %s", source, destination)


def download_if_needed(url: str, destination: Path, force: bool, logger: object) -> None:
    ensure_dir(destination.parent)
    if destination.exists() and not force:
        return
    logger.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def repair_wheel_filename(downloaded_wheel: Path) -> Path:
    with zipfile.ZipFile(downloaded_wheel) as wheel_file:
        metadata_path = next(name for name in wheel_file.namelist() if name.endswith(".dist-info/METADATA"))
        wheel_path = next(name for name in wheel_file.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata_message = email.message_from_bytes(wheel_file.read(metadata_path))
        wheel_message = email.message_from_bytes(wheel_file.read(wheel_path))

    distribution_name = metadata_message.get("Name", downloaded_wheel.stem).replace("-", "_")
    version = metadata_message.get("Version", "0.0.0")
    tag = wheel_message.get_all("Tag", ["py3-none-any"])[0]
    repaired_wheel = downloaded_wheel.with_name(f"{distribution_name}-{version}-{tag}.whl")
    if repaired_wheel != downloaded_wheel:
        if repaired_wheel.exists():
            repaired_wheel.unlink()
        downloaded_wheel.replace(repaired_wheel)
    return repaired_wheel


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    logger = configure_logger("fetch_assets", root / "outputs" / "logs" / "fetch_assets.log")

    existing_dataset_root = (
        root.parent / "OpenNyai" / "ground_truth_benchmark" / "ground_truth_datasets_Opennyai"
    )
    copytree_if_needed(existing_dataset_root / "InLegalNER", root / "datasets" / "InLegalNER", args.force, logger)
    copytree_if_needed(
        existing_dataset_root / "InRhetoricalRoles",
        root / "datasets" / "InRhetoricalRoles",
        args.force,
        logger,
    )

    ner_model_path = root / "models" / "ner" / "en_legal_ner_trf-any-py3-none-any.whl"
    preamble_model_path = root / "models" / "ner" / "en_core_web_sm-3.2.0-py3-none-any.whl"
    rr_model_path = root / "models" / "rr" / "model.pt"
    download_if_needed(NER_MODEL_URL, ner_model_path, args.force, logger)
    download_if_needed(PREAMBLE_MODEL_URL, preamble_model_path, args.force, logger)
    download_if_needed(RR_MODEL_URL, rr_model_path, args.force, logger)
    ner_model_path = repair_wheel_filename(ner_model_path)

    manifest = {
        "updated_at": now_utc_iso(),
        "repos": {
            "legal_NER": {
                "path": str(root / "external" / "legal_NER"),
                "commit": git_commit(root / "external" / "legal_NER"),
            },
            "rhetorical-role-baseline": {
                "path": str(root / "external" / "rhetorical-role-baseline"),
                "commit": git_commit(root / "external" / "rhetorical-role-baseline"),
            },
        },
        "datasets": {
            "InLegalNER": {
                "path": str(root / "datasets" / "InLegalNER"),
            },
            "InRhetoricalRoles": {
                "path": str(root / "datasets" / "InRhetoricalRoles"),
            },
        },
        "models": {
            "en_legal_ner_trf": {
                "url": NER_MODEL_URL,
                "path": str(ner_model_path),
                "sha256": sha256sum(ner_model_path),
            },
            "en_core_web_sm_3_2_0": {
                "url": PREAMBLE_MODEL_URL,
                "path": str(preamble_model_path),
                "sha256": sha256sum(preamble_model_path),
            },
            "rr_model_pt": {
                "url": RR_MODEL_URL,
                "path": str(rr_model_path),
                "sha256": sha256sum(rr_model_path),
            },
        },
    }
    write_json(root / "replication_manifest.json", manifest)
    logger.info("Wrote asset manifest to %s", root / "replication_manifest.json")


if __name__ == "__main__":
    main()
