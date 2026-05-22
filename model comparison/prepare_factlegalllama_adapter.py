#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


REPO_ID = "L-NLProc/TathyaNyaya-and-FactLegalLlama-Large-Language-Models-Based-Models"
PREDICTION_ONLY_ARCHIVE = "Prediction-Only SFT.zip"
PREDICTION_EXPLANATION_ARCHIVE = "Prediction_Explanation SFT.zip"

CHECKPOINTS = {
    "nyaya_facts_single": {
        "archive": PREDICTION_ONLY_ARCHIVE,
        "nested_zip": "Prediction-Only SFT/SFT_Over_NyayaFacts_Single.zip",
    },
    "nyaya_facts_multi": {
        "archive": PREDICTION_ONLY_ARCHIVE,
        "nested_zip": "Prediction-Only SFT/SFT_Over_NyayaFacts_Multi.zip",
    },
    "nyaya_scrape_single": {
        "archive": PREDICTION_ONLY_ARCHIVE,
        "nested_zip": "Prediction-Only SFT/SFT_Over_NyayaScrape_Single.zip",
    },
    "nyaya_scrape_multi": {
        "archive": PREDICTION_ONLY_ARCHIVE,
        "nested_zip": "Prediction-Only SFT/SFT_Over_NyayaScrape_Multi.zip",
    },
    "nyaya_simplify": {
        "archive": PREDICTION_ONLY_ARCHIVE,
        "nested_zip": "Prediction-Only SFT/SFT_Over_NyayaSimplify.zip",
    },
}

SKIP_SUFFIXES = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "training_args.bin",
    "trainer_state.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract a TathyaNyaya/FactLegalLlama PEFT adapter checkpoint."
    )
    parser.add_argument("--checkpoint", choices=sorted(CHECKPOINTS), default="nyaya_facts_single")
    parser.add_argument("--output-root", type=Path, default=Path("models/factlegalllama"))
    parser.add_argument("--cache-dir", default="/scratch/ziv_baretto/Thesis_Ziv/hf_cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-path", action="store_true")
    parser.add_argument("--print-base-model", action="store_true")
    return parser.parse_args()


def should_extract(member_name: str) -> bool:
    return not member_name.endswith(SKIP_SUFFIXES) and not member_name.endswith("/")


def adapter_dir(output_root: Path, checkpoint: str) -> Path:
    return output_root / checkpoint


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def prepare_adapter(
    checkpoint: str,
    output_root: Path,
    cache_dir: str,
    force: bool = False,
) -> Path:
    spec = CHECKPOINTS[checkpoint]
    destination = adapter_dir(output_root, checkpoint).resolve()
    adapter_config_path = destination / "adapter_config.json"
    adapter_model_path = destination / "adapter_model.safetensors"
    if not force and adapter_config_path.exists() and adapter_model_path.exists():
        return destination

    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    archive_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=spec["archive"],
            cache_dir=cache_dir,
        )
    )
    with zipfile.ZipFile(archive_path) as outer_zip:
        nested_bytes = outer_zip.read(spec["nested_zip"])

    extracted_files: list[str] = []
    checkpoint_prefix: str | None = None
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested_zip:
        adapter_configs = [
            item.filename
            for item in nested_zip.infolist()
            if item.filename.endswith("adapter_config.json")
        ]
        if not adapter_configs:
            raise RuntimeError(f"No adapter_config.json found inside {spec['nested_zip']}")
        checkpoint_prefix = adapter_configs[0].rsplit("/", 1)[0] + "/"

        for item in nested_zip.infolist():
            if not item.filename.startswith(checkpoint_prefix) or not should_extract(item.filename):
                continue
            relative_name = item.filename[len(checkpoint_prefix) :]
            target_path = destination / relative_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with nested_zip.open(item) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted_files.append(relative_name)

    adapter_config = read_json(adapter_config_path)
    manifest = {
        "repo_id": REPO_ID,
        "checkpoint": checkpoint,
        "archive": spec["archive"],
        "nested_zip": spec["nested_zip"],
        "checkpoint_prefix": checkpoint_prefix,
        "adapter_dir": str(destination),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "extracted_files": sorted(extracted_files),
        "skipped_suffixes": list(SKIP_SUFFIXES),
    }
    with (destination / "extraction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return destination


def main() -> None:
    args = parse_args()
    path = prepare_adapter(
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        force=args.force,
    )
    if args.print_base_model:
        print(read_json(path / "adapter_config.json").get("base_model_name_or_path", ""))
    elif args.print_path:
        print(path)
    else:
        print(json.dumps(read_json(path / "extraction_manifest.json"), indent=2))


if __name__ == "__main__":
    main()
