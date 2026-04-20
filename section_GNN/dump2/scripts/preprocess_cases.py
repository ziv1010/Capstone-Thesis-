#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.extract import clean_case_for_gnn
from src.preprocessing.loader import load_case_json
from src.utils.io import dump_json, ensure_dir, list_json_files, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess legal case JSONs for leakage-safe GNN training.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn_case_star.yaml"))
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths_cfg = cfg.get("paths", {})
    input_dir = Path(args.input_dir or paths_cfg.get("raw_json_dir"))
    cleaned_dir = ensure_dir(paths_cfg.get("cleaned_case_dir"))
    entity_dir = ensure_dir(paths_cfg.get("normalized_entity_dir"))
    audit_dir = ensure_dir(paths_cfg.get("audits_dir"))
    log_dir = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "logs")
    logger = configure_logger("preprocess_cases", log_dir=log_dir)

    files = list_json_files(input_dir, pattern=str(cfg.get("data", {}).get("file_glob", "*.json")))
    if args.limit is not None:
        files = files[: args.limit]
    logger.info("Preprocessing %d case files from %s", len(files), input_dir)

    summary = {
        "input_dir": str(input_dir),
        "num_files": len(files),
        "cases": [],
    }

    for path in files:
        case_json = load_case_json(path)
        cleaned_case = clean_case_for_gnn(case_json=case_json, file_path=path, cfg=cfg.get("preprocessing", {}))

        dump_json(cleaned_case.to_dict(), cleaned_dir / path.name)
        dump_json(
            {
                "case_id": cleaned_case.case_id,
                "entities": [entity.to_dict() for entity in cleaned_case.entities],
            },
            entity_dir / path.name,
        )
        dump_json(cleaned_case.leakage_audit, audit_dir / path.name)
        summary["cases"].append(
            {
                "case_id": cleaned_case.case_id,
                "raw_label": cleaned_case.raw_label,
                "retained_lengths": cleaned_case.leakage_audit.get("retained_texts", {}).get("retained_lengths", {}),
                "dropped_fields": cleaned_case.leakage_audit.get("fields_dropped", []),
                "dropped_annotations": len(cleaned_case.leakage_audit.get("annotations_dropped", [])),
            }
        )

    dump_json(summary, Path(paths_cfg.get("processed_dir")) / "preprocess_summary.json")
    logger.info("Wrote cleaned cases to %s", cleaned_dir)


if __name__ == "__main__":
    main()
