#!/usr/bin/env python3
"""Run the paper-faithful Legal NER replication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import configure_logger, write_json
from src.ner_eval import run_ner_replication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "ner")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_name", default="en_legal_ner_trf")
    parser.add_argument("--preamble_model_name", default="en_core_web_sm")
    parser.add_argument("--run_type", choices=["sent", "doc"], default="sent")
    parser.add_argument("--no_postprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger("run_ner_paper_replication", PROJECT_ROOT / "outputs" / "logs" / "ner_run.log")
    logger.info("CLI command: %s", " ".join(sys.argv))
    report = run_ner_replication(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        repo_root=(PROJECT_ROOT / "external" / "legal_NER").resolve(),
        logger=logger,
        use_gpu=args.use_gpu,
        gpu_id=args.gpu_id,
        model_name=args.model_name,
        preamble_model_name=args.preamble_model_name,
        run_type=args.run_type,
        do_postprocess=not args.no_postprocess,
    )
    write_json(args.output_dir.resolve() / "run_metadata.json", report)
    logger.info("NER replication finished.")


if __name__ == "__main__":
    main()
