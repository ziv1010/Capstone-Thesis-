#!/usr/bin/env python3
"""Run the paper-faithful rhetorical-role replication."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import configure_logger, write_json
from src.rr_eval import run_rr_replication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "rr")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = configure_logger("run_rr_paper_replication", PROJECT_ROOT / "outputs" / "logs" / "rr_run.log")
    logger.info("CLI command: %s", " ".join(sys.argv))
    report = run_rr_replication(
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        repo_root=(PROJECT_ROOT / "external" / "rhetorical-role-baseline").resolve(),
        model_path=(PROJECT_ROOT / "models" / "rr" / "model.pt").resolve(),
        runtime_dir=(PROJECT_ROOT / "outputs" / "rr" / "runtime").resolve(),
        logger=logger,
        device_name=args.device,
    )
    write_json(args.output_dir.resolve() / "run_metadata.json", report)
    logger.info("RR replication finished.")


if __name__ == "__main__":
    main()
