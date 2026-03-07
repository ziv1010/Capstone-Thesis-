from __future__ import annotations

import argparse
from pathlib import Path

from src_ml.common.config_utils import load_config
from src_ml.common.logging_utils import make_run_name, setup_logger
from src_ml.common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run text embedding pipeline")
    p.add_argument("--config", default="configs/ml.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--text_mode", default=None)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src_ml.models.text.train_text_model import run_text_pipeline

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit
    if args.run_name:
        cfg.setdefault("text_model", {})["run_name"] = args.run_name
    if args.text_mode:
        cfg.setdefault("text_model", {})["text_mode"] = args.text_mode

    seed_everything(int(cfg.get("splits", {}).get("seed", 42)))
    out_root = Path(cfg["outputs"]["root"])
    run_name = make_run_name("text")
    logger = setup_logger("src_ml.text", out_root / "logs", run_name, level=args.log_level)
    run_text_pipeline(cfg, logger)


if __name__ == "__main__":
    main()
