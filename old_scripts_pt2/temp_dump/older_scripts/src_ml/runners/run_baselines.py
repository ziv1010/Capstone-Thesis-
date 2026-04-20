from __future__ import annotations

import argparse
from pathlib import Path

from src_ml.common.config_utils import load_config
from src_ml.common.logging_utils import make_run_name, setup_logger
from src_ml.common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline models")
    p.add_argument("--config", default="configs/ml.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log_level", default="INFO")
    p.add_argument("--force_rebuild_splits", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src_ml.baselines.baseline_runner import run_baselines

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit
    if args.force_rebuild_splits:
        cfg.setdefault("splits", {})["force_rebuild"] = True

    seed_everything(int(cfg.get("splits", {}).get("seed", 42)))
    out_root = Path(cfg["outputs"]["root"])
    run_name = make_run_name("baselines")
    logger = setup_logger("src_ml.baselines", out_root / "logs", run_name, level=args.log_level)
    run_baselines(cfg, logger)


if __name__ == "__main__":
    main()
