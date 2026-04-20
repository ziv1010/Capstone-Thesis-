from __future__ import annotations

import argparse
from pathlib import Path

from src_ml.common.config_utils import load_config
from src_ml.common.logging_utils import make_run_name, setup_logger
from src_ml.common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EDA phase")
    p.add_argument("--config", default="configs/ml.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src_ml.eda.eda_runner import run_eda

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit

    seed_everything(int(cfg.get("splits", {}).get("seed", 42)))
    out_root = Path(cfg["outputs"]["root"])
    run_name = make_run_name("eda")
    logger = setup_logger("src_ml.eda", out_root / "logs", run_name, level=args.log_level)
    run_eda(cfg, logger)


if __name__ == "__main__":
    main()
