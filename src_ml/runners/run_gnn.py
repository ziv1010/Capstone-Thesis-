from __future__ import annotations

import argparse
from pathlib import Path

from src_ml.common.config_utils import load_config
from src_ml.common.logging_utils import make_run_name, setup_logger
from src_ml.common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GNN pipeline")
    p.add_argument("--config", default="configs/ml.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_pyg", action="store_true")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src_ml.models.gnn.train_gnn import run_gnn_pipeline

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit
    if args.run_name:
        cfg.setdefault("gnn", {})["run_name"] = args.run_name
    if args.no_pyg:
        cfg.setdefault("gnn", {})["use_pyg"] = False

    seed_everything(int(cfg.get("splits", {}).get("seed", 42)))
    out_root = Path(cfg["outputs"]["root"])
    run_name = make_run_name("gnn")
    logger = setup_logger("src_ml.gnn", out_root / "logs", run_name, level=args.log_level)
    run_gnn_pipeline(cfg, logger)


if __name__ == "__main__":
    main()
