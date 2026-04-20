from __future__ import annotations

import argparse
from pathlib import Path

from src_ml.common.config_utils import load_config
from src_ml.common.logging_utils import make_run_name, setup_logger
from src_ml.common.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM + RAG pipeline")
    p.add_argument("--config", default="configs/ml.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from src_ml.models.llm_rag.eval_rag import run_rag_pipeline

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg.setdefault("dataset", {})["limit"] = args.limit
    if args.top_k is not None:
        cfg.setdefault("rag", {})["top_k"] = args.top_k

    seed_everything(int(cfg.get("splits", {}).get("seed", 42)))
    out_root = Path(cfg["outputs"]["root"])
    run_name = make_run_name("rag")
    logger = setup_logger("src_ml.rag", out_root / "logs", run_name, level=args.log_level)
    run_rag_pipeline(cfg, logger)


if __name__ == "__main__":
    main()
