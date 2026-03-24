#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.metrics import save_confusion_matrix_plot
from src.training.train import train_model
from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hetero GNN on the cached case graph.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn_case_star.yaml"))
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--run-name", default="gnn_case_star_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    paths_cfg = cfg.get("paths", {})
    graph_cache_path = Path(
        args.graph_cache
        or (Path(paths_cfg.get("graph_cache_dir")) / str(cfg.get("graph", {}).get("cache_name", "case_star_graph.pt")))
    )
    run_dir = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "models" / args.run_name)
    log_dir = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "logs")
    logger = configure_logger("train_gnn", log_dir=log_dir)

    bundle = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    data = bundle["data"]
    metadata = bundle["metadata"]
    label_names = list(metadata["label_names"])
    logger.info("Loaded graph bundle from %s", graph_cache_path)
    result = train_model(data=data, label_names=label_names, cfg=cfg, seed=seed, logger=logger)

    torch.save(result["model_state_dict"], run_dir / "model.pt")
    result["predictions_df"].to_csv(run_dir / "predictions.csv", index=False)
    dump_json(result["metrics"], run_dir / "metrics.json")
    dump_yaml(cfg, run_dir / "run_config_snapshot.yaml")

    test_confusion = result["metrics"].get("test", {}).get("confusion_matrix", [])
    if test_confusion:
        save_confusion_matrix_plot(
            confusion=test_confusion,
            label_names=label_names,
            output_path=run_dir / "confusion_matrix_test.png",
            title="Test Confusion Matrix",
        )
    logger.info("Training complete. Outputs saved under %s", run_dir)


if __name__ == "__main__":
    main()
