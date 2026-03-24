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
from src.utils.io import deep_merge_dict, dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.pipeline import build_graph_bundle, load_cleaned_cases
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph ablations for the case star GNN.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn_case_star.yaml"))
    parser.add_argument("--variants", nargs="+", default=["full_star_global"])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    set_global_seed(int(base_cfg.get("project", {}).get("seed", 42)))

    cleaned_cases = load_cleaned_cases(base_cfg.get("paths", {}).get("cleaned_case_dir"), limit=args.limit)
    logger = configure_logger(
        "run_ablation",
        log_dir=ensure_dir(Path(base_cfg.get("paths", {}).get("outputs_dir")) / "logs"),
    )

    variants_cfg = base_cfg.get("ablation", {}).get("variants", {})
    selected_variants = args.variants
    if selected_variants == ["all"]:
        selected_variants = list(variants_cfg.keys())

    summary = {}
    for variant_name in selected_variants:
        variant_override = variants_cfg.get(variant_name)
        if variant_override is None:
            raise KeyError(f"Unknown ablation variant: {variant_name}")

        cfg = deep_merge_dict(base_cfg, variant_override)
        variant_dir = ensure_dir(Path(cfg.get("paths", {}).get("outputs_dir")) / "ablations" / variant_name)
        graph_dir = ensure_dir(variant_dir / "graph_cache")
        logger.info("Running ablation variant=%s", variant_name)

        bundle = build_graph_bundle(cleaned_cases, cfg, logger=logger)
        torch.save(bundle, graph_dir / "graph_bundle.pt")
        dump_json(bundle["metadata"], graph_dir / "graph_metadata.json")

        result = train_model(
            data=bundle["data"],
            label_names=list(bundle["metadata"]["label_names"]),
            cfg=cfg,
            seed=int(cfg.get("project", {}).get("seed", 42)),
            logger=logger,
        )
        result["predictions_df"].to_csv(variant_dir / "predictions.csv", index=False)
        dump_json(result["metrics"], variant_dir / "metrics.json")
        dump_yaml(cfg, variant_dir / "run_config_snapshot.yaml")
        test_confusion = result["metrics"].get("test", {}).get("confusion_matrix", [])
        if test_confusion:
            save_confusion_matrix_plot(
                confusion=test_confusion,
                label_names=list(bundle["metadata"]["label_names"]),
                output_path=variant_dir / "confusion_matrix_test.png",
                title=f"Test Confusion Matrix: {variant_name}",
            )
        summary[variant_name] = {
            "graph_metadata": bundle["metadata"],
            "metrics": result["metrics"],
        }

    dump_json(summary, Path(base_cfg.get("paths", {}).get("outputs_dir")) / "ablation_summary.json")
    logger.info("Completed %d ablation runs", len(summary))


if __name__ == "__main__":
    main()
