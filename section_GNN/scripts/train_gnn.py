#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.metrics import (
    save_confusion_matrix_plot,
    save_split_metric_bar_plot,
    save_training_history_plot,
)
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


def _save_run_outputs(
    run_dir: Path,
    result: dict[str, object],
    cfg: dict[str, object],
    label_names: list[str],
    title_prefix: str,
) -> None:
    run_dir = ensure_dir(run_dir)
    metrics = result["metrics"]

    torch.save(result["model_state_dict"], run_dir / "model.pt")
    result["predictions_df"].to_csv(run_dir / "predictions.csv", index=False)
    dump_json(metrics, run_dir / "metrics.json")
    dump_yaml(cfg, run_dir / "run_config_snapshot.yaml")

    history = metrics.get("history", [])
    if history:
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_training_history_plot(
            history=history,
            output_path=run_dir / "training_history.png",
            best_epoch=int(metrics.get("best_epoch", 0)),
            title=f"{title_prefix} Training History",
        )

    save_split_metric_bar_plot(
        split_metrics={split_name: metrics.get(split_name, {}) for split_name in ("train", "val", "test")},
        output_path=run_dir / "split_metrics.png",
        title=f"{title_prefix} Split Metrics",
    )

    for split_name in ("train", "val", "test"):
        confusion = metrics.get(split_name, {}).get("confusion_matrix", [])
        if confusion:
            save_confusion_matrix_plot(
                confusion=confusion,
                label_names=label_names,
                output_path=run_dir / f"confusion_matrix_{split_name}.png",
                title=f"{title_prefix} {split_name.title()} Confusion Matrix",
            )


def _copy_best_run_outputs(best_run_dir: Path, target_dir: Path) -> None:
    artifact_names = [
        "model.pt",
        "predictions.csv",
        "metrics.json",
        "run_config_snapshot.yaml",
        "history.csv",
        "training_history.png",
        "split_metrics.png",
        "confusion_matrix_train.png",
        "confusion_matrix_val.png",
        "confusion_matrix_test.png",
    ]
    for artifact_name in artifact_names:
        source_path = best_run_dir / artifact_name
        if source_path.exists():
            shutil.copy2(source_path, target_dir / artifact_name)


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(args.config)
    base_seed = int(base_cfg.get("project", {}).get("seed", 42))
    repeat_runs = int(base_cfg.get("training", {}).get("repeat_runs", 1))
    seed_stride = int(base_cfg.get("training", {}).get("seed_stride", 1))

    paths_cfg = base_cfg.get("paths", {})
    graph_cache_path = Path(
        args.graph_cache
        or (
            Path(paths_cfg.get("graph_cache_dir"))
            / str(base_cfg.get("graph", {}).get("cache_name", "case_star_graph.pt"))
        )
    )
    run_dir = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "models" / args.run_name)
    log_dir = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "logs")
    logger = configure_logger("train_gnn", log_dir=log_dir)

    bundle = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    metadata = bundle["metadata"]
    label_names = list(metadata["label_names"])
    logger.info("Loaded graph bundle from %s", graph_cache_path)

    if repeat_runs <= 1:
        set_global_seed(base_seed)
        result = train_model(
            data=deepcopy(bundle["data"]),
            label_names=label_names,
            cfg=base_cfg,
            seed=base_seed,
            logger=logger,
        )
        _save_run_outputs(
            run_dir=run_dir,
            result=result,
            cfg=base_cfg,
            label_names=label_names,
            title_prefix=args.run_name,
        )
        logger.info("Training complete. Outputs saved under %s", run_dir)
        return

    repeats_root = ensure_dir(run_dir / "repeats")
    run_summaries: list[dict[str, object]] = []
    best_run_dir: Path | None = None
    best_run_label: str | None = None
    best_run_cfg: dict[str, object] | None = None
    best_val_macro_f1 = float("-inf")
    best_val_accuracy = float("-inf")

    for run_index in range(repeat_runs):
        run_seed = base_seed + run_index * seed_stride
        run_label = f"run_{run_index + 1:02d}_seed_{run_seed}"
        run_cfg = deepcopy(base_cfg)
        run_cfg.setdefault("project", {})
        run_cfg["project"]["seed"] = run_seed
        set_global_seed(run_seed)

        logger.info("Starting repeated run %d/%d with seed=%d", run_index + 1, repeat_runs, run_seed)
        result = train_model(
            data=deepcopy(bundle["data"]),
            label_names=label_names,
            cfg=run_cfg,
            seed=run_seed,
            logger=logger,
        )

        current_run_dir = ensure_dir(repeats_root / run_label)
        _save_run_outputs(
            run_dir=current_run_dir,
            result=result,
            cfg=run_cfg,
            label_names=label_names,
            title_prefix=f"{args.run_name} {run_label}",
        )

        val_metrics = result["metrics"].get("val", {})
        run_val_macro_f1 = float(val_metrics.get("macro_f1", 0.0))
        run_val_accuracy = float(val_metrics.get("accuracy", 0.0))
        run_summary = {
            "run_label": run_label,
            "seed": run_seed,
            "best_epoch": int(result["metrics"].get("best_epoch", 0)),
            "best_val_macro_f1": float(result["metrics"].get("best_val_macro_f1", run_val_macro_f1)),
            "epochs_completed": int(result["metrics"].get("epochs_completed", 0)),
            "train": result["metrics"].get("train", {}),
            "val": val_metrics,
            "test": result["metrics"].get("test", {}),
        }
        run_summaries.append(run_summary)

        is_better_run = (
            run_val_macro_f1 > best_val_macro_f1
            or (run_val_macro_f1 == best_val_macro_f1 and run_val_accuracy > best_val_accuracy)
        )
        if is_better_run:
            best_val_macro_f1 = run_val_macro_f1
            best_val_accuracy = run_val_accuracy
            best_run_dir = current_run_dir
            best_run_label = run_label
            best_run_cfg = run_cfg

    if best_run_dir is None or best_run_label is None or best_run_cfg is None:
        raise RuntimeError("Repeated training finished without selecting a best run.")

    _copy_best_run_outputs(best_run_dir=best_run_dir, target_dir=run_dir)
    dump_json(
        {
            "selection_metric": "val.macro_f1",
            "tie_breaker": "val.accuracy",
            "repeat_runs": repeat_runs,
            "base_seed": base_seed,
            "seed_stride": seed_stride,
            "best_run_label": best_run_label,
            "best_run_dir": str(best_run_dir),
            "runs": run_summaries,
        },
        run_dir / "multi_run_summary.json",
    )
    dump_yaml(best_run_cfg, run_dir / "best_run_config_snapshot.yaml")
    logger.info("Training complete. Best run=%s. Outputs saved under %s", best_run_label, run_dir)


if __name__ == "__main__":
    main()
