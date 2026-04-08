#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

UPDATED_GRAPH_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UPDATED_GRAPH_ROOT.parent
for path in (UPDATED_GRAPH_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.training.metrics import (
    save_confusion_matrix_plot,
    save_split_metric_bar_plot,
    save_training_history_plot,
)
from src.training.train import train_model
from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.pipeline import assert_cleaned_case_integrity
from src.utils.seed import set_global_seed
from updated_graph.pipeline import build_graph_bundle, load_cleaned_cases


DEFAULT_CONFIG = (
    PROJECT_ROOT / "updated graph" / "fixed_open_pipeline" / "fixed_open_reasoning_config.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single fixed-open reasoning experiment on the first N valid cleaned cases "
            "using the normal config-defined train/val/test split."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--limit", type=int, default=2105)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-on-invalid-cases",
        action="store_true",
        help="Stop immediately if a cleaned case fails integrity checks instead of skipping it.",
    )
    return parser.parse_args()


def _save_run_outputs(
    run_dir: Path,
    result: dict[str, Any],
    cfg: dict[str, Any],
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


def _save_graph_artifacts(bundle: dict[str, Any], cfg: dict[str, Any], graph_cache_dir: Path) -> Path:
    metadata = bundle["metadata"]
    effective_graph_cfg = metadata.get("effective_graph_config", {})
    graph_cache_path = graph_cache_dir / str(
        effective_graph_cfg.get("cache_name", "reasoning_focused_case_star_graph.pt")
    )

    torch.save(bundle, graph_cache_path)
    dump_json(metadata, graph_cache_dir / "graph_metadata.reasoning_focused.json")
    dump_json(metadata.get("node_mappings", {}), graph_cache_dir / "node_mappings.reasoning_focused.json")
    dump_json(metadata.get("relation_mappings", []), graph_cache_dir / "relation_mappings.reasoning_focused.json")
    dump_json(metadata.get("split_assignments", {}), graph_cache_dir / "split_assignments.reasoning_focused.json")
    dump_json(metadata.get("debug_samples", []), graph_cache_dir / "graph_debug_samples.reasoning_focused.json")
    dump_yaml(cfg, graph_cache_dir / "graph_config_snapshot.reasoning_focused.yaml")
    dump_yaml(effective_graph_cfg, graph_cache_dir / "effective_graph_config.reasoning_focused.yaml")
    return graph_cache_path


def _collect_valid_cases(
    cleaned_cases: list[Any],
    limit: int,
    preprocessing_cfg: dict[str, Any],
    logger: Any,
    fail_on_invalid_cases: bool,
) -> tuple[list[Any], list[dict[str, str]]]:
    selected_cases: list[Any] = []
    invalid_cases: list[dict[str, str]] = []

    for cleaned_case in cleaned_cases:
        try:
            assert_cleaned_case_integrity(cleaned_case, preprocessing_cfg)
        except AssertionError as exc:
            invalid_record = {
                "case_id": str(cleaned_case.case_id),
                "file_name": str(cleaned_case.file_name),
                "reason": str(exc),
            }
            invalid_cases.append(invalid_record)
            if fail_on_invalid_cases:
                raise
            logger.warning("Skipping invalid cleaned case %s: %s", cleaned_case.case_id, exc)
            continue

        selected_cases.append(cleaned_case)
        if len(selected_cases) >= limit:
            break

    if len(selected_cases) < limit:
        raise ValueError(f"Requested {limit} valid cases, but only found {len(selected_cases)}.")

    return selected_cases, invalid_cases


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be > 0.")

    base_cfg = load_yaml(args.config)
    base_seed = int(base_cfg.get("project", {}).get("seed", 42))
    experiment_name = args.experiment_name or f"fixed_open_reasoning_limit_{args.limit}"
    run_name = args.run_name or experiment_name

    paths_cfg = base_cfg.get("paths", {})
    cleaned_dir = Path(paths_cfg.get("cleaned_case_dir"))
    outputs_root = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "dataset_size_sweep" / experiment_name)
    graph_cache_dir = ensure_dir(Path(paths_cfg.get("graph_cache_dir")) / "dataset_size_sweep" / experiment_name)
    log_dir = ensure_dir(outputs_root / "logs")
    logger = configure_logger("fixed_open_limited_experiment", log_dir=log_dir)

    all_cleaned_cases = load_cleaned_cases(cleaned_dir)
    logger.info("Loaded %d cleaned cases from %s", len(all_cleaned_cases), cleaned_dir)

    selected_cases, invalid_cases = _collect_valid_cases(
        cleaned_cases=all_cleaned_cases,
        limit=args.limit,
        preprocessing_cfg=base_cfg.get("preprocessing", {}),
        logger=logger,
        fail_on_invalid_cases=args.fail_on_invalid_cases,
    )
    logger.info("Selected %d valid cleaned cases for the experiment", len(selected_cases))

    if invalid_cases:
        dump_json(invalid_cases, outputs_root / "invalid_cases_skipped.json")

    experiment_cfg = deepcopy(base_cfg)
    experiment_cfg.setdefault("project", {})
    experiment_cfg["project"]["name"] = experiment_name
    experiment_cfg["project"]["seed"] = base_seed
    experiment_cfg.setdefault("paths", {})
    experiment_cfg["paths"]["graph_cache_dir"] = str(graph_cache_dir)
    experiment_cfg["paths"]["outputs_dir"] = str(outputs_root)
    experiment_cfg.setdefault("data", {})
    experiment_cfg["data"]["limit"] = args.limit
    experiment_cfg["data"]["selected_case_dir"] = str(cleaned_dir)
    experiment_cfg.setdefault("graph", {})
    original_cache_name = str(
        experiment_cfg["graph"].get("cache_name", "reasoning_focused_case_star_graph.pt")
    )
    experiment_cfg["graph"]["cache_name"] = f"{Path(original_cache_name).stem}.limit_{args.limit}.pt"

    selection_manifest = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "config_path": str(Path(args.config).resolve()),
        "cleaned_dir": str(cleaned_dir),
        "requested_limit": args.limit,
        "selected_case_count": len(selected_cases),
        "selected_case_ids": [case.case_id for case in selected_cases],
        "invalid_cases_skipped": len(invalid_cases),
        "split_cfg": experiment_cfg.get("splits", {}),
    }
    dump_json(selection_manifest, outputs_root / "selection_manifest.json")

    if args.dry_run:
        logger.info("Dry run complete. Selection manifest saved to %s", outputs_root / "selection_manifest.json")
        return

    set_global_seed(base_seed)
    bundle = build_graph_bundle(selected_cases, experiment_cfg, logger=logger)
    graph_cache_path = _save_graph_artifacts(bundle=bundle, cfg=experiment_cfg, graph_cache_dir=graph_cache_dir)

    run_dir = ensure_dir(outputs_root / "models" / run_name)
    label_names = list(bundle["metadata"]["label_names"])
    set_global_seed(base_seed)
    result = train_model(
        data=deepcopy(bundle["data"]),
        label_names=label_names,
        cfg=experiment_cfg,
        seed=base_seed,
        logger=logger,
    )
    _save_run_outputs(
        run_dir=run_dir,
        result=result,
        cfg=experiment_cfg,
        label_names=label_names,
        title_prefix=run_name,
    )

    summary = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "requested_limit": args.limit,
        "selected_case_count": len(selected_cases),
        "graph_cache_path": str(graph_cache_path),
        "run_dir": str(run_dir),
        "case_split_counts": bundle["metadata"].get("case_split_counts", {}),
        "label_names": label_names,
        "train_metrics": result["metrics"].get("train", {}),
        "val_metrics": result["metrics"].get("val", {}),
        "test_metrics": result["metrics"].get("test", {}),
        "best_epoch": int(result["metrics"].get("best_epoch", 0)),
        "best_val_macro_f1": float(result["metrics"].get("best_val_macro_f1", 0.0)),
    }
    dump_json(summary, outputs_root / "summary.json")
    logger.info("Experiment complete. Outputs saved under %s", outputs_root)


if __name__ == "__main__":
    main()
