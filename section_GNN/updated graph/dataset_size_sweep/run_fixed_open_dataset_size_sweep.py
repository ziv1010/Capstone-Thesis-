#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

UPDATED_GRAPH_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UPDATED_GRAPH_ROOT.parent
for path in (UPDATED_GRAPH_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.graph.global_graph_builder import merge_case_graphs_into_global_graph
from src.graph.pyg_builder import build_pyg_heterodata
from src.training.dataset import PreparedCases, prepare_cases_for_task
from src.training.metrics import (
    save_confusion_matrix_plot,
    save_split_metric_bar_plot,
    save_training_history_plot,
)
from src.training.train import train_model
from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.pipeline import assert_cleaned_case_integrity, load_cleaned_cases
from src.utils.seed import set_global_seed
from updated_graph.case_star_builder import build_case_star_graph
from updated_graph.reasoning_graph_policy import apply_reasoning_graph_policy


DEFAULT_CONFIG = (
    PROJECT_ROOT / "updated graph" / "fixed_open_pipeline" / "fixed_open_reasoning_config.yaml"
)


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_key: str
    requested_size: str
    selected_dev_count: int
    train_count: int
    val_count: int
    test_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed-open reasoning graph size sweep with a fixed last-N test set "
            "and three dataset sizes."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-sizes", nargs="+", default=["2500", "5000", "all"])
    parser.add_argument("--holdout-test-count", type=int, default=1000)
    parser.add_argument("--experiment-name", default="fixed_open_reasoning_dataset_size_sweep")
    parser.add_argument(
        "--fail-on-invalid-cases",
        action="store_true",
        help="Stop immediately if a cleaned case fails the leakage/integrity checks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute and save the sweep plan without building graphs or training models.",
    )
    return parser.parse_args()


def _bounded_val_count(total_count: int, desired_val_count: int) -> int:
    if total_count < 2:
        raise ValueError("Need at least 2 cases to create a train/val split.")
    return max(1, min(total_count - 1, desired_val_count))


def _resolve_experiment_plans(
    train_sizes: list[str],
    dev_pool_size: int,
    holdout_test_count: int,
    splits_cfg: dict[str, Any],
) -> list[ExperimentPlan]:
    train_ratio = float(splits_cfg.get("train_size", 0.7))
    val_ratio = float(splits_cfg.get("val_size", 0.15))
    test_ratio = float(splits_cfg.get("test_size", 0.15))
    if train_ratio <= 0:
        raise ValueError("splits.train_size must be > 0.")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Base split ratios must sum to 1.0.")

    dev_ratio = train_ratio + val_ratio
    if dev_ratio <= 0:
        raise ValueError("train_size + val_size must be > 0.")

    plans: list[ExperimentPlan] = []
    for size_token in train_sizes:
        normalized = str(size_token).strip().lower()
        if normalized == "all":
            selected_dev_count = dev_pool_size
            desired_val_count = int(round(selected_dev_count * (val_ratio / dev_ratio)))
            val_count = _bounded_val_count(selected_dev_count, desired_val_count)
            train_count = selected_dev_count - val_count
            experiment_key = "train_all_remaining"
        else:
            train_count = int(normalized)
            if train_count <= 0:
                raise ValueError(f"Invalid train size: {size_token}")
            desired_val_count = int(round(train_count * (val_ratio / train_ratio)))
            val_count = _bounded_val_count(train_count + max(desired_val_count, 1), desired_val_count)
            selected_dev_count = train_count + val_count
            if selected_dev_count > dev_pool_size:
                raise ValueError(
                    f"Requested train size {train_count} needs {selected_dev_count} dev cases "
                    f"(train + val), but only {dev_pool_size} remain after the fixed test split."
                )
            experiment_key = f"train_{train_count}"

        plans.append(
            ExperimentPlan(
                experiment_key=experiment_key,
                requested_size=size_token,
                selected_dev_count=selected_dev_count,
                train_count=train_count,
                val_count=val_count,
                test_count=holdout_test_count,
            )
        )

    return plans


def _filter_valid_cases(
    cleaned_cases: list[Any],
    cfg: dict[str, Any],
    logger: Any,
    fail_on_invalid_cases: bool,
) -> tuple[list[Any], list[dict[str, str]]]:
    valid_cases: list[Any] = []
    invalid_cases: list[dict[str, str]] = []
    preprocessing_cfg = cfg.get("preprocessing", {})

    for case in cleaned_cases:
        try:
            assert_cleaned_case_integrity(case, preprocessing_cfg)
        except AssertionError as exc:
            record = {
                "case_id": str(case.case_id),
                "file_name": str(case.file_name),
                "reason": str(exc),
            }
            invalid_cases.append(record)
            if fail_on_invalid_cases:
                raise
            logger.warning("Skipping invalid cleaned case %s: %s", case.case_id, exc)
            continue
        valid_cases.append(case)

    return valid_cases, invalid_cases


def _make_train_val_assignments(
    selected_cases: list[Any],
    selected_y: np.ndarray,
    plan: ExperimentPlan,
    random_state: int,
    stratify: bool,
) -> dict[str, str]:
    case_ids = [case.case_id for case in selected_cases]
    indices = np.arange(len(selected_cases))
    train_size = int(plan.train_count)
    val_size = int(plan.val_count)

    if len(selected_cases) != plan.selected_dev_count:
        raise ValueError(
            f"Selected case count mismatch for {plan.experiment_key}: "
            f"expected {plan.selected_dev_count}, got {len(selected_cases)}"
        )

    if train_size + val_size != len(selected_cases):
        raise ValueError(
            f"Train/val counts do not cover selected cases for {plan.experiment_key}: "
            f"{train_size} + {val_size} != {len(selected_cases)}"
        )

    stratify_labels = selected_y if stratify and len(np.unique(selected_y)) > 1 else None
    try:
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_size,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
    except ValueError:
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_size,
            test_size=val_size,
            random_state=random_state,
            stratify=None,
        )

    assignments = {case_ids[idx]: "train" for idx in train_idx}
    assignments.update({case_ids[idx]: "val" for idx in val_idx})
    return assignments


def _build_bundle_with_fixed_splits(
    prepared: PreparedCases,
    selected_cases: list[Any],
    selected_y: np.ndarray,
    test_cases: list[Any],
    test_y: np.ndarray,
    split_assignments: dict[str, str],
    cfg: dict[str, Any],
    logger: Any,
) -> dict[str, Any]:
    experiment_cases = selected_cases + test_cases
    experiment_y = np.concatenate([selected_y, test_y], axis=0).astype(np.int64)
    graph_cfg = apply_reasoning_graph_policy(cfg.get("graph", {}))
    case_graphs = [build_case_star_graph(case, graph_cfg) for case in experiment_cases]
    global_graph = merge_case_graphs_into_global_graph(case_graphs)
    data, pyg_metadata = build_pyg_heterodata(
        global_graph=global_graph,
        cleaned_cases=experiment_cases,
        labels=experiment_y,
        label_names=prepared.label_names,
        split_assignments=split_assignments,
        cfg={**cfg, "graph": graph_cfg},
        logger=logger,
    )

    debug_sample_size = int(graph_cfg.get("debug_sample_size", 3))
    debug_samples = []
    for case, case_graph in zip(experiment_cases[:debug_sample_size], case_graphs[:debug_sample_size]):
        debug_samples.append(
            {
                "case_id": case.case_id,
                "file_name": case.file_name,
                "retained_text": case.texts,
                "dropped_fields": case.leakage_audit.get("fields_dropped", []),
                "dropped_annotations": len(case.leakage_audit.get("annotations_dropped", [])),
                "node_counts": case_graph.metadata.get("node_count_by_type", {}),
                "edge_counts": case_graph.metadata.get("edge_count_by_type", {}),
            }
        )

    label_distribution: dict[str, int] = {}
    for case in experiment_cases:
        label_key = str(case.raw_label)
        label_distribution[label_key] = label_distribution.get(label_key, 0) + 1

    return {
        "data": data,
        "metadata": {
            **pyg_metadata,
            "label_names": prepared.label_names,
            "dropped_case_ids": prepared.dropped_case_ids,
            "split_assignments": split_assignments,
            "case_graph_count": len(case_graphs),
            "raw_label_distribution_after_filtering": dict(sorted(label_distribution.items())),
            "global_node_stats": global_graph["node_stats"],
            "global_relation_stats": global_graph["relation_stats"],
            "debug_samples": debug_samples,
            "case_summaries": global_graph["case_summaries"],
            "effective_graph_config": graph_cfg,
        },
    }


def _save_graph_artifacts(
    bundle: dict[str, Any],
    graph_cache_dir: Path,
    cfg: dict[str, Any],
    experiment_key: str,
) -> Path:
    metadata = bundle["metadata"]
    effective_graph_cfg = dict(metadata.get("effective_graph_config", {}))
    base_cache_name = str(effective_graph_cfg.get("cache_name", "reasoning_focused_case_star_graph.pt"))
    cache_path = graph_cache_dir / f"{Path(base_cache_name).stem}.{experiment_key}.pt"

    torch.save(bundle, cache_path)
    dump_json(metadata, graph_cache_dir / "graph_metadata.reasoning_focused.json")
    dump_json(metadata.get("node_mappings", {}), graph_cache_dir / "node_mappings.reasoning_focused.json")
    dump_json(metadata.get("relation_mappings", []), graph_cache_dir / "relation_mappings.reasoning_focused.json")
    dump_json(metadata.get("split_assignments", {}), graph_cache_dir / "split_assignments.reasoning_focused.json")
    dump_json(metadata.get("debug_samples", []), graph_cache_dir / "graph_debug_samples.reasoning_focused.json")
    dump_yaml(cfg, graph_cache_dir / "graph_config_snapshot.reasoning_focused.yaml")
    dump_yaml(effective_graph_cfg, graph_cache_dir / "effective_graph_config.reasoning_focused.yaml")
    return cache_path


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


def _flatten_summary(
    plan: ExperimentPlan,
    metrics: dict[str, Any],
    graph_cache_path: Path,
    run_dir: Path,
) -> dict[str, Any]:
    train_metrics = metrics.get("train", {})
    val_metrics = metrics.get("val", {})
    test_metrics = metrics.get("test", {})
    return {
        "experiment_key": plan.experiment_key,
        "requested_size": plan.requested_size,
        "selected_dev_count": plan.selected_dev_count,
        "train_count": plan.train_count,
        "val_count": plan.val_count,
        "test_count": plan.test_count,
        "best_epoch": int(metrics.get("best_epoch", 0)),
        "best_val_macro_f1": float(metrics.get("best_val_macro_f1", 0.0)),
        "train_macro_f1": float(train_metrics.get("macro_f1", 0.0)),
        "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
        "test_macro_f1": float(test_metrics.get("macro_f1", 0.0)),
        "train_accuracy": float(train_metrics.get("accuracy", 0.0)),
        "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
        "graph_cache_path": str(graph_cache_path),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    base_seed = int(cfg.get("project", {}).get("seed", 42))
    random_state = int(cfg.get("splits", {}).get("random_state", base_seed))
    stratify = bool(cfg.get("splits", {}).get("stratify", True))

    paths_cfg = cfg.get("paths", {})
    cleaned_dir = Path(paths_cfg.get("cleaned_case_dir"))
    outputs_root = ensure_dir(Path(paths_cfg.get("outputs_dir")) / "dataset_size_sweep" / args.experiment_name)
    graphs_root = ensure_dir(Path(paths_cfg.get("graph_cache_dir")) / "dataset_size_sweep" / args.experiment_name)
    log_dir = ensure_dir(outputs_root / "logs")
    logger = configure_logger("dataset_size_sweep", log_dir=log_dir)

    raw_cleaned_cases = load_cleaned_cases(cleaned_dir)
    logger.info("Loaded %d cleaned cases from %s", len(raw_cleaned_cases), cleaned_dir)

    valid_cleaned_cases, invalid_cases = _filter_valid_cases(
        cleaned_cases=raw_cleaned_cases,
        cfg=cfg,
        logger=logger,
        fail_on_invalid_cases=args.fail_on_invalid_cases,
    )
    if invalid_cases:
        dump_json(invalid_cases, outputs_root / "invalid_cases.json")
        logger.warning("Dropped %d invalid cleaned cases before the sweep.", len(invalid_cases))

    prepared = prepare_cases_for_task(valid_cleaned_cases, cfg.get("labels", {}))
    if prepared.dropped_case_ids:
        dump_json(prepared.dropped_case_ids, outputs_root / "label_dropped_case_ids.json")
        logger.warning("Dropped %d cases because their labels are excluded by the config.", len(prepared.dropped_case_ids))

    if len(prepared.cases) <= args.holdout_test_count:
        raise ValueError(
            f"Need more than {args.holdout_test_count} usable cases, but only {len(prepared.cases)} remain."
        )

    dev_cases = prepared.cases[:-args.holdout_test_count]
    dev_y = prepared.y[:-args.holdout_test_count]
    test_cases = prepared.cases[-args.holdout_test_count :]
    test_y = prepared.y[-args.holdout_test_count :]

    plans = _resolve_experiment_plans(
        train_sizes=args.train_sizes,
        dev_pool_size=len(dev_cases),
        holdout_test_count=args.holdout_test_count,
        splits_cfg=cfg.get("splits", {}),
    )

    dump_json(
        {
            "config_path": str(Path(args.config).resolve()),
            "cleaned_dir": str(cleaned_dir),
            "raw_cleaned_case_count": len(raw_cleaned_cases),
            "invalid_case_count": len(invalid_cases),
            "usable_case_count": len(prepared.cases),
            "dev_pool_count": len(dev_cases),
            "test_pool_count": len(test_cases),
            "train_sizes": args.train_sizes,
            "plans": [plan.__dict__ for plan in plans],
        },
        outputs_root / "sweep_plan.json",
    )

    logger.info(
        "Prepared sweep with %d usable cases: dev_pool=%d test_pool=%d",
        len(prepared.cases),
        len(dev_cases),
        len(test_cases),
    )
    for plan in plans:
        logger.info(
            "Plan %s: selected_dev=%d train=%d val=%d test=%d",
            plan.experiment_key,
            plan.selected_dev_count,
            plan.train_count,
            plan.val_count,
            plan.test_count,
        )

    if args.dry_run:
        logger.info("Dry run complete. Saved sweep plan to %s", outputs_root / "sweep_plan.json")
        return

    summaries: list[dict[str, Any]] = []
    label_names = list(prepared.label_names)

    for plan in plans:
        logger.info("Starting experiment %s", plan.experiment_key)
        selected_cases = dev_cases[: plan.selected_dev_count]
        selected_y = dev_y[: plan.selected_dev_count]
        split_assignments = _make_train_val_assignments(
            selected_cases=selected_cases,
            selected_y=selected_y,
            plan=plan,
            random_state=random_state,
            stratify=stratify,
        )
        split_assignments.update({case.case_id: "test" for case in test_cases})

        experiment_cfg = deepcopy(cfg)
        experiment_cfg.setdefault("project", {})
        experiment_cfg["project"]["seed"] = base_seed
        experiment_cfg.setdefault("splits", {})
        experiment_cfg["splits"].update(
            {
                "mode": "fixed_last_n_test_prefix_dev",
                "holdout_test_count": args.holdout_test_count,
                "requested_train_size": plan.requested_size,
                "selected_dev_count": plan.selected_dev_count,
                "train_count": plan.train_count,
                "val_count": plan.val_count,
                "random_state": random_state,
                "stratify": stratify,
            }
        )

        experiment_graph_dir = ensure_dir(graphs_root / plan.experiment_key)
        experiment_run_dir = ensure_dir(outputs_root / plan.experiment_key)
        experiment_cfg.setdefault("paths", {})
        experiment_cfg["paths"]["graph_cache_dir"] = str(experiment_graph_dir)
        experiment_cfg["paths"]["outputs_dir"] = str(experiment_run_dir)

        set_global_seed(base_seed)
        bundle = _build_bundle_with_fixed_splits(
            prepared=prepared,
            selected_cases=selected_cases,
            selected_y=selected_y,
            test_cases=test_cases,
            test_y=test_y,
            split_assignments=split_assignments,
            cfg=experiment_cfg,
            logger=logger,
        )
        graph_cache_path = _save_graph_artifacts(
            bundle=bundle,
            graph_cache_dir=experiment_graph_dir,
            cfg=experiment_cfg,
            experiment_key=plan.experiment_key,
        )

        dump_json(
            {
                "experiment_key": plan.experiment_key,
                "requested_size": plan.requested_size,
                "selected_case_ids": [case.case_id for case in selected_cases],
                "test_case_ids": [case.case_id for case in test_cases],
                "split_assignments": split_assignments,
            },
            experiment_run_dir / "split_manifest.json",
        )

        set_global_seed(base_seed)
        result = train_model(
            data=deepcopy(bundle["data"]),
            label_names=label_names,
            cfg=experiment_cfg,
            seed=base_seed,
            logger=logger,
        )
        _save_run_outputs(
            run_dir=experiment_run_dir,
            result=result,
            cfg=experiment_cfg,
            label_names=label_names,
            title_prefix=plan.experiment_key,
        )

        summary_row = _flatten_summary(
            plan=plan,
            metrics=result["metrics"],
            graph_cache_path=graph_cache_path,
            run_dir=experiment_run_dir,
        )
        summaries.append(summary_row)
        logger.info(
            "Finished %s with val_macro_f1=%.4f test_macro_f1=%.4f",
            plan.experiment_key,
            summary_row["val_macro_f1"],
            summary_row["test_macro_f1"],
        )

    dump_json(summaries, outputs_root / "summary.json")
    pd.DataFrame(summaries).to_csv(outputs_root / "summary.csv", index=False)
    logger.info("Sweep complete. Summary saved under %s", outputs_root)


if __name__ == "__main__":
    main()
