#!/usr/bin/env python3
"""
Stratified K-fold cross-validation for the Hetero-GNN.

Each fold uses an 80/20 train/test split (controlled by --k, default 5).
A small fraction of each fold's train set is held out as internal validation
for best-checkpoint selection during training.

Hashing baseline
----------------
Point --graph-cache at a graph built with the hashing encoder:

    python kfold_cv.py --config X.yaml \\
        --graph-cache /path/to/hashing_graph.pt \\
        --run-name family_matrimonial_hashing_kfold

Multi-GPU (parallel folds)
--------------------------
Run each fold on a separate GPU simultaneously, then aggregate:

    CUDA_VISIBLE_DEVICES=0 python kfold_cv.py --config X.yaml --run-name NAME --fold 0 &
    CUDA_VISIBLE_DEVICES=1 python kfold_cv.py --config X.yaml --run-name NAME --fold 1 &
    CUDA_VISIBLE_DEVICES=2 python kfold_cv.py --config X.yaml --run-name NAME --fold 2 &
    CUDA_VISIBLE_DEVICES=3 python kfold_cv.py --config X.yaml --run-name NAME --fold 3 &
    wait
    # fold 4 can reuse any free GPU, or run it on a 5th GPU
    CUDA_VISIBLE_DEVICES=0 python kfold_cv.py --config X.yaml --run-name NAME --fold 4
    # once all folds are done, compute the aggregate summary:
    python kfold_cv.py --config X.yaml --run-name NAME --aggregate-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.metrics import save_confusion_matrix_plot
from src.training.train import train_model
from src.utils.io import dump_json, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified K-fold CV for Hetero-GNN.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument(
        "--graph-cache", default=None,
        help="Override graph cache .pt path (e.g. for hashing baseline)",
    )
    p.add_argument("--run-name", default="kfold_run")
    p.add_argument(
        "--k", type=int, default=5,
        help="Number of folds (default 5 → 80%% train / 20%% test per fold)",
    )
    p.add_argument(
        "--fold", type=int, default=None,
        help="Run only this fold index (0-based). Omit to run all folds sequentially.",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of train split reserved as internal val for best-checkpoint "
             "selection (default 0.1). Does not affect the 80/20 train/test ratio.",
    )
    p.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip training; load saved per-fold results and recompute aggregate summary.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold_masks(
    n_cases: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    val_fraction: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_mask, val_mask, test_mask) boolean tensors for one fold."""
    if val_fraction > 0 and len(train_idx) > 1:
        n_val = max(1, int(len(train_idx) * val_fraction))
        perm = rng.permutation(len(train_idx))
        val_idx = train_idx[perm[:n_val]]
        actual_train_idx = train_idx[perm[n_val:]]
    else:
        val_idx = np.array([], dtype=np.int64)
        actual_train_idx = train_idx

    train_mask = torch.zeros(n_cases, dtype=torch.bool)
    val_mask = torch.zeros(n_cases, dtype=torch.bool)
    test_mask = torch.zeros(n_cases, dtype=torch.bool)
    train_mask[actual_train_idx] = True
    if len(val_idx):
        val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _aggregate(run_dir: Path, k: int, run_name: str, graph_cache: str) -> dict:
    """Load per-fold summaries and compute mean ± std across completed folds."""
    folds = []
    missing = []
    for fold_idx in range(k):
        summary_path = run_dir / f"fold_{fold_idx:02d}" / "fold_summary.json"
        if summary_path.exists():
            folds.append(json.loads(summary_path.read_text()))
        else:
            missing.append(fold_idx)

    if not folds:
        raise FileNotFoundError(f"No fold_summary.json files found under {run_dir}")
    if missing:
        print(f"[warn] Missing folds: {missing}. Aggregate over {len(folds)}/{k} folds.")

    accs = [r["test_accuracy"] for r in folds]
    f1s = [r["test_macro_f1"] for r in folds]
    mif1s = [r.get("test_micro_f1", 0.0) for r in folds]

    summary = {
        "run_name": run_name,
        "k": k,
        "n_folds_completed": len(folds),
        "graph_cache": graph_cache,
        "folds": folds,
        "aggregate": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "micro_f1_mean": float(np.mean(mif1s)),
            "micro_f1_std": float(np.std(mif1s)),
        },
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    paths_cfg = cfg.get("paths", {})
    graph_cache_path = Path(
        args.graph_cache
        or (Path(paths_cfg["graph_cache_dir"]) / str(cfg["graph"]["cache_name"]))
    )
    run_dir = ensure_dir(
        Path(paths_cfg["outputs_dir"]) / "models" / args.run_name / "kfold"
    )
    log_dir = ensure_dir(Path(paths_cfg["outputs_dir"]) / "logs")
    logger = configure_logger(f"kfold_cv_{args.run_name}", log_dir=log_dir)

    # ── Aggregate-only shortcut ────────────────────────────────────────────
    if args.aggregate_only:
        summary = _aggregate(run_dir, args.k, args.run_name, str(graph_cache_path))
        dump_json(summary, run_dir / "kfold_summary.json")
        agg = summary["aggregate"]
        logger.info(
            "Aggregate (%d/%d folds) | accuracy=%.4f±%.4f  macro_f1=%.4f±%.4f",
            summary["n_folds_completed"], args.k,
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["macro_f1_mean"], agg["macro_f1_std"],
        )
        return

    # ── Load graph bundle ─────────────────────────────────────────────────
    logger.info("Loading graph bundle from %s …", graph_cache_path)
    bundle = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    metadata = bundle["metadata"]
    label_names = list(metadata["label_names"])
    logger.info("Graph loaded | n_cases=%d | labels=%s", len(bundle["data"]["case"].y), label_names)

    base_seed = int(cfg.get("project", {}).get("seed", 42))
    y_all = bundle["data"]["case"].y.numpy()
    n_cases = len(y_all)

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=base_seed)
    fold_splits = list(skf.split(np.arange(n_cases), y_all))

    folds_to_run = [args.fold] if args.fold is not None else list(range(args.k))
    logger.info(
        "Running %d fold(s) of %d-fold CV | val_fraction=%.2f",
        len(folds_to_run), args.k, args.val_fraction,
    )

    # ── Per-fold training ─────────────────────────────────────────────────
    for fold_idx in folds_to_run:
        fold_dir = ensure_dir(run_dir / f"fold_{fold_idx:02d}")
        train_fold_idx, test_fold_idx = fold_splits[fold_idx]
        fold_seed = base_seed + fold_idx
        rng = np.random.default_rng(fold_seed)

        logger.info(
            "── Fold %d/%d | train=%d  test=%d  seed=%d",
            fold_idx + 1, args.k,
            len(train_fold_idx), len(test_fold_idx), fold_seed,
        )

        train_mask, val_mask, test_mask = _make_fold_masks(
            n_cases=n_cases,
            train_idx=train_fold_idx,
            test_idx=test_fold_idx,
            val_fraction=args.val_fraction,
            rng=rng,
        )

        # Inject new masks into a fresh copy of the graph data
        fold_data = deepcopy(bundle["data"])
        fold_data["case"].train_mask = train_mask
        fold_data["case"].val_mask = val_mask
        fold_data["case"].test_mask = test_mask

        fold_cfg = deepcopy(cfg)
        fold_cfg.setdefault("project", {})["seed"] = fold_seed

        set_global_seed(fold_seed)
        result = train_model(
            data=fold_data,
            label_names=label_names,
            cfg=fold_cfg,
            seed=fold_seed,
            logger=logger,
        )

        test_m = result["metrics"].get("test", {})
        fold_summary = {
            "fold": fold_idx,
            "seed": fold_seed,
            "train_n": int(train_mask.sum().item()),
            "val_n": int(val_mask.sum().item()),
            "test_n": int(test_mask.sum().item()),
            "best_epoch": int(result["metrics"].get("best_epoch", 0)),
            "epochs_completed": int(result["metrics"].get("epochs_completed", 0)),
            "test_accuracy": float(test_m.get("accuracy", 0.0)),
            "test_macro_f1": float(test_m.get("macro_f1", 0.0)),
            "test_micro_f1": float(test_m.get("micro_f1", 0.0)),
            "test_per_class": test_m.get("per_class", {}),
        }
        if "roc_auc" in test_m:
            fold_summary["test_roc_auc"] = float(test_m["roc_auc"])

        # Save artifacts
        torch.save(result["model_state_dict"], fold_dir / "model.pt")
        result["predictions_df"].to_csv(fold_dir / "predictions.csv", index=False)
        dump_json(result["metrics"], fold_dir / "metrics.json")
        dump_json(fold_summary, fold_dir / "fold_summary.json")

        confusion = test_m.get("confusion_matrix", [])
        if confusion:
            save_confusion_matrix_plot(
                confusion=confusion,
                label_names=label_names,
                output_path=fold_dir / "confusion_matrix_test.png",
                title=f"{args.run_name} Fold {fold_idx + 1} Test",
            )

        logger.info(
            "Fold %d done | test_acc=%.4f  test_macro_f1=%.4f",
            fold_idx + 1, fold_summary["test_accuracy"], fold_summary["test_macro_f1"],
        )

    # ── Aggregate when all folds were run in one shot ─────────────────────
    if args.fold is None:
        summary = _aggregate(run_dir, args.k, args.run_name, str(graph_cache_path))
        dump_json(summary, run_dir / "kfold_summary.json")
        agg = summary["aggregate"]
        logger.info(
            "K-fold complete | accuracy=%.4f±%.4f  macro_f1=%.4f±%.4f",
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["macro_f1_mean"], agg["macro_f1_std"],
        )
        logger.info("Results written to %s", run_dir / "kfold_summary.json")
    else:
        logger.info(
            "Fold %d complete. Run remaining folds then aggregate with --aggregate-only.",
            args.fold,
        )


if __name__ == "__main__":
    main()
