#!/usr/bin/env python3
"""
SHAP analysis on post-GNN case embeddings.

Uses SHAP's LinearExplainer on the probing logistic regression trained on
post-GNN embeddings. This is fast and exact (unlike KernelExplainer) and
answers: "which dimensions of the post-GNN embedding are most predictive?"

Since the 64-dimensional post-GNN space is not directly interpretable, the
analysis focuses on:
  1. Overall feature importance — mean |SHAP| per dimension
  2. Class-level SHAP summary — top dimensions driving "allowed" vs "dismissed"
  3. Dimension correlation with domain/outcome — which dims are domain-specific

Outputs (written to --output-dir or same dir as embeddings .npz):
  <run_name>_shap_bar.png          — top-N SHAP feature importance bar chart
  <run_name>_shap_beeswarm.png     — beeswarm / violin summary (matplotlib fallback)
  <run_name>_shap_summary.json     — ranked feature importances

Usage
-----
  python shap_analysis.py --embeddings <path/to/run_name_embeddings.npz> \\
      [--top-n 20] [--output-dir <dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP analysis on post-GNN embeddings.")
    p.add_argument("--embeddings", required=True, help="Path to *_embeddings.npz")
    p.add_argument("--top-n",      type=int, default=20,  help="Top N dimensions to show")
    p.add_argument("--max-iter",   type=int, default=500)
    p.add_argument("--C",          type=float, default=1.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Fallback: gradient of logistic regression (if no shap installed)
# ─────────────────────────────────────────────────────────────

def _lr_coefficient_importance(
    clf: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (importance, dim_indices) sorted descending by |coef| for class 1."""
    coef = clf.coef_  # (n_classes, n_features) or (1, n_features)
    if coef.shape[0] == 1:
        importance = np.abs(coef[0])
    else:
        # binary: class 1 vs rest
        importance = np.abs(coef[1] - coef[0]) if coef.shape[0] == 2 else np.abs(coef).mean(axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    return importance[order], order


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def _bar_chart(
    importance: np.ndarray,
    dim_indices: np.ndarray,
    title: str,
    out_path: Path,
    ylabel: str = "Mean |SHAP value|",
) -> None:
    labels = [f"dim {d}" for d in dim_indices]
    fig, ax = plt.subplots(figsize=(10, max(5, len(labels) * 0.4)))
    ax.barh(labels[::-1], importance[::-1], color="#1f77b4", alpha=0.85)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] bar chart → {out_path.name}")


def _beeswarm_fallback(
    shap_values: np.ndarray,
    X_scaled: np.ndarray,
    dim_indices: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Scatter plot of SHAP value vs feature value for top dims."""
    n_dims = len(dim_indices)
    fig, axes = plt.subplots(1, n_dims, figsize=(n_dims * 2, 4), sharey=False)
    if n_dims == 1:
        axes = [axes]
    cmap = {0: "#d62728", 1: "#1f77b4"}
    for ax, dim in zip(axes, dim_indices):
        sv  = shap_values[:, dim]
        fv  = X_scaled[:, dim]
        col = [cmap.get(int(y), "#888") for y in labels]
        ax.scatter(sv, fv, c=col, s=4, alpha=0.4)
        ax.set_xlabel(f"SHAP\ndim {dim}", fontsize=7)
        ax.set_ylabel("feature val" if dim == dim_indices[0] else "", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] beeswarm → {out_path.name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    npz_path = Path(args.embeddings)
    out_dir  = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = npz_path.stem.replace("_embeddings", "")

    data     = np.load(npz_path, allow_pickle=True)
    labels   = data["labels"].astype(int)
    splits   = data["splits"]
    post_gnn = data["post_gnn"].astype(np.float32)

    train_mask = splits == "train"
    test_mask  = splits == "test"
    if train_mask.sum() == 0:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_mask = np.zeros(len(labels), dtype=bool)
        test_mask  = np.zeros(len(labels), dtype=bool)
        train_mask[idx[:cut]] = True
        test_mask[idx[cut:]]  = True

    X_train, y_train = post_gnn[train_mask], labels[train_mask]
    X_test,  y_test  = post_gnn[test_mask],  labels[test_mask]

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    print(f"[shap] fitting logistic regression on {X_tr_sc.shape} post-GNN embeddings ...")
    clf = LogisticRegression(C=args.C, max_iter=args.max_iter,
                             random_state=args.seed, class_weight="balanced",
                             solver="lbfgs")
    clf.fit(X_tr_sc, y_train)
    acc = float((clf.predict(X_te_sc) == y_test).mean())
    print(f"[shap] logistic regression test accuracy = {acc:.4f}")

    top_n = min(args.top_n, post_gnn.shape[1])

    if _HAS_SHAP:
        print("[shap] running SHAP LinearExplainer ...")
        explainer   = shap.LinearExplainer(clf, X_tr_sc, feature_perturbation="correlation_dependent")
        shap_values = explainer.shap_values(X_te_sc)  # list of arrays or single array

        # Binary case: shap_values is list of 2 arrays (one per class) or single array
        if isinstance(shap_values, list):
            # Use class-1 (allowed) SHAP values
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)  # (n_features,)
        order = np.argsort(mean_abs_shap)[::-1][:top_n]

        _bar_chart(
            importance = mean_abs_shap[order],
            dim_indices = order,
            title = f"SHAP feature importance (post-GNN dims)\n{run_name}",
            out_path = out_dir / f"{run_name}_shap_bar.png",
        )
        _beeswarm_fallback(
            shap_values = sv,
            X_scaled    = X_te_sc,
            dim_indices = order[:min(8, top_n)],
            labels      = y_test,
            title       = f"SHAP value vs feature value — top dims\n{run_name}",
            out_path    = out_dir / f"{run_name}_shap_beeswarm.png",
        )

        # Per-class sign: positive SHAP → pushes toward "allowed"
        pos_dims = order[sv[:, order].mean(axis=0) > 0][:5]
        neg_dims = order[sv[:, order].mean(axis=0) < 0][:5]
        print(f"[shap] dims pushing toward 'allowed' (+1): {pos_dims.tolist()}")
        print(f"[shap] dims pushing toward 'dismissed'(0): {neg_dims.tolist()}")

        summary = {
            "run_name":       run_name,
            "lr_test_accuracy": acc,
            "method":         "SHAP LinearExplainer",
            "top_dims_by_importance": [
                {"dim": int(d), "mean_abs_shap": float(mean_abs_shap[d])}
                for d in order
            ],
        }
    else:
        print("[shap] SHAP not installed — falling back to |logistic regression coef|")
        importance, order = _lr_coefficient_importance(clf, scaler, X_te_sc, y_test, top_n)
        _bar_chart(
            importance  = importance,
            dim_indices = order,
            title       = f"|LR coefficient| importance (post-GNN dims)\n{run_name}",
            out_path    = out_dir / f"{run_name}_shap_bar.png",
            ylabel      = "Abs LR coefficient",
        )
        summary = {
            "run_name":       run_name,
            "lr_test_accuracy": acc,
            "method":         "logistic_regression_coef (shap not installed)",
            "top_dims_by_importance": [
                {"dim": int(d), "abs_coef": float(importance[i])}
                for i, d in enumerate(order)
            ],
        }

    json_path = out_dir / f"{run_name}_shap_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[shap] summary → {json_path.name}")
    print("[shap] done.")


if __name__ == "__main__":
    main()
