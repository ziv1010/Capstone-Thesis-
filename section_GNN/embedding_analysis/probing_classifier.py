#!/usr/bin/env python3
"""
Probing classifier: logistic regression on pre-GNN vs post-GNN case embeddings.

Trains a logistic regression on the case node representations at two stages:
  - pre_gnn:  raw input features (BGE-M3 text embedding + scalar features)
  - post_gnn: hidden["case"] after all HGT message-passing layers

Reports accuracy and macro-F1 for each stage on the held-out test split.
The gap between pre_gnn and post_gnn accuracy quantifies how much predictive
structure the GNN message-passing adds beyond the text embeddings alone.

Also fits a dummy (majority-class) baseline for reference.

Outputs:
  <run_name>_probing_results.json   — full metrics per stage
  <run_name>_probing_bar.png        — bar chart comparing the three stages

Usage
-----
  python probing_classifier.py --embeddings <path/to/run_name_embeddings.npz>

Optional flags:
  --max-iter   Logistic regression solver iterations (default 500)
  --C          Regularisation strength (default 1.0, smaller = more regularised)
  --output-dir Where to write outputs (defaults to same dir as .npz)
  --cv         k for cross-val inside train split (default 0 = no inner CV, use val split)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probing classifiers on GNN embeddings.")
    p.add_argument("--embeddings", required=True, help="Path to *_embeddings.npz")
    p.add_argument("--max-iter",   type=int,   default=500)
    p.add_argument("--C",          type=float, default=1.0)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    max_iter: int,
    seed: int,
    label: str,
) -> dict:
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    acc    = float(accuracy_score(y_test, y_pred))
    macro  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    binary = float(f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0))

    print(
        f"[probe] {label:20s}  test_acc={acc:.4f}  macro_f1={macro:.4f}  binary_f1={binary:.4f}"
        f"  (train_n={len(y_train)}  test_n={len(y_test)})"
    )
    return {
        "stage":          label,
        "test_accuracy":  acc,
        "test_macro_f1":  macro,
        "test_binary_f1": binary,
        "train_n":        int(len(y_train)),
        "test_n":         int(len(y_test)),
        "n_features":     int(X_train.shape[1]),
    }


def _dummy(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(y_train.reshape(-1, 1), y_train)
    y_pred = clf.predict(y_test.reshape(-1, 1))
    acc   = float(accuracy_score(y_test, y_pred))
    macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    print(f"[probe] {'majority_baseline':20s}  test_acc={acc:.4f}  macro_f1={macro:.4f}")
    return {
        "stage":          "majority_baseline",
        "test_accuracy":  acc,
        "test_macro_f1":  macro,
        "test_binary_f1": None,
        "train_n":        int(len(y_train)),
        "test_n":         int(len(y_test)),
        "n_features":     0,
    }


def _bar_chart(results: list[dict], out_path: Path, run_name: str) -> None:
    stages   = [r["stage"] for r in results]
    accs     = [r["test_accuracy"] for r in results]
    macros   = [r["test_macro_f1"] for r in results]

    x = np.arange(len(stages))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, accs,   width, label="Accuracy",  color="#1f77b4", alpha=0.85)
    ax.bar(x + width / 2, macros, width, label="Macro-F1",  color="#ff7f0e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_", "\n") for s in stages],
        fontsize=9,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Probing Classifier: Pre-GNN vs Post-GNN\n({run_name})")
    ax.legend()
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    for rect in ax.patches:
        h = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            h + 0.01, f"{h:.3f}",
            ha="center", va="bottom", fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[probe] bar chart → {out_path.name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    npz_path = Path(args.embeddings)
    out_dir  = Path(args.output_dir) if args.output_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = npz_path.stem.replace("_embeddings", "")

    data = np.load(npz_path, allow_pickle=True)
    labels   = data["labels"].astype(int)
    splits   = data["splits"]
    pre_gnn  = data["pre_gnn"].astype(np.float32)
    post_gnn = data["post_gnn"].astype(np.float32)

    train_mask = splits == "train"
    test_mask  = splits == "test"

    # If the kfold stored no explicit masks, fall back to all-vs-all
    if train_mask.sum() == 0:
        print("[probe] no train/test masks found — using 80/20 random split")
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_mask = np.zeros(len(labels), dtype=bool)
        test_mask  = np.zeros(len(labels), dtype=bool)
        train_mask[idx[:cut]] = True
        test_mask[idx[cut:]]  = True

    print(f"[probe] {run_name}  train_n={train_mask.sum()}  test_n={test_mask.sum()}")

    results = []
    results.append(_dummy(labels[train_mask], labels[test_mask]))
    results.append(_probe(
        pre_gnn[train_mask],  labels[train_mask],
        pre_gnn[test_mask],   labels[test_mask],
        C=args.C, max_iter=args.max_iter, seed=args.seed,
        label="pre_gnn_raw_features",
    ))
    results.append(_probe(
        post_gnn[train_mask], labels[train_mask],
        post_gnn[test_mask],  labels[test_mask],
        C=args.C, max_iter=args.max_iter, seed=args.seed,
        label="post_gnn_embeddings",
    ))

    # ── Delta analysis ────────────────────────────────────────
    pre_acc  = next(r["test_accuracy"] for r in results if r["stage"] == "pre_gnn_raw_features")
    post_acc = next(r["test_accuracy"] for r in results if r["stage"] == "post_gnn_embeddings")
    delta    = post_acc - pre_acc
    print(f"[probe] delta (post-GNN − pre-GNN) accuracy = {delta:+.4f}")

    summary = {
        "run_name":   run_name,
        "C":          args.C,
        "max_iter":   args.max_iter,
        "delta_accuracy_post_minus_pre": round(delta, 6),
        "results":    results,
    }

    json_path = out_dir / f"{run_name}_probing_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[probe] results → {json_path.name}")

    bar_path = out_dir / f"{run_name}_probing_bar.png"
    _bar_chart(results, bar_path, run_name)
    print("[probe] done.")


if __name__ == "__main__":
    main()
