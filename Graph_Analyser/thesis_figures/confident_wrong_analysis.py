#!/usr/bin/env python3
"""Confidently-wrong neighbour analysis.

Generalises the case-10212-vs-13701 anecdote into a quantitative claim:

Question: when the model is confidently wrong (confidence >= threshold,
pred != true), do the test case's k nearest training neighbours in the
post-GNN embedding space systematically align with the *predicted* label
rather than the *true* label? If yes, the failure is a structural property
of the embedding — the case has been placed in the wrong cluster — not
random noise.

Method:
  1. Filter predictions.csv for test-split rows.
  2. Split into confidently-wrong (conf >= thr, pred != true) and
     confidently-correct (conf >= thr, pred == true).
  3. For each case, L2-normalise its post-GNN embedding, query the same
     FAISS index Phase 1-2 built (cosine / IP over train-split embeddings),
     take top-k neighbours.
  4. For each neighbour, read its ground-truth label from predictions.csv.
  5. Per case, compute:
       frac_match_pred  — share of neighbours with true-label == pred-label
                          (high means the embedding supports the model's
                          mistake: near-neighbours come from the predicted
                          class)
       frac_match_true  — share of neighbours with true-label == true-label
                          (for reference)
  6. Compare distributions of frac_match_pred between wrong vs correct
     populations, and to the train-split base rate of the predicted class.

Outputs in outputs/thesis_figures/confident_wrong_analysis/:
  per_case_confidently_wrong.csv
  per_case_confidently_correct.csv
  summary.json
  neighbour_alignment_hist.png
  README.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import Paths  # noqa: E402


def _normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return x / n


def _label_str(x) -> str:
    return str(x).strip()


def _analyse_pool(
    cases: pd.DataFrame,
    case_embs: np.ndarray,
    faiss_idx: faiss.Index,
    train_indices: np.ndarray,
    train_label_by_node_idx: dict[int, str],
    k: int,
) -> pd.DataFrame:
    if len(cases) == 0:
        return pd.DataFrame()
    node_ids = cases["node_index"].to_numpy()
    q = _normalise(case_embs[node_ids].astype(np.float32))
    sims, neigh_rows = faiss_idx.search(q, k)
    rows = []
    for i, nidx in enumerate(node_ids):
        row = cases.iloc[i]
        pred = _label_str(row["pred_label"])
        true = _label_str(row["target_label"])
        neighbour_node_indices = train_indices[neigh_rows[i]]
        neigh_labels = [train_label_by_node_idx[int(j)] for j in neighbour_node_indices]
        match_pred = sum(1 for l in neigh_labels if l == pred) / k
        match_true = sum(1 for l in neigh_labels if l == true) / k
        rows.append(
            {
                "node_index": int(nidx),
                "case_id": row["case_id"],
                "true_label": true,
                "pred_label": pred,
                "confidence": float(row["confidence"]),
                "top_k_sim_mean": float(np.mean(sims[i])),
                "top_k_sim_min": float(np.min(sims[i])),
                "frac_neigh_match_pred": match_pred,
                "frac_neigh_match_true": match_true,
                "neigh_node_indices": ",".join(map(str, neighbour_node_indices.tolist())),
                "neigh_labels": ",".join(neigh_labels),
            }
        )
    return pd.DataFrame(rows)


def _plot_distributions(
    wrong_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    train_base_rate_allowed: float,
    out_fp: Path,
    k: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bins = np.linspace(-1e-6, 1 + 1e-6, k + 2)
    ax.hist(
        corr_df["frac_neigh_match_pred"],
        bins=bins,
        alpha=0.6,
        color="#1a6b1a",
        label=f"confidently CORRECT (n={len(corr_df)})",
        density=True,
    )
    ax.hist(
        wrong_df["frac_neigh_match_pred"],
        bins=bins,
        alpha=0.7,
        color="#8b0000",
        label=f"confidently WRONG (n={len(wrong_df)})",
        density=True,
    )
    ax.set_xlabel(f"fraction of top-{k} neighbours with true label == MODEL-PREDICTED label")
    ax.set_ylabel("density")
    ax.set_title(
        "Do the confidently-wrong cases still sit next to\n"
        "training cases of the predicted class?"
    )
    # base-rate reference: if we randomly retrieved k train cases, the
    # expected fraction matching "pred = allowed" is the train allowed rate.
    ax.axvline(
        train_base_rate_allowed,
        linestyle="--",
        color="grey",
        label=f"train base rate (allowed)={train_base_rate_allowed:.2f}",
    )
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(
        wrong_df["frac_neigh_match_true"],
        bins=bins,
        alpha=0.7,
        color="#8b0000",
        label="confidently WRONG: match TRUE label",
        density=True,
    )
    ax.hist(
        wrong_df["frac_neigh_match_pred"],
        bins=bins,
        alpha=0.4,
        color="#d62728",
        label="confidently WRONG: match PRED label",
        density=True,
    )
    ax.set_xlabel(f"fraction of top-{k} neighbours with true label == reference label")
    ax.set_ylabel("density")
    ax.set_title(
        "For confidently-wrong cases only: neighbours match\n"
        "the predicted label far more than the true label"
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_fp, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(
    out_dir: Path,
    k: int,
    threshold: float,
    wrong_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    base_rate: float,
    summary: dict,
) -> None:
    worst = wrong_df.sort_values("frac_neigh_match_pred", ascending=False).head(10)
    lines = [
        "# Confidently-wrong neighbour analysis",
        "",
        f"**Threshold:** confidence ≥ {threshold:.2f}.  **k (neighbours):** {k}.",
        f"**Train base rate (allowed):** {base_rate:.4f}.",
        "",
        "## Populations",
        "",
        f"- Confidently WRONG test cases: **{summary['n_confidently_wrong']}**",
        f"- Confidently CORRECT test cases: **{summary['n_confidently_correct']}**",
        "",
        "## Headline numbers",
        "",
        "| Metric | Confidently wrong | Confidently correct |",
        "| ------ | ----------------: | ------------------: |",
        "| mean fraction of top-k neighbours matching **pred** label "
        f"| {summary['wrong_mean_match_pred']:.4f} "
        f"| {summary['correct_mean_match_pred']:.4f} |",
        "| mean fraction of top-k neighbours matching **true** label "
        f"| {summary['wrong_mean_match_true']:.4f} "
        f"| {summary['correct_mean_match_true']:.4f} |",
        f"| cases with ALL k neighbours of predicted class "
        f"| {summary['wrong_all_k_match_pred']}"
        f" / {summary['n_confidently_wrong']} "
        f"| {summary['correct_all_k_match_pred']}"
        f" / {summary['n_confidently_correct']} |",
        "",
        "## Reading",
        "",
        "If `mean fraction matching pred` for the wrong population is well above",
        "the train base rate and close to the correct population's value, the",
        "failure mode is **structural to the embedding**: when the model is",
        "confidently wrong, the case has been placed inside the cluster of the",
        "wrong class and its k nearest training neighbours all belong to that",
        "wrong class. The FAISS retrieval that Graph_Analyser Phase 2 uses to",
        "build the LLM context therefore reinforces the mistake rather than",
        "correcting it.",
        "",
        "## Top-10 most unambiguously mis-clustered wrong cases",
        "",
        "(sorted by fraction of neighbours matching the predicted label)",
        "",
        "| node_index | true | pred | conf | match_pred | match_true | case |",
        "| ---------: | :--: | :--: | ---: | ---------: | ---------: | :--- |",
    ]
    for _, r in worst.iterrows():
        lines.append(
            f"| {int(r['node_index'])} | {r['true_label']} | {r['pred_label']} "
            f"| {r['confidence']:.4f} "
            f"| {r['frac_neigh_match_pred']:.2f} "
            f"| {r['frac_neigh_match_true']:.2f} "
            f"| {str(r['case_id'])[:70].replace('|', '/')} |"
        )
    lines += [
        "",
        "## Files",
        "- `per_case_confidently_wrong.csv`",
        "- `per_case_confidently_correct.csv`",
        "- `neighbour_alignment_hist.png`",
        "- `summary.json`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    paths = Paths.default()
    out_dir = paths.fig_dir / "confident_wrong_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(paths.predictions_csv)
    preds["pred_label"] = preds["pred_label"].astype(str).str.strip()
    preds["target_label"] = preds["target_label"].astype(str).str.strip()
    case_embs = np.load(paths.case_embeddings)

    p12 = paths.predictions_csv.parent
    faiss_idx = faiss.read_index(str(p12 / "train_faiss.index"))
    train_indices = np.load(p12 / "train_indices.npy")

    train_rows = preds[preds["split"] == "train"].set_index("node_index")
    train_label_by_node_idx = train_rows["target_label"].to_dict()
    base_rate_allowed = float((train_rows["target_label"] == "1").mean())

    test = preds[preds["split"] == "test"]
    wrong = test[
        (test["pred_label"] != test["target_label"])
        & (test["confidence"] >= args.threshold)
    ].copy()
    correct = test[
        (test["pred_label"] == test["target_label"])
        & (test["confidence"] >= args.threshold)
    ].copy()

    print(
        f"[confident_wrong] threshold={args.threshold}  k={args.k}  "
        f"wrong={len(wrong)}  correct={len(correct)}  "
        f"train base-rate allowed={base_rate_allowed:.4f}",
        flush=True,
    )

    wrong_df = _analyse_pool(
        wrong, case_embs, faiss_idx, train_indices,
        train_label_by_node_idx, args.k,
    )
    correct_df = _analyse_pool(
        correct, case_embs, faiss_idx, train_indices,
        train_label_by_node_idx, args.k,
    )

    wrong_csv = out_dir / "per_case_confidently_wrong.csv"
    correct_csv = out_dir / "per_case_confidently_correct.csv"
    wrong_df.to_csv(wrong_csv, index=False)
    correct_df.to_csv(correct_csv, index=False)

    summary = {
        "threshold": args.threshold,
        "k": args.k,
        "n_confidently_wrong": int(len(wrong_df)),
        "n_confidently_correct": int(len(correct_df)),
        "train_base_rate_allowed": base_rate_allowed,
        "wrong_mean_match_pred": float(wrong_df["frac_neigh_match_pred"].mean())
            if len(wrong_df) else None,
        "wrong_mean_match_true": float(wrong_df["frac_neigh_match_true"].mean())
            if len(wrong_df) else None,
        "correct_mean_match_pred": float(correct_df["frac_neigh_match_pred"].mean())
            if len(correct_df) else None,
        "correct_mean_match_true": float(correct_df["frac_neigh_match_true"].mean())
            if len(correct_df) else None,
        "wrong_all_k_match_pred": int((wrong_df["frac_neigh_match_pred"] == 1.0).sum())
            if len(wrong_df) else 0,
        "correct_all_k_match_pred": int((correct_df["frac_neigh_match_pred"] == 1.0).sum())
            if len(correct_df) else 0,
    }
    # Break the wrong population down by which direction the error goes.
    if len(wrong_df):
        for direction, sub in wrong_df.groupby(["true_label", "pred_label"]):
            tag = f"wrong_{direction[0]}_to_{direction[1]}"
            summary[tag] = {
                "n": int(len(sub)),
                "mean_match_pred": float(sub["frac_neigh_match_pred"].mean()),
                "mean_match_true": float(sub["frac_neigh_match_true"].mean()),
            }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_distributions(
        wrong_df, correct_df, base_rate_allowed,
        out_dir / "neighbour_alignment_hist.png", args.k,
    )
    _write_readme(
        out_dir, args.k, args.threshold, wrong_df, correct_df,
        base_rate_allowed, summary,
    )
    print(f"[confident_wrong] wrote {out_dir}")


if __name__ == "__main__":
    main()
