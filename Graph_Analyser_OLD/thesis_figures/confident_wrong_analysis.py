#!/usr/bin/env python3
"""Confidently-wrong PGExplainer evidence analysis.

This helper summarizes Phase 6 diagnostics for confident predictions. It uses
only PGExplainer-surfaced nodes and their training-label diagnostic rows.

Outputs in outputs/thesis_figures/confident_wrong_analysis/:
  per_case_confidently_wrong.csv
  per_case_confidently_correct.csv
  summary.json
  evidence_alignment_hist.png
  README.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import Paths  # noqa: E402


def _load_phase6_summary(paths: Paths) -> pd.DataFrame:
    summary_path = paths.predictions_csv.parent.parent / "phase6_misclass_diagnostic" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Phase 6 summary missing at {summary_path}. Run phase6_misclass_diagnostic.py first."
        )
    return pd.read_json(summary_path)


def _join_predictions(paths: Paths) -> pd.DataFrame:
    preds = pd.read_csv(paths.predictions_csv)
    diag = _load_phase6_summary(paths)
    merged = diag.merge(
        preds[["node_index", "case_id", "split"]],
        left_on="case_node_index",
        right_on="node_index",
        how="left",
        suffixes=("", "_preds"),
    )
    return merged


def _population(df: pd.DataFrame, threshold: float, correct: bool) -> pd.DataFrame:
    mask = (df["confidence"] >= threshold) & (df["target_label"] == df["predicted_label"])
    if not correct:
        mask = (df["confidence"] >= threshold) & (df["target_label"] != df["predicted_label"])
    cols = [
        "case_node_index",
        "case_id",
        "target_label",
        "predicted_label",
        "confidence",
        "diagnostic_scope",
        "evidence_majority",
        "evidence_strength",
        "evidence_supports_prediction",
        "n_nodes",
        "n_traceable_nodes",
        "traceable_importance_share",
    ]
    return df.loc[mask, [c for c in cols if c in df.columns]].copy()


def _safe_mean(series: pd.Series) -> float | None:
    vals = series.dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _plot(wrong: pd.DataFrame, correct: pd.DataFrame, out_fp: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(0, 1, 11)
    if len(correct):
        ax.hist(correct["evidence_strength"].fillna(0), bins=bins, alpha=0.55, label="confident correct")
    if len(wrong):
        ax.hist(wrong["evidence_strength"].fillna(0), bins=bins, alpha=0.70, label="confident wrong")
    ax.set_xlabel("Phase 6 evidence skew strength")
    ax.set_ylabel("cases")
    ax.set_title("How decisive are the surfaced PGExplainer nodes?")
    ax.legend(fontsize=9)

    ax = axes[1]
    labels = ["wrong", "correct"]
    values = [
        float(wrong["evidence_supports_prediction"].mean()) if len(wrong) else 0.0,
        float(correct["evidence_supports_prediction"].mean()) if len(correct) else 0.0,
    ]
    ax.bar(labels, values, color=["#d62728", "#1f77b4"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share")
    ax.set_title("Evidence majority matches model prediction")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_readme(
    out_dir: Path,
    threshold: float,
    wrong: pd.DataFrame,
    correct: pd.DataFrame,
    summary: dict,
) -> None:
    lines = [
        "# Confidently-wrong PGExplainer evidence analysis",
        "",
        f"**Threshold:** confidence >= {threshold:.2f}.",
        "",
        "This analysis uses Phase 6 training-label diagnostics for PGExplainer-surfaced nodes.",
        "No embedding index is used.",
        "",
        "## Headline",
        "",
        f"- Confidently wrong cases: **{summary['n_confidently_wrong']}**",
        f"- Confidently correct cases: **{summary['n_confidently_correct']}**",
        f"- Wrong mean evidence strength: **{summary['wrong_mean_evidence_strength']}**",
        f"- Correct mean evidence strength: **{summary['correct_mean_evidence_strength']}**",
        "",
        "## Most decisive wrong cases",
        "",
        "| node_index | true | pred | conf | evidence | strength | traceable | case |",
        "| ---------: | :--: | :--: | ---: | :------- | -------: | --------: | :--- |",
    ]
    if len(wrong):
        ranked = wrong.sort_values("evidence_strength", ascending=False).head(10)
        for _, r in ranked.iterrows():
            lines.append(
                f"| {int(r['case_node_index'])} | {r['target_label']} | {r['predicted_label']} "
                f"| {r['confidence']:.4f} | {r.get('evidence_majority', 'tie')} "
                f"| {r.get('evidence_strength', 0.0):.4f} "
                f"| {int(r.get('n_traceable_nodes', 0))}/{int(r.get('n_nodes', 0))} "
                f"| {str(r['case_id'])[:70].replace('|', '/')} |"
            )
    else:
        lines.append("| - | - | - | - | - | - | - | _none_ |")
    lines += [
        "",
        "## Files",
        "- `per_case_confidently_wrong.csv`",
        "- `per_case_confidently_correct.csv`",
        "- `evidence_alignment_hist.png`",
        "- `summary.json`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    paths = Paths.default()
    out_dir = paths.fig_dir / "confident_wrong_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _join_predictions(paths)
    wrong = _population(df, threshold=args.threshold, correct=False)
    correct = _population(df, threshold=args.threshold, correct=True)

    wrong.to_csv(out_dir / "per_case_confidently_wrong.csv", index=False)
    correct.to_csv(out_dir / "per_case_confidently_correct.csv", index=False)

    summary = {
        "threshold": args.threshold,
        "n_confidently_wrong": int(len(wrong)),
        "n_confidently_correct": int(len(correct)),
        "wrong_mean_evidence_strength": _safe_mean(wrong.get("evidence_strength", pd.Series(dtype=float))),
        "correct_mean_evidence_strength": _safe_mean(correct.get("evidence_strength", pd.Series(dtype=float))),
        "wrong_supports_prediction_share": _safe_mean(wrong.get("evidence_supports_prediction", pd.Series(dtype=float))),
        "correct_supports_prediction_share": _safe_mean(correct.get("evidence_supports_prediction", pd.Series(dtype=float))),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot(wrong, correct, out_dir / "evidence_alignment_hist.png")
    _write_readme(out_dir, args.threshold, wrong, correct, summary)
    print(f"[confident_wrong] wrote {out_dir}")


if __name__ == "__main__":
    main()
