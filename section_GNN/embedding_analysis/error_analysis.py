#!/usr/bin/env python3
"""
Per-characteristic error analysis for the trained HeteroLegalOutcomeGNN.

Joins predictions.csv (from a kfold fold directory) with graph_metadata.json
(from the graph cache directory) to answer: "Does the model fail more on
cases with certain structural characteristics?"

Characteristics analysed (from case_summaries in graph_metadata.json):
  - case_year           — does error rate vary over time?
  - judge_count         — do multi-judge benches affect accuracy?
  - statute_count       — do statute-heavy cases confuse the model?
  - precedent_count     — do precedent-heavy cases confuse the model?
  - facts_length        — short vs long fact sections
  - arguments_length    — argument length
  - domain              — error rate per legal domain

Outputs (written to --output-dir or next to predictions.csv):
  <run_name>_error_analysis.json     — grouped statistics
  <run_name>_error_by_domain.png     — accuracy per domain
  <run_name>_error_by_year.png       — accuracy per year
  <run_name>_error_by_statute.png    — accuracy vs statute count (binned)
  <run_name>_error_by_facts_len.png  — accuracy vs facts length (binned)
  <run_name>_error_by_judges.png     — accuracy vs judge count

Usage
-----
  python error_analysis.py \\
      --predictions  <path/to/fold_00/predictions.csv> \\
      --metadata     <path/to/graph_cache/graph_metadata.reasoning_focused.json> \\
      [--output-dir  <dir>] [--run-name <name>]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-characteristic error analysis.")
    p.add_argument("--predictions", required=True, help="Path to predictions.csv from a kfold fold")
    p.add_argument("--metadata",    required=True, help="Path to graph_metadata.*.json")
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--run-name",    default=None)
    p.add_argument("--split",       default="test", choices=["test", "val", "train", "all"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_DOMAIN_RE = re.compile(r"^(.+?)__")

def _domain(case_id: str) -> str:
    m = _DOMAIN_RE.match(case_id)
    return m.group(1).replace("_timed_mistral", "").replace("_", " ") if m else "unknown"


def _bin_column(series: pd.Series, n_bins: int = 5) -> pd.Series:
    return pd.cut(series, bins=n_bins, include_lowest=True)


def _accuracy_by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)
    return pd.DataFrame({
        "group":    list(g.groups.keys()),
        "accuracy": g.apply(lambda x: (x["correct"]).mean()).values,
        "n":        g.size().values,
        "n_error":  g.apply(lambda x: (~x["correct"]).sum()).values,
    }).sort_values("group")


def _bar(df_g: pd.DataFrame, title: str, xlabel: str, out_path: Path, color: str = "#1f77b4") -> None:
    labels = [str(g) for g in df_g["group"]]
    accs   = df_g["accuracy"].values
    ns     = df_g["n"].values

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), 5))
    bars = ax.bar(range(len(labels)), accs, color=color, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.axhline(accs.mean(), color="grey", linestyle="--", linewidth=0.9, label=f"mean={accs.mean():.3f}")
    ax.legend(fontsize=9)

    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={n}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[error] {title[:40]} → {out_path.name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pred_path = Path(args.predictions)
    run_name  = args.run_name or pred_path.parent.name
    out_dir   = Path(args.output_dir) if args.output_dir else pred_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load predictions ──────────────────────────────────────
    preds = pd.read_csv(pred_path)
    if args.split != "all":
        preds = preds[preds["split"] == args.split].copy()
    preds["correct"] = preds["target_label"] == preds["pred_label"]
    print(f"[error] {len(preds)} cases ({args.split})  overall accuracy={preds['correct'].mean():.4f}")

    # ── Load metadata ─────────────────────────────────────────
    with open(args.metadata) as f:
        meta = json.load(f)
    case_summaries = meta.get("case_summaries", [])
    meta_df = pd.DataFrame([
        {
            "case_id":          cs["case_id"],
            "judge_count":      cs["metadata"].get("judge_count", 0),
            "lawyer_count":     cs["metadata"].get("lawyer_count", 0),
            "statute_count":    cs["metadata"].get("statute_count", 0),
            "precedent_count":  cs["metadata"].get("precedent_count", 0),
            "preamble_length":  cs["metadata"].get("preamble_length", 0),
            "facts_length":     cs["metadata"].get("facts_length", 0),
            "arguments_length": cs["metadata"].get("arguments_length", 0),
            "case_year":        cs["metadata"].get("case_year", 0),
        }
        for cs in case_summaries
    ])

    df = preds.merge(meta_df, on="case_id", how="left")
    df["domain"] = df["case_id"].apply(_domain)
    print(f"[error] joined {df['case_id'].notna().sum()} / {len(df)} cases with metadata")

    results: dict = {"run_name": run_name, "split": args.split,
                     "overall_accuracy": float(preds["correct"].mean()),
                     "n": int(len(preds))}

    # ── Domain ────────────────────────────────────────────────
    g_domain = _accuracy_by_group(df, "domain")
    _bar(g_domain, f"Accuracy by legal domain\n({run_name})", "Domain",
         out_dir / f"{run_name}_error_by_domain.png", "#1f77b4")
    results["by_domain"] = g_domain.to_dict("records")

    # ── Year (every 2 years) ──────────────────────────────────
    df_yr = df[df["case_year"].between(1990, 2030)].copy()
    df_yr["year_bin"] = (df_yr["case_year"] // 2) * 2  # 2-year buckets
    g_year = _accuracy_by_group(df_yr, "year_bin")
    _bar(g_year, f"Accuracy by case year\n({run_name})", "Year (2-year bins)",
         out_dir / f"{run_name}_error_by_year.png", "#2ca02c")
    results["by_year"] = g_year.to_dict("records")

    # ── Statute count ─────────────────────────────────────────
    df["statute_bin"] = pd.cut(df["statute_count"], bins=[0,1,3,6,10,999],
                                labels=["0","1-3","4-6","7-10","10+"],
                                include_lowest=True)
    g_statute = _accuracy_by_group(df, "statute_bin")
    _bar(g_statute, f"Accuracy by statute count\n({run_name})", "Statute count",
         out_dir / f"{run_name}_error_by_statute.png", "#d62728")
    results["by_statute_count"] = g_statute.to_dict("records")

    # ── Facts length (quintiles) ──────────────────────────────
    _fact_labels = ["Q1\n(short)","Q2","Q3","Q4","Q5\n(long)"]
    _, fact_edges = pd.qcut(df["facts_length"], q=5, duplicates="drop", retbins=True)
    n_fact_bins = len(fact_edges) - 1
    df["facts_bin"] = pd.qcut(df["facts_length"], q=5, duplicates="drop",
                               labels=_fact_labels[:n_fact_bins])
    g_facts = _accuracy_by_group(df, "facts_bin")
    _bar(g_facts, f"Accuracy by facts length (quintiles)\n({run_name})", "Facts length quintile",
         out_dir / f"{run_name}_error_by_facts_len.png", "#ff7f0e")
    results["by_facts_length"] = g_facts.to_dict("records")

    # ── Judge count ───────────────────────────────────────────
    df["judge_bin"] = pd.cut(df["judge_count"], bins=[0,1,2,3,4,99],
                              labels=["0-1","2","3","4","5+"],
                              include_lowest=True)
    g_judges = _accuracy_by_group(df, "judge_bin")
    _bar(g_judges, f"Accuracy by judge count\n({run_name})", "Number of judges",
         out_dir / f"{run_name}_error_by_judges.png", "#9467bd")
    results["by_judge_count"] = g_judges.to_dict("records")

    # ── Save summary ──────────────────────────────────────────
    def _serialise(obj):
        if isinstance(obj, (np.integer,)):     return int(obj)
        if isinstance(obj, (np.floating,)):    return float(obj)
        if isinstance(obj, pd.Interval):       return str(obj)
        if isinstance(obj, pd.CategoricalDtype): return str(obj)
        raise TypeError(type(obj))

    json_path = out_dir / f"{run_name}_error_analysis.json"
    json_path.write_text(json.dumps(results, indent=2, default=_serialise))
    print(f"[error] summary → {json_path.name}")

    # ── Print key findings ────────────────────────────────────
    print("\n[error] Key findings:")
    domain_rows = sorted(results["by_domain"], key=lambda x: x["accuracy"])
    if domain_rows:
        print(f"  Lowest accuracy domain : {domain_rows[0]['group']:35s}  {domain_rows[0]['accuracy']:.4f}  (n={domain_rows[0]['n']})")
        print(f"  Highest accuracy domain: {domain_rows[-1]['group']:35s}  {domain_rows[-1]['accuracy']:.4f}  (n={domain_rows[-1]['n']})")

    statute_rows = results["by_statute_count"]
    if statute_rows:
        print(f"  Accuracy, 0 statutes : {next((r['accuracy'] for r in statute_rows if r['group']=='0'), 'N/A')}")
        print(f"  Accuracy, 7-10 statutes: {next((r['accuracy'] for r in statute_rows if r['group']=='7-10'), 'N/A')}")

    print("[error] done.")


if __name__ == "__main__":
    main()
