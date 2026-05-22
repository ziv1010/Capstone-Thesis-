#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CSV_NAMES = [
    "case_counterfactual_groups.csv",
    "case_top_explanations.csv",
    "case_summary.csv",
    "connected_case_label_distribution.csv",
    "attention_counterfactual_overlap.csv",
]


def read_csvs(shard_dirs: list[Path], name: str) -> pd.DataFrame:
    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / name
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def aggregate_importance(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty or key not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["delta_pred_proba"] = pd.to_numeric(work["delta_pred_proba"], errors="coerce")
    work["abs_delta_pred_proba"] = pd.to_numeric(work["abs_delta_pred_proba"], errors="coerce")
    work["masked_edge_count"] = pd.to_numeric(work["masked_edge_count"], errors="coerce")
    work["attention_score"] = pd.to_numeric(work.get("attention_score"), errors="coerce")
    work["prediction_flipped_bool"] = (
        work["prediction_flipped"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    out = (
        work.groupby(key, dropna=False)
        .agg(
            n_groups=("case_index", "size"),
            n_cases=("case_index", "nunique"),
            mean_delta_pred_proba=("delta_pred_proba", "mean"),
            mean_abs_delta_pred_proba=("abs_delta_pred_proba", "mean"),
            sum_abs_delta_pred_proba=("abs_delta_pred_proba", "sum"),
            flip_rate=("prediction_flipped_bool", "mean"),
            mean_masked_edge_count=("masked_edge_count", "mean"),
            mean_attention_score=("attention_score", "mean"),
        )
        .reset_index()
        .sort_values("sum_abs_delta_pred_proba", ascending=False)
    )
    return out


def relation_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "relation_types" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["relation_type"] = work["relation_types"].fillna("").astype(str).str.split("|")
    work = work.explode("relation_type")
    work = work[work["relation_type"].astype(str).str.len() > 0]
    out = aggregate_importance(work, "relation_type")
    if out.empty:
        return out
    parts = out["relation_type"].astype(str).str.split("__", expand=True)
    if parts.shape[1] >= 3:
        out["src_type"] = parts[0]
        out["relation"] = parts[1]
        out["dst_type"] = parts[2]
    return out


def leakage_summary(df: pd.DataFrame) -> pd.DataFrame:
    leakage_types = {
        "judge",
        "petitioner",
        "respondent",
        "petitioner_lawyer",
        "defence_lawyer",
        "lawyer",
        "court",
    }
    if df.empty or "evidence_type" not in df.columns:
        return pd.DataFrame()
    work = df[df["evidence_type"].isin(leakage_types)].copy()
    if work.empty:
        return pd.DataFrame()
    work["abs_delta_pred_proba"] = pd.to_numeric(work["abs_delta_pred_proba"], errors="coerce")
    work["prediction_flipped_bool"] = (
        work["prediction_flipped"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    out = (
        work.groupby("evidence_type", dropna=False)
        .agg(
            n_groups=("case_index", "size"),
            mean_abs_delta_pred_proba=("abs_delta_pred_proba", "mean"),
            median_abs_delta_pred_proba=("abs_delta_pred_proba", "median"),
            p95_abs_delta_pred_proba=("abs_delta_pred_proba", lambda x: x.quantile(0.95)),
            max_abs_delta_pred_proba=("abs_delta_pred_proba", "max"),
            flip_rate=("prediction_flipped_bool", "mean"),
        )
        .reset_index()
        .sort_values("mean_abs_delta_pred_proba", ascending=False)
    )
    return out


def load_run_summaries(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for shard_dir in shard_dirs:
        path = shard_dir / "run_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FINAL_EXPLANATION shard outputs.")
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shard_dirs = sorted(
        path for path in args.shards_dir.iterdir() if path.is_dir() and path.name.startswith("shard_")
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found under {args.shards_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, str] = {}
    for name in CSV_NAMES:
        df = read_csvs(shard_dirs, name)
        if name == "connected_case_label_distribution.csv" and not df.empty:
            subset = [col for col in ["evidence_type", "evidence_global_index"] if col in df.columns]
            if subset:
                df = df.drop_duplicates(subset=subset)
        target = args.output_dir / name
        write(df, target)
        merged[name.removesuffix(".csv")] = str(target)

    long_df = pd.read_csv(args.output_dir / "case_counterfactual_groups.csv", low_memory=False)
    write(aggregate_importance(long_df, "path_family"), args.output_dir / "typed_path_importance.csv")
    write(relation_importance(long_df), args.output_dir / "relation_type_importance.csv")
    write(aggregate_importance(long_df, "evidence_type"), args.output_dir / "evidence_type_importance.csv")
    write(leakage_summary(long_df), args.output_dir / "leakage_sensitivity_summary.csv")

    run_summaries = load_run_summaries(shard_dirs)
    summary = {
        "shards_dir": str(args.shards_dir),
        "output_dir": str(args.output_dir),
        "n_shards": len(shard_dirs),
        "processed_cases": int(sum(item.get("processed_cases", 0) for item in run_summaries)),
        "processed_groups": int(sum(item.get("processed_groups", 0) for item in run_summaries)),
        "failed_cases": [
            failure
            for item in run_summaries
            for failure in item.get("failed_cases", [])
        ],
        "shard_summaries": run_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = {
        "counterfactual_groups": str(args.output_dir / "case_counterfactual_groups.csv"),
        "top_explanations": str(args.output_dir / "case_top_explanations.csv"),
        "case_summary": str(args.output_dir / "case_summary.csv"),
        "connected_case_label_distribution": str(args.output_dir / "connected_case_label_distribution.csv"),
        "typed_path_importance": str(args.output_dir / "typed_path_importance.csv"),
        "relation_type_importance": str(args.output_dir / "relation_type_importance.csv"),
        "evidence_type_importance": str(args.output_dir / "evidence_type_importance.csv"),
        "leakage_sensitivity_summary": str(args.output_dir / "leakage_sensitivity_summary.csv"),
        "attention_counterfactual_overlap": str(args.output_dir / "attention_counterfactual_overlap.csv"),
        "run_summary": str(args.output_dir / "run_summary.json"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[merge done] shards={len(shard_dirs)} cases={summary['processed_cases']} "
        f"groups={summary['processed_groups']} out={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
