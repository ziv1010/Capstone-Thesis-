#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from validate_explanations import (
    BUCKET_CASE_FIELDS,
    BUCKET_EVIDENCE_FIELDS,
    BUCKET_SUMMARY_FIELDS,
    FAITHFULNESS_AUC_FIELDS,
    FAITHFULNESS_AUC_SUMMARY_FIELDS,
    FAITHFULNESS_CURVE_FIELDS,
    FAITHFULNESS_CURVE_SUMMARY_FIELDS,
    aggregate_faithfulness_auc,
    aggregate_faithfulness_curves,
    build_prediction_bucket_outputs,
)


DETAIL_FILES = [
    "faithfulness_curves.csv",
    "faithfulness_auc_by_case.csv",
    "prediction_bucket_cases.csv",
]

DETAIL_FIELDS = {
    "faithfulness_curves.csv": FAITHFULNESS_CURVE_FIELDS,
    "faithfulness_auc_by_case.csv": FAITHFULNESS_AUC_FIELDS,
    "prediction_bucket_cases.csv": BUCKET_CASE_FIELDS,
}


def read_csvs(shard_dirs: list[Path], name: str) -> pd.DataFrame:
    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / name
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path, low_memory=False))
            except pd.errors.EmptyDataError:
                pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write(df: pd.DataFrame, path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and fieldnames is not None:
        df = pd.DataFrame(columns=fieldnames)
    df.to_csv(path, index=False)


def load_run_summaries(shard_dirs: list[Path]) -> list[dict[str, Any]]:
    summaries = []
    for shard_dir in shard_dirs:
        path = shard_dir / "validation_run_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validation-analysis shard outputs.")
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--explanation-dir", type=Path, default=None)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    args = parser.parse_args()

    shard_dirs = sorted(
        path for path in args.shards_dir.iterdir() if path.is_dir() and path.name.startswith("shard_")
    )
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found under {args.shards_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged = {}
    for name in DETAIL_FILES:
        df = read_csvs(shard_dirs, name)
        target = args.output_dir / name
        write(df, target, DETAIL_FIELDS.get(name))
        merged[name.removesuffix(".csv")] = str(target)

    curves = pd.read_csv(args.output_dir / "faithfulness_curves.csv", low_memory=False)
    auc_by_case = pd.read_csv(args.output_dir / "faithfulness_auc_by_case.csv", low_memory=False)
    curve_summary = aggregate_faithfulness_curves(curves)
    auc_summary = aggregate_faithfulness_auc(auc_by_case)
    write(curve_summary, args.output_dir / "faithfulness_curve_summary.csv", FAITHFULNESS_CURVE_SUMMARY_FIELDS)
    write(auc_summary, args.output_dir / "faithfulness_auc_summary.csv", FAITHFULNESS_AUC_SUMMARY_FIELDS)
    merged["faithfulness_curve_summary"] = str(args.output_dir / "faithfulness_curve_summary.csv")
    merged["faithfulness_auc_summary"] = str(args.output_dir / "faithfulness_auc_summary.csv")

    bucket_cases = pd.read_csv(args.output_dir / "prediction_bucket_cases.csv", low_memory=False)
    if not bucket_cases.empty:
        bucket_summary_rows = []
        for bucket, part in bucket_cases.groupby("confidence_bucket", dropna=False):
            evidence_counts = part["top_evidence_type_from_rank1"].fillna("missing").value_counts()
            common = "; ".join(
                f"{name}:{count} ({count / len(part):.1%})"
                for name, count in evidence_counts.head(5).items()
            )
            bucket_summary_rows.append(
                {
                    "confidence_bucket": bucket,
                    "n_cases": int(len(part)),
                    "accuracy": float(part["correct"].astype(str).str.lower().isin(["true", "1"]).mean()),
                    "mean_confidence": float(pd.to_numeric(part["baseline_pred_proba"], errors="coerce").mean()),
                    "mean_top_abs_delta_pred_proba": float(pd.to_numeric(part["top_abs_delta_pred_proba"], errors="coerce").mean()),
                    "median_top_abs_delta_pred_proba": float(pd.to_numeric(part["top_abs_delta_pred_proba"], errors="coerce").median()),
                    "mean_evidence_purity": float(pd.to_numeric(part["evidence_purity"], errors="coerce").mean()),
                    "mean_support_train_n": float(pd.to_numeric(part["support_train_n"], errors="coerce").mean()),
                    "most_common_evidence_types": common,
                }
            )
        bucket_summary = pd.DataFrame(bucket_summary_rows).sort_values("confidence_bucket")

        evidence_rows = []
        bucket_sizes = bucket_cases.groupby("confidence_bucket")["case_index"].size().to_dict()
        for (bucket, evidence_type), part in bucket_cases.groupby(
            ["confidence_bucket", "top_evidence_type_from_rank1"],
            dropna=False,
        ):
            denom = bucket_sizes.get(bucket, len(part))
            evidence_rows.append(
                {
                    "confidence_bucket": bucket,
                    "evidence_type": evidence_type,
                    "n_cases": int(len(part)),
                    "bucket_share": float(len(part) / denom) if denom else None,
                    "mean_top_abs_delta_pred_proba": float(pd.to_numeric(part["top_abs_delta_pred_proba"], errors="coerce").mean()),
                    "mean_evidence_purity": float(pd.to_numeric(part["evidence_purity"], errors="coerce").mean()),
                    "mean_support_train_n": float(pd.to_numeric(part["support_train_n"], errors="coerce").mean()),
                }
            )
        bucket_evidence = pd.DataFrame(evidence_rows).sort_values(
            ["confidence_bucket", "n_cases"],
            ascending=[True, False],
        )
    elif args.explanation_dir:
        case_summary = pd.read_csv(args.explanation_dir / "case_summary.csv", low_memory=False)
        top_explanations = pd.read_csv(args.explanation_dir / "case_top_explanations.csv", low_memory=False)
        selected = case_summary["case_index"].astype(int).to_numpy()
        _, bucket_summary, bucket_evidence = build_prediction_bucket_outputs(
            case_summary=case_summary,
            top_explanations=top_explanations,
            selected_case_indices=selected,
            high_confidence_threshold=args.high_confidence_threshold,
        )
    else:
        bucket_summary = pd.DataFrame()
        bucket_evidence = pd.DataFrame()

    write(bucket_summary, args.output_dir / "prediction_bucket_summary.csv", BUCKET_SUMMARY_FIELDS)
    write(bucket_evidence, args.output_dir / "prediction_bucket_evidence_types.csv", BUCKET_EVIDENCE_FIELDS)
    merged["prediction_bucket_summary"] = str(args.output_dir / "prediction_bucket_summary.csv")
    merged["prediction_bucket_evidence_types"] = str(args.output_dir / "prediction_bucket_evidence_types.csv")

    run_summaries = load_run_summaries(shard_dirs)
    summary = {
        "shards_dir": str(args.shards_dir),
        "output_dir": str(args.output_dir),
        "explanation_dir": str(args.explanation_dir) if args.explanation_dir else None,
        "n_shards": len(shard_dirs),
        "processed_cases": int(sum(item.get("processed_cases", 0) for item in run_summaries)),
        "faithfulness_failures": [
            failure
            for item in run_summaries
            for failure in item.get("faithfulness_failures", [])
        ],
        "shard_summaries": run_summaries,
    }
    (args.output_dir / "validation_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    merged["validation_run_summary"] = str(args.output_dir / "validation_run_summary.json")
    (args.output_dir / "validation_manifest.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[validation merge done] shards={len(shard_dirs)} "
        f"cases={summary['processed_cases']} out={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
