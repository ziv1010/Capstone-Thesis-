#!/usr/bin/env python3
"""Summarize k-fold results across the GNN ablation matrix.

By default this prints a TSV table with one row per encoder, experiment, and
bucket. It can also write:

* outputs/master_ablation_results.csv: long format, one row per encoder/run.
* outputs/inlegalbert_vs_bge_comparison.csv: wide format, preserving the
  previous comparison columns while adding newly-run experiments.

Missing metrics are emitted as blank cells.

Usage:
  python3 summarize_bge_vs_inlegalbert.py
  python3 summarize_bge_vs_inlegalbert.py --markdown
  python3 summarize_bge_vs_inlegalbert.py --write-csvs
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECTION_GNN = Path(__file__).resolve().parent
BGE_ROOT = SECTION_GNN / "outputs" / "timed_bucket_runs"
INLEGAL_ROOT = SECTION_GNN / "outputs" / "inlegalbert_runs"
ABLATIONS_ROOT = SECTION_GNN / "outputs" / "ablations"
OUTPUTS_ROOT = SECTION_GNN / "outputs"

BUCKETS = [
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
    "cross_bucket_total_dataset",
]

EXPERIMENTS = [
    "baseline",
    "baseline_lr_decay",
    "no_names",
    "text_only",
    "no_cross_case",
    "hierarchical_enc",
    "section_sep_enc",
    "section_sep_enc_lr_decay",
    "case_node_minimised",
    "party_args_no_lr",
    "party_args_lr_decay",
    "party_args_preamble_no_lr",
    "party_args_preamble_lr_decay",
    "entity_resolved_party_args_preamble_lr_decay",
    "entity_resolved_section_sep_lr_decay",
    "central_authorities_removed_party_args_preamble_lr_decay",
    "central_authorities_removed_section_sep_no_lr",
]


@dataclass(frozen=True)
class RunSpec:
    encoder: str
    experiment: str
    bucket: str
    path: Path


def short_bucket(bucket: str) -> str:
    if bucket == "cross_bucket_total_dataset":
        return "cross_bucket"
    return bucket.removesuffix("_timed_mistral")


def bge_run_name(exp: str, bucket: str) -> str:
    short = short_bucket(bucket)
    if exp == "baseline":
        return f"{bucket}_kfold"
    if exp == "baseline_lr_decay":
        return f"{short}_baseline_lr_decay_kfold"
    if exp == "party_args_lr_decay":
        return f"{short}_party_args_lr_decay_kfold"
    if exp == "party_args_no_lr":
        return f"{short}_party_args_no_lr_kfold"
    if exp == "party_args_preamble_lr_decay":
        return f"{short}_party_args_preamble_lr_decay_kfold"
    if exp == "party_args_preamble_no_lr":
        return f"{short}_party_args_preamble_no_lr_kfold"
    if exp == "section_sep_enc_lr_decay":
        return f"ablation_section_sep_enc_lr_decay_{short}_kfold"
    return f"ablation_{exp}_{short}_kfold"


def inlegalbert_run_name(exp: str, bucket: str) -> str:
    return f"inlegalbert_{short_bucket(bucket)}_{exp}_kfold"


def ablation_run_name(exp: str, bucket: str) -> str | None:
    short = short_bucket(bucket)
    if exp == "entity_resolved_party_args_preamble_lr_decay":
        return f"{short}_entity_resolved_party_args_preamble_lr_decay_kfold"
    if exp == "entity_resolved_section_sep_lr_decay":
        return f"ablation_entity_resolved_section_sep_lr_decay_{short}_kfold"
    if exp == "central_authorities_removed_party_args_preamble_lr_decay":
        return f"{short}_central_authorities_removed_party_args_preamble_lr_decay_kfold"
    if exp == "central_authorities_removed_section_sep_no_lr":
        return f"ablation_central_authorities_removed_section_sep_no_lr_{short}_kfold"
    return None


def summary_path(encoder: str, exp: str, bucket: str) -> Path:
    if encoder == "bge-m3":
        ablation = ablation_run_name(exp, bucket)
        if ablation is not None:
            group = "entity_resolved_data" if exp.startswith("entity_resolved") else "remove_central_authorities"
            return ABLATIONS_ROOT / group / bucket / "models" / ablation / "kfold" / "kfold_summary.json"
        return BGE_ROOT / bucket / "models" / bge_run_name(exp, bucket) / "kfold" / "kfold_summary.json"
    if encoder == "inlegalbert":
        return INLEGAL_ROOT / bucket / "models" / inlegalbert_run_name(exp, bucket) / "kfold" / "kfold_summary.json"
    raise ValueError(encoder)


def load_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    return {
        "folds": f"{data.get('n_folds_completed', 0)}/{data.get('k', 0)}",
        "accuracy_mean": agg.get("accuracy_mean"),
        "accuracy_std": agg.get("accuracy_std"),
        "macro_f1_mean": agg.get("macro_f1_mean"),
        "macro_f1_std": agg.get("macro_f1_std"),
        "micro_f1_mean": agg.get("micro_f1_mean"),
        "micro_f1_std": agg.get("micro_f1_std"),
        "roc_auc_mean": agg.get("roc_auc_mean"),
        "roc_auc_std": agg.get("roc_auc_std"),
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def run_specs() -> list[RunSpec]:
    return [
        RunSpec(encoder, exp, bucket, summary_path(encoder, exp, bucket))
        for exp in EXPERIMENTS
        for bucket in BUCKETS
        for encoder in ("bge-m3", "inlegalbert")
    ]


def rows() -> list[list[str]]:
    table: list[list[str]] = []
    for spec in run_specs():
        metrics = load_metrics(spec.path)
        if metrics is None:
            table.append([spec.encoder, spec.experiment, spec.bucket, "", "", "", "", "", "", "", "", "", str(spec.path)])
            continue
        table.append(
            [
                spec.encoder,
                spec.experiment,
                spec.bucket,
                metrics["folds"],
                fmt(metrics["accuracy_mean"]),
                fmt(metrics["accuracy_std"]),
                fmt(metrics["macro_f1_mean"]),
                fmt(metrics["macro_f1_std"]),
                fmt(metrics["micro_f1_mean"]),
                fmt(metrics["micro_f1_std"]),
                fmt(metrics["roc_auc_mean"]),
                fmt(metrics["roc_auc_std"]),
                str(spec.path),
            ]
        )
    return table


LONG_HEADER = [
    "encoder",
    "experiment",
    "bucket",
    "folds",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "micro_f1_mean",
    "micro_f1_std",
    "roc_auc_mean",
    "roc_auc_std",
    "summary_path",
]


def print_tsv(table: list[list[str]]) -> None:
    print("\t".join(LONG_HEADER))
    for row in table:
        print("\t".join(row))


def print_markdown(table: list[list[str]]) -> None:
    header = [
        "Encoder",
        "Experiment",
        "Bucket",
        "Folds",
        "Accuracy",
        "Macro F1",
        "ROC AUC",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for row in table:
        encoder, exp, bucket, folds = row[:4]
        accuracy = row[4] if row[4] else "MISSING"
        macro_f1 = row[6] if row[6] else "MISSING"
        roc_auc = row[10] if row[10] else ""
        print(f"| {encoder} | {exp} | {bucket} | {folds} | {accuracy} | {macro_f1} | {roc_auc} |")


def write_long_csv(table: list[list[str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(LONG_HEADER)
        writer.writerows(table)


def write_wide_comparison_csv(table: list[list[str]], out_path: Path) -> None:
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in table:
        encoder, exp, bucket = row[:3]
        by_key[(exp, bucket, encoder)] = {
            "acc": row[4],
            "f1": row[6],
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["exp", "bucket", "in_acc", "bge_acc", "d_acc", "in_f1", "bge_f1", "d_f1"])
        for exp in EXPERIMENTS:
            for bucket in BUCKETS:
                short = short_bucket(bucket)
                in_metrics = by_key.get((exp, bucket, "inlegalbert"), {})
                bge_metrics = by_key.get((exp, bucket, "bge-m3"), {})
                in_acc = in_metrics.get("acc", "")
                bge_acc = bge_metrics.get("acc", "")
                in_f1 = in_metrics.get("f1", "")
                bge_f1 = bge_metrics.get("f1", "")
                d_acc = fmt(float(in_acc) - float(bge_acc)) if in_acc and bge_acc else ""
                d_f1 = fmt(float(in_f1) - float(bge_f1)) if in_f1 and bge_f1 else ""
                writer.writerow([exp, short, in_acc, bge_acc, d_acc, in_f1, bge_f1, d_f1])


def write_csvs(table: list[list[str]]) -> None:
    write_long_csv(table, OUTPUTS_ROOT / "master_ablation_results.csv")
    write_wide_comparison_csv(table, OUTPUTS_ROOT / "inlegalbert_vs_bge_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--write-csvs", action="store_true")
    args = parser.parse_args()

    table = rows()
    if args.write_csvs:
        write_csvs(table)
    if args.markdown:
        print_markdown(table)
    else:
        print_tsv(table)


if __name__ == "__main__":
    main()
