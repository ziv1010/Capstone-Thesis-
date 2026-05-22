#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import explain_hgt as hgt


DEFAULT_EXPLANATION_DIR = hgt.REPO_ROOT / "FINAL_EXPLANATION/outputs/target_fold_00_8gpu"

VALIDATION_CSVS = [
    "faithfulness_curves.csv",
    "faithfulness_curve_summary.csv",
    "faithfulness_auc_by_case.csv",
    "faithfulness_auc_summary.csv",
    "prediction_bucket_cases.csv",
    "prediction_bucket_summary.csv",
    "prediction_bucket_evidence_types.csv",
]

FAITHFULNESS_CURVE_FIELDS = [
    "case_index",
    "case_id",
    "ranker",
    "random_trial",
    "k_requested",
    "k_effective",
    "k_fraction",
    "n_groups",
    "baseline_pred_label",
    "baseline_pred_proba",
    "baseline_positive_proba",
    "sufficiency_proba",
    "sufficiency_positive_proba",
    "sufficiency_preserved_fraction",
    "sufficiency_pred_label",
    "sufficiency_flipped",
    "comprehensiveness_proba",
    "comprehensiveness_positive_proba",
    "comprehensiveness_drop",
    "comprehensiveness_drop_fraction",
    "comprehensiveness_pred_label",
    "comprehensiveness_flipped",
]
FAITHFULNESS_AUC_FIELDS = [
    "case_index",
    "case_id",
    "ranker",
    "random_trial",
    "n_groups",
    "max_k",
    "sufficiency_auc",
    "comprehensiveness_auc",
]
FAITHFULNESS_CURVE_SUMMARY_FIELDS = [
    "ranker",
    "k_requested",
    "n_cases",
    "n_rows",
    "mean_k_effective",
    "mean_sufficiency_proba",
    "mean_sufficiency_preserved_fraction",
    "mean_comprehensiveness_proba",
    "mean_comprehensiveness_drop",
    "mean_comprehensiveness_drop_fraction",
]
FAITHFULNESS_AUC_SUMMARY_FIELDS = [
    "ranker",
    "n_cases",
    "n_rows",
    "mean_sufficiency_auc",
    "median_sufficiency_auc",
    "mean_comprehensiveness_auc",
    "median_comprehensiveness_auc",
]
BUCKET_CASE_FIELDS = [
    "case_index",
    "case_id",
    "target_label",
    "baseline_pred_label",
    "baseline_pred_proba",
    "correct",
    "confidence_bucket",
    "top_evidence_type_from_rank1",
    "top_evidence_name_from_rank1",
    "top_path_family_from_rank1",
    "top_abs_delta_pred_proba",
    "support_train_n",
    "evidence_purity",
]
BUCKET_SUMMARY_FIELDS = [
    "confidence_bucket",
    "n_cases",
    "accuracy",
    "mean_confidence",
    "mean_top_abs_delta_pred_proba",
    "median_top_abs_delta_pred_proba",
    "mean_evidence_purity",
    "mean_support_train_n",
    "most_common_evidence_types",
]
BUCKET_EVIDENCE_FIELDS = [
    "confidence_bucket",
    "evidence_type",
    "n_cases",
    "bucket_share",
    "mean_top_abs_delta_pred_proba",
    "mean_evidence_purity",
    "mean_support_train_n",
]


def parse_k_values(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or values[0] != 0:
        values = [0] + values
    return values


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_df(path: Path, df: pd.DataFrame, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty and fieldnames is not None:
        df = pd.DataFrame(columns=fieldnames)
    df.to_csv(path, index=False)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for filename in VALIDATION_CSVS + ["validation_run_summary.json", "validation_manifest.json"]:
        target = path / filename
        if target.exists():
            target.unlink()


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except Exception:
        return None
    if math.isnan(result):
        return None
    return result


def bool_from_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def all_edge_positions(local_rf: hgt.LocalRF) -> dict[tuple[str, str, str], Tensor]:
    return {
        edge_type: torch.arange(edge_index.shape[1], dtype=torch.long)
        for edge_type, edge_index in local_rf.edge_index_dict.items()
        if edge_index.shape[1] > 0
    }


def union_mask_positions(groups: list[hgt.MaskGroup]) -> dict[tuple[str, str, str], Tensor]:
    chunks: dict[tuple[str, str, str], list[Tensor]] = defaultdict(list)
    for group in groups:
        for edge_type, positions in group.mask_positions.items():
            if positions.numel() > 0:
                chunks[edge_type].append(positions.detach().cpu().to(torch.long))
    result: dict[tuple[str, str, str], Tensor] = {}
    for edge_type, tensors in chunks.items():
        result[edge_type] = torch.cat(tensors).unique(sorted=True)
    return result


def complement_mask_positions(
    local_rf: hgt.LocalRF,
    keep_positions: dict[tuple[str, str, str], Tensor],
) -> dict[tuple[str, str, str], Tensor]:
    result: dict[tuple[str, str, str], Tensor] = {}
    for edge_type, edge_index in local_rf.edge_index_dict.items():
        edge_count = edge_index.shape[1]
        if edge_count == 0:
            continue
        mask = torch.ones(edge_count, dtype=torch.bool)
        keep = keep_positions.get(edge_type)
        if keep is not None and keep.numel() > 0:
            keep = keep[(keep >= 0) & (keep < edge_count)]
            mask[keep] = False
        result[edge_type] = torch.arange(edge_count, dtype=torch.long)[mask]
    return result


def make_combined_masked_batch(
    local_rf: hgt.LocalRF,
    mask_specs: list[dict[tuple[str, str, str], Tensor]],
) -> tuple[dict[str, Tensor], dict[tuple[str, str, str], Tensor], list[int]]:
    n_variants = len(mask_specs)
    node_counts = {node_type: int(x.shape[0]) for node_type, x in local_rf.x_dict.items()}
    x_dict = {node_type: x.repeat((n_variants, 1)) for node_type, x in local_rf.x_dict.items()}
    edge_index_dict: dict[tuple[str, str, str], Tensor] = {}

    for edge_type, base_edge_index in local_rf.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        base_edge_count = base_edge_index.shape[1]
        chunks = []
        for variant_idx, mask_spec in enumerate(mask_specs):
            keep = torch.ones(base_edge_count, dtype=torch.bool)
            masked_positions = mask_spec.get(edge_type)
            if masked_positions is not None and masked_positions.numel() > 0:
                masked_positions = masked_positions[(masked_positions >= 0) & (masked_positions < base_edge_count)]
                keep[masked_positions] = False
            if not bool(keep.any()):
                continue
            shifted = base_edge_index[:, keep].clone()
            shifted[0] += variant_idx * node_counts[src_type]
            shifted[1] += variant_idx * node_counts[dst_type]
            chunks.append(shifted)
        edge_index_dict[edge_type] = (
            torch.cat(chunks, dim=1) if chunks else torch.empty((2, 0), dtype=torch.long)
        )

    target_indices = [
        local_rf.target_case_local + variant_idx * node_counts["case"]
        for variant_idx in range(n_variants)
    ]
    return x_dict, edge_index_dict, target_indices


@torch.no_grad()
def forward_mask_specs(
    model: torch.nn.Module,
    local_rf: hgt.LocalRF,
    mask_specs: list[dict[tuple[str, str, str], Tensor]],
    device: torch.device,
    batch_size: int,
) -> Tensor:
    chunks = []
    for start in range(0, len(mask_specs), batch_size):
        current = mask_specs[start : start + batch_size]
        batch_x, batch_edges, target_indices = make_combined_masked_batch(local_rf, current)
        chunks.append(
            hgt.forward_local_cases(
                model=model,
                x_dict=batch_x,
                edge_index_dict=batch_edges,
                case_indices=target_indices,
                device=device,
            )
        )
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))


def unique_ranked_group_ids(rows: pd.DataFrame, score_column: str, descending: bool = True) -> list[str]:
    if rows.empty or score_column not in rows:
        return []
    work = rows.copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work = work.dropna(subset=[score_column])
    work = work.sort_values(score_column, ascending=not descending, na_position="last")
    seen = set()
    result = []
    for group_id in work["group_id"].astype(str).tolist():
        if group_id not in seen:
            seen.add(group_id)
            result.append(group_id)
    return result


def random_group_ids(group_ids: list[str], seed: int, case_index: int, trial: int) -> list[str]:
    rng = np.random.default_rng(seed + int(case_index) * 1009 + trial * 9176)
    result = list(group_ids)
    rng.shuffle(result)
    return result


def auc_from_points(points: list[tuple[float, float]]) -> float | None:
    clean = [(float(x), float(y)) for x, y in points if numeric(x) is not None and numeric(y) is not None]
    if len(clean) < 2:
        return None
    clean.sort(key=lambda item: item[0])
    y_values = [item[1] for item in clean]
    x_values = [item[0] for item in clean]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def aggregate_faithfulness_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()
    work = curves.copy()
    numeric_cols = [
        "k_requested",
        "k_effective",
        "sufficiency_proba",
        "sufficiency_preserved_fraction",
        "comprehensiveness_proba",
        "comprehensiveness_drop",
        "comprehensiveness_drop_fraction",
    ]
    for column in numeric_cols:
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    return (
        work.groupby(["ranker", "k_requested"], dropna=False)
        .agg(
            n_cases=("case_index", "nunique"),
            n_rows=("case_index", "size"),
            mean_k_effective=("k_effective", "mean"),
            mean_sufficiency_proba=("sufficiency_proba", "mean"),
            mean_sufficiency_preserved_fraction=("sufficiency_preserved_fraction", "mean"),
            mean_comprehensiveness_proba=("comprehensiveness_proba", "mean"),
            mean_comprehensiveness_drop=("comprehensiveness_drop", "mean"),
            mean_comprehensiveness_drop_fraction=("comprehensiveness_drop_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["ranker", "k_requested"])
    )


def aggregate_faithfulness_auc(auc_rows: pd.DataFrame) -> pd.DataFrame:
    if auc_rows.empty:
        return pd.DataFrame()
    work = auc_rows.copy()
    for column in ["sufficiency_auc", "comprehensiveness_auc"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return (
        work.groupby("ranker", dropna=False)
        .agg(
            n_cases=("case_index", "nunique"),
            n_rows=("case_index", "size"),
            mean_sufficiency_auc=("sufficiency_auc", "mean"),
            median_sufficiency_auc=("sufficiency_auc", "median"),
            mean_comprehensiveness_auc=("comprehensiveness_auc", "mean"),
            median_comprehensiveness_auc=("comprehensiveness_auc", "median"),
        )
        .reset_index()
        .sort_values("mean_comprehensiveness_auc", ascending=False)
    )


def bucket_name(confidence: float, correct: bool, threshold: float) -> str:
    if confidence >= threshold and correct:
        return "high_confidence_correct"
    if confidence >= threshold and not correct:
        return "high_confidence_wrong"
    return "low_confidence"


def build_prediction_bucket_outputs(
    case_summary: pd.DataFrame,
    top_explanations: pd.DataFrame,
    selected_case_indices: np.ndarray,
    high_confidence_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if case_summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    summary = case_summary[case_summary["case_index"].astype(int).isin(set(map(int, selected_case_indices)))].copy()
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for column in ["baseline_pred_proba", "max_abs_delta_pred_proba"]:
        if column in summary:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")

    top = top_explanations.copy()
    if not top.empty and "group_rank_abs" in top:
        top["group_rank_abs"] = pd.to_numeric(top["group_rank_abs"], errors="coerce")
        top = top.sort_values(["case_index", "group_rank_abs"]).drop_duplicates("case_index", keep="first")
    top_cols = [
        "case_index",
        "evidence_type",
        "evidence_name",
        "path_family",
        "abs_delta_pred_proba",
        "support_train_n",
        "support_positive_rate",
        "support_negative_rate",
    ]
    top = top[[column for column in top_cols if column in top.columns]] if not top.empty else pd.DataFrame(columns=top_cols)
    for column in top_cols:
        if column not in top:
            top[column] = np.nan
    merged = summary.merge(top, on="case_index", how="left", suffixes=("", "_top"))

    for column in ["abs_delta_pred_proba", "support_train_n", "support_positive_rate", "support_negative_rate"]:
        if column in merged:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    merged["correct"] = merged["target_label"].astype(str) == merged["baseline_pred_label"].astype(str)
    merged["confidence_bucket"] = [
        bucket_name(float(conf or 0.0), bool(correct), high_confidence_threshold)
        for conf, correct in zip(merged["baseline_pred_proba"], merged["correct"])
    ]
    merged["evidence_purity"] = merged[["support_positive_rate", "support_negative_rate"]].max(axis=1)
    merged.loc[merged["support_train_n"].fillna(0) <= 0, "evidence_purity"] = np.nan

    case_rows = merged.rename(
        columns={
            "evidence_type": "top_evidence_type_from_rank1",
            "evidence_name": "top_evidence_name_from_rank1",
            "path_family": "top_path_family_from_rank1",
            "abs_delta_pred_proba": "top_abs_delta_pred_proba",
        }
    )
    keep_cols = [
        "case_index",
        "case_id",
        "target_label",
        "baseline_pred_label",
        "baseline_pred_proba",
        "correct",
        "confidence_bucket",
        "top_evidence_type_from_rank1",
        "top_evidence_name_from_rank1",
        "top_path_family_from_rank1",
        "top_abs_delta_pred_proba",
        "support_train_n",
        "evidence_purity",
    ]
    case_rows = case_rows[[column for column in keep_cols if column in case_rows.columns]]

    summary_rows = []
    for bucket, part in case_rows.groupby("confidence_bucket", dropna=False):
        evidence_counts = part["top_evidence_type_from_rank1"].fillna("missing").value_counts()
        common = "; ".join(
            f"{name}:{count} ({count / len(part):.1%})"
            for name, count in evidence_counts.head(5).items()
        )
        summary_rows.append(
            {
                "confidence_bucket": bucket,
                "n_cases": int(len(part)),
                "accuracy": float(part["correct"].mean()) if len(part) else None,
                "mean_confidence": float(part["baseline_pred_proba"].mean()),
                "mean_top_abs_delta_pred_proba": float(part["top_abs_delta_pred_proba"].mean()),
                "median_top_abs_delta_pred_proba": float(part["top_abs_delta_pred_proba"].median()),
                "mean_evidence_purity": float(part["evidence_purity"].mean()),
                "mean_support_train_n": float(part["support_train_n"].mean()),
                "most_common_evidence_types": common,
            }
        )
    bucket_summary = pd.DataFrame(summary_rows).sort_values("confidence_bucket")

    evidence_rows = []
    bucket_sizes = case_rows.groupby("confidence_bucket")["case_index"].size().to_dict()
    for (bucket, evidence_type), part in case_rows.groupby(
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
                "mean_top_abs_delta_pred_proba": float(part["top_abs_delta_pred_proba"].mean()),
                "mean_evidence_purity": float(part["evidence_purity"].mean()),
                "mean_support_train_n": float(part["support_train_n"].mean()),
            }
        )
    bucket_evidence = pd.DataFrame(evidence_rows)
    if not bucket_evidence.empty:
        bucket_evidence = bucket_evidence.sort_values(
            ["confidence_bucket", "n_cases"],
            ascending=[True, False],
        )
    return case_rows, bucket_summary, bucket_evidence


def run_faithfulness(
    args: argparse.Namespace,
    selected_case_indices: np.ndarray,
    data: Any,
    node_ids: dict[str, list[str]],
    label_names: list[str],
    model: torch.nn.Module,
    adjacency: hgt.SortedAdjacency,
    counterfactual_rows: pd.DataFrame,
    device: torch.device,
    num_hops: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    k_values = parse_k_values(args.k_values)
    max_k = max(k_values) if k_values else 1
    curve_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    cf_by_case = {
        int(case_index): part.copy()
        for case_index, part in counterfactual_rows.groupby(counterfactual_rows["case_index"].astype(int))
    }
    positive_index = label_names.index("1") if "1" in label_names else len(label_names) - 1

    for ordinal, case_index_np in enumerate(selected_case_indices, start=1):
        case_index = int(case_index_np)
        try:
            rows = cf_by_case.get(case_index)
            if rows is None or rows.empty:
                continue
            selected_nodes, selected_edges = hgt.extract_l_hop_induced_receptive_field(
                data=data,
                adjacency=adjacency,
                case_index=case_index,
                num_hops=num_hops,
            )
            local_rf = hgt.build_local_rf(data, selected_nodes, selected_edges, case_index)
            groups = hgt.build_mask_groups(
                local_rf=local_rf,
                node_ids=node_ids,
                num_hops=num_hops,
                include_relation_groups=args.include_relation_groups,
                include_node_groups=args.include_node_groups,
            )
            group_by_id = {group.group_id: group for group in groups}
            if not group_by_id:
                continue

            baseline_probs = hgt.forward_local_cases(
                model=model,
                x_dict=local_rf.x_dict,
                edge_index_dict=local_rf.edge_index_dict,
                case_indices=[local_rf.target_case_local],
                device=device,
            )[0]
            baseline_pred_idx = int(torch.argmax(baseline_probs).item())
            baseline_pred_label = label_names[baseline_pred_idx]
            baseline_pred_proba = float(baseline_probs[baseline_pred_idx].item())
            baseline_positive_proba = float(baseline_probs[positive_index].item())

            cf_ids = [
                group_id for group_id in unique_ranked_group_ids(rows, "abs_delta_pred_proba")
                if group_id in group_by_id
            ]
            attention_ids = [
                group_id for group_id in unique_ranked_group_ids(rows, "attention_score")
                if group_id in group_by_id
            ]
            rankings: list[tuple[str, int | None, list[str]]] = [
                ("counterfactual", None, cf_ids),
            ]
            if attention_ids:
                rankings.append(("attention", None, attention_ids))
            all_group_ids = list(group_by_id)
            for trial in range(args.random_trials):
                rankings.append(("random", trial, random_group_ids(all_group_ids, args.random_seed, case_index, trial)))

            for ranker, trial, ranked_ids in rankings:
                if not ranked_ids:
                    continue
                variant_meta = []
                mask_specs = []
                for k_requested in k_values:
                    k_effective = min(int(k_requested), len(ranked_ids))
                    selected_groups = [group_by_id[group_id] for group_id in ranked_ids[:k_effective]]
                    top_union = union_mask_positions(selected_groups)
                    sufficiency_mask = complement_mask_positions(local_rf, top_union)
                    comprehensiveness_mask = top_union
                    variant_meta.append(("sufficiency", k_requested, k_effective))
                    mask_specs.append(sufficiency_mask)
                    variant_meta.append(("comprehensiveness", k_requested, k_effective))
                    mask_specs.append(comprehensiveness_mask)

                probs = forward_mask_specs(
                    model=model,
                    local_rf=local_rf,
                    mask_specs=mask_specs,
                    device=device,
                    batch_size=args.faithfulness_batch_size,
                )
                values_by_k: dict[int, dict[str, Any]] = {}
                for meta, prob_vec in zip(variant_meta, probs):
                    metric, k_requested, k_effective = meta
                    pred_prob = float(prob_vec[baseline_pred_idx].item())
                    positive_prob = float(prob_vec[positive_index].item())
                    pred_label = label_names[int(torch.argmax(prob_vec).item())]
                    row = values_by_k.setdefault(
                        k_requested,
                        {
                            "case_index": case_index,
                            "case_id": data["case"].case_id[case_index],
                            "ranker": ranker,
                            "random_trial": trial,
                            "k_requested": k_requested,
                            "k_effective": k_effective,
                            "k_fraction": (k_requested / max_k) if max_k else 0.0,
                            "n_groups": len(ranked_ids),
                            "baseline_pred_label": baseline_pred_label,
                            "baseline_pred_proba": baseline_pred_proba,
                            "baseline_positive_proba": baseline_positive_proba,
                        },
                    )
                    if metric == "sufficiency":
                        row["sufficiency_proba"] = pred_prob
                        row["sufficiency_positive_proba"] = positive_prob
                        row["sufficiency_preserved_fraction"] = (
                            pred_prob / baseline_pred_proba if baseline_pred_proba else None
                        )
                        row["sufficiency_pred_label"] = pred_label
                        row["sufficiency_flipped"] = pred_label != baseline_pred_label
                    else:
                        row["comprehensiveness_proba"] = pred_prob
                        row["comprehensiveness_positive_proba"] = positive_prob
                        row["comprehensiveness_drop"] = baseline_pred_proba - pred_prob
                        row["comprehensiveness_drop_fraction"] = (
                            (baseline_pred_proba - pred_prob) / baseline_pred_proba
                            if baseline_pred_proba
                            else None
                        )
                        row["comprehensiveness_pred_label"] = pred_label
                        row["comprehensiveness_flipped"] = pred_label != baseline_pred_label

                ranker_rows = [values_by_k[k] for k in sorted(values_by_k)]
                curve_rows.extend(ranker_rows)
                suff_auc = auc_from_points(
                    [(row["k_fraction"], row.get("sufficiency_proba")) for row in ranker_rows]
                )
                comp_auc = auc_from_points(
                    [(row["k_fraction"], row.get("comprehensiveness_drop_fraction")) for row in ranker_rows]
                )
                auc_rows.append(
                    {
                        "case_index": case_index,
                        "case_id": data["case"].case_id[case_index],
                        "ranker": ranker,
                        "random_trial": trial,
                        "n_groups": len(ranked_ids),
                        "max_k": max_k,
                        "sufficiency_auc": suff_auc,
                        "comprehensiveness_auc": comp_auc,
                    }
                )

            if ordinal % args.progress_every == 0:
                print(
                    f"[faithfulness] cases={ordinal}/{len(selected_case_indices)} "
                    f"curve_rows={len(curve_rows)}",
                    flush=True,
                )
        except Exception as exc:
            failures.append(
                {
                    "case_index": case_index,
                    "case_id": data["case"].case_id[case_index],
                    "error": repr(exc),
                }
            )
            print(f"[warn] faithfulness case_index={case_index} failed: {exc!r}", flush=True)

    return curve_rows, auc_rows, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation analyses for FINAL_EXPLANATION outputs: faithfulness curves and prediction buckets."
    )
    parser.add_argument("--model-path", type=Path, default=hgt.DEFAULT_MODEL_PATH)
    parser.add_argument("--graph-cache", type=Path, default=hgt.DEFAULT_GRAPH_CACHE)
    parser.add_argument("--config", type=Path, default=hgt.DEFAULT_CONFIG_PATH)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --explanation-dir. New validation CSVs are added there.",
    )
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-hops", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k-values", default="0,1,2,3,5,10,20")
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--faithfulness-batch-size", type=int, default=64)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--skip-faithfulness", action="store_true")
    parser.add_argument("--skip-buckets", action="store_true")
    parser.add_argument("--include-relation-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-node-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-validation", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_time = time.time()
    output_dir = args.output_dir or args.explanation_dir
    prepare_output_dir(output_dir, overwrite=args.overwrite_validation)

    counterfactual_path = args.explanation_dir / "case_counterfactual_groups.csv"
    top_path = args.explanation_dir / "case_top_explanations.csv"
    case_summary_path = args.explanation_dir / "case_summary.csv"
    if not case_summary_path.exists():
        raise FileNotFoundError(f"Missing case summary: {case_summary_path}")
    if not counterfactual_path.exists() and not args.skip_faithfulness:
        raise FileNotFoundError(f"Missing counterfactual groups: {counterfactual_path}")

    case_summary = pd.read_csv(case_summary_path, low_memory=False)
    top_explanations = pd.read_csv(top_path, low_memory=False) if top_path.exists() else pd.DataFrame()
    all_case_indices = case_summary["case_index"].astype(int).to_numpy()
    all_case_indices = all_case_indices[args.shard_index :: args.num_shards]
    if args.case_start:
        all_case_indices = all_case_indices[args.case_start :]
    if args.case_limit is not None:
        all_case_indices = all_case_indices[: args.case_limit]

    print(
        f"[validation] cases={len(all_case_indices)} shard={args.shard_index}/{args.num_shards} "
        f"explanations={args.explanation_dir} out={output_dir}",
        flush=True,
    )

    faithfulness_failures: list[dict[str, Any]] = []
    if not args.skip_faithfulness and len(all_case_indices):
        cfg = hgt.load_yaml(args.config)
        num_hops = args.num_hops or int(cfg.get("model", {}).get("num_layers", 2))
        device = hgt.choose_device(args.device)
        predictions_csv = args.predictions_csv or (args.model_path.parent / "predictions.csv")

        print(f"[load] graph={args.graph_cache}", flush=True)
        bundle = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
        data = bundle["data"]
        metadata = bundle["metadata"]
        label_names = [str(label) for label in metadata["label_names"]]
        node_ids = {node_type: list(data[node_type].node_id) for node_type in data.node_types}
        print(f"[load] model={args.model_path}", flush=True)
        model = hgt.load_model(
            data=data,
            label_names=label_names,
            cfg=cfg,
            model_path=args.model_path,
            device=device,
            record_attention=False,
        )
        adjacency = hgt.SortedAdjacency(data)
        counterfactual_rows = pd.read_csv(counterfactual_path, low_memory=False)
        counterfactual_rows = counterfactual_rows[
            counterfactual_rows["case_index"].astype(int).isin(set(map(int, all_case_indices)))
        ].copy()
        curve_rows, auc_rows, faithfulness_failures = run_faithfulness(
            args=args,
            selected_case_indices=all_case_indices,
            data=data,
            node_ids=node_ids,
            label_names=label_names,
            model=model,
            adjacency=adjacency,
            counterfactual_rows=counterfactual_rows,
            device=device,
            num_hops=num_hops,
        )
        write_csv(output_dir / "faithfulness_curves.csv", curve_rows, FAITHFULNESS_CURVE_FIELDS)
        write_csv(output_dir / "faithfulness_auc_by_case.csv", auc_rows, FAITHFULNESS_AUC_FIELDS)
        curve_summary = aggregate_faithfulness_curves(pd.DataFrame(curve_rows))
        auc_summary = aggregate_faithfulness_auc(pd.DataFrame(auc_rows))
        write_df(output_dir / "faithfulness_curve_summary.csv", curve_summary, FAITHFULNESS_CURVE_SUMMARY_FIELDS)
        write_df(output_dir / "faithfulness_auc_summary.csv", auc_summary, FAITHFULNESS_AUC_SUMMARY_FIELDS)
    else:
        write_csv(output_dir / "faithfulness_curves.csv", [], FAITHFULNESS_CURVE_FIELDS)
        write_csv(output_dir / "faithfulness_auc_by_case.csv", [], FAITHFULNESS_AUC_FIELDS)
        write_df(output_dir / "faithfulness_curve_summary.csv", pd.DataFrame(), FAITHFULNESS_CURVE_SUMMARY_FIELDS)
        write_df(output_dir / "faithfulness_auc_summary.csv", pd.DataFrame(), FAITHFULNESS_AUC_SUMMARY_FIELDS)

    if not args.skip_buckets:
        bucket_cases, bucket_summary, bucket_evidence = build_prediction_bucket_outputs(
            case_summary=case_summary,
            top_explanations=top_explanations,
            selected_case_indices=all_case_indices,
            high_confidence_threshold=args.high_confidence_threshold,
        )
        write_df(output_dir / "prediction_bucket_cases.csv", bucket_cases, BUCKET_CASE_FIELDS)
        write_df(output_dir / "prediction_bucket_summary.csv", bucket_summary, BUCKET_SUMMARY_FIELDS)
        write_df(output_dir / "prediction_bucket_evidence_types.csv", bucket_evidence, BUCKET_EVIDENCE_FIELDS)
    else:
        write_csv(output_dir / "prediction_bucket_cases.csv", [], BUCKET_CASE_FIELDS)
        write_csv(output_dir / "prediction_bucket_summary.csv", [], BUCKET_SUMMARY_FIELDS)
        write_csv(output_dir / "prediction_bucket_evidence_types.csv", [], BUCKET_EVIDENCE_FIELDS)

    manifest = {
        "faithfulness_curves": str(output_dir / "faithfulness_curves.csv"),
        "faithfulness_curve_summary": str(output_dir / "faithfulness_curve_summary.csv"),
        "faithfulness_auc_by_case": str(output_dir / "faithfulness_auc_by_case.csv"),
        "faithfulness_auc_summary": str(output_dir / "faithfulness_auc_summary.csv"),
        "prediction_bucket_cases": str(output_dir / "prediction_bucket_cases.csv"),
        "prediction_bucket_summary": str(output_dir / "prediction_bucket_summary.csv"),
        "prediction_bucket_evidence_types": str(output_dir / "prediction_bucket_evidence_types.csv"),
        "validation_run_summary": str(output_dir / "validation_run_summary.json"),
    }
    dump_json(output_dir / "validation_manifest.json", manifest)
    dump_json(
        output_dir / "validation_run_summary.json",
        {
            "model_path": str(args.model_path),
            "graph_cache": str(args.graph_cache),
            "config": str(args.config),
            "predictions_csv": str(args.predictions_csv or (args.model_path.parent / "predictions.csv")),
            "explanation_dir": str(args.explanation_dir),
            "output_dir": str(output_dir),
            "case_limit": args.case_limit,
            "case_start": args.case_start,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "k_values": parse_k_values(args.k_values),
            "random_trials": args.random_trials,
            "high_confidence_threshold": args.high_confidence_threshold,
            "processed_cases": int(len(all_case_indices)),
            "faithfulness_failures": faithfulness_failures,
            "elapsed_seconds": time.time() - start_time,
        },
    )
    print(
        f"[validation done] cases={len(all_case_indices)} out={output_dir} "
        f"elapsed_min={(time.time() - start_time) / 60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
