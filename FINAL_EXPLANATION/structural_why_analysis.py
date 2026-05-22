#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import power_divergence

from pattern_explainability_common import (
    DEFAULT_EXPLANATION_DIR,
    DEFAULT_PATTERN_OUTPUT_DIR,
    STRUCTURAL_FEATURE_TYPES,
    build_case_feature_matrix,
    build_case_metadata,
    dump_json,
    infer_artifact_paths,
    load_graph,
    load_predictions,
    normalise_label,
    require_file,
    save_case_feature_artifacts,
    select_case_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build rarity-weighted case-case structural communities, score evidence "
            "label skew, and summarize community-level success/failure."
        )
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATTERN_OUTPUT_DIR)
    parser.add_argument("--split", default="all", help="Case split for community assignment; default all.")
    parser.add_argument("--case-limit", type=int, default=None, help="Smoke-test limit after split filtering.")
    parser.add_argument(
        "--feature-types",
        default="statute,provision,precedent,judge,court",
        help="Comma-separated structural feature node types.",
    )
    parser.add_argument("--min-feature-cases", type=int, default=2)
    parser.add_argument(
        "--max-feature-cases",
        type=int,
        default=5000,
        help=(
            "Prune extreme hub features before projection to keep the graph interpretable. "
            "Set to 0 or a negative value for no upper cap."
        ),
    )
    parser.add_argument("--min-edge-weight", type=float, default=0.0)
    parser.add_argument("--top-k-neighbors", type=int, default=80)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--leiden-iterations", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-features-per-type", type=int, default=10)
    parser.add_argument("--representative-cases", type=int, default=5)
    parser.add_argument("--skew-alpha", type=float, default=0.05)
    parser.add_argument("--skew-min-support", type=int, default=20)
    parser.add_argument("--skew-effect-min", type=float, default=0.10)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.80)
    return parser


def prune_topk_csr(matrix: sp.csr_matrix, top_k: int) -> sp.csr_matrix:
    matrix = matrix.tocsr()
    if top_k <= 0:
        raise ValueError("--top-k-neighbors must be positive.")
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        row_indices = matrix.indices[start:end]
        row_data = matrix.data[start:end]
        if row_data.size > top_k:
            keep = np.argpartition(row_data, -top_k)[-top_k:]
            order = keep[np.argsort(row_data[keep])[::-1]]
            row_indices = row_indices[order]
            row_data = row_data[order]
        indices.extend(row_indices.tolist())
        data.extend(row_data.astype(float).tolist())
        indptr.append(len(indices))
    pruned = sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=matrix.shape,
    )
    return pruned


def build_projection(
    feature_matrix: sp.csr_matrix,
    feature_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    counts = feature_df["corpus_case_count"].to_numpy(dtype=np.int64)
    keep_mask = counts >= args.min_feature_cases
    if args.max_feature_cases and args.max_feature_cases > 0:
        keep_mask &= counts <= args.max_feature_cases
    if not bool(keep_mask.any()):
        raise ValueError("No features survived min/max case-count projection filters.")
    kept_features = feature_df.loc[keep_mask].copy()
    kept_matrix = feature_matrix[:, keep_mask].astype(np.float32).tocsr()
    idf = kept_features["idf"].to_numpy(dtype=np.float32)
    weighted = kept_matrix @ sp.diags(np.sqrt(idf), dtype=np.float32)
    projection = (weighted @ weighted.T).tocsr()
    projection.setdiag(0.0)
    projection.eliminate_zeros()
    if args.min_edge_weight > 0:
        projection.data[projection.data < args.min_edge_weight] = 0.0
        projection.eliminate_zeros()
    projection = prune_topk_csr(projection, args.top_k_neighbors)
    projection = projection.maximum(projection.T).tocsr()
    projection.eliminate_zeros()
    if projection.nnz == 0:
        raise ValueError("Case-case projection has no edges after pruning.")
    return projection, kept_features


def run_leiden(projection: sp.csr_matrix, args: argparse.Namespace) -> tuple[np.ndarray, list[float]]:
    upper = sp.triu(projection, k=1).tocoo()
    edges = list(zip(upper.row.astype(int).tolist(), upper.col.astype(int).tolist(), strict=False))
    graph = ig.Graph(n=projection.shape[0], edges=edges, directed=False)
    graph.es["weight"] = upper.data.astype(float).tolist()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=args.resolution,
        n_iterations=args.leiden_iterations,
        seed=args.seed,
    )
    pagerank = graph.pagerank(weights="weight")
    return np.asarray(partition.membership, dtype=np.int64), pagerank


def label_count_columns(labels: pd.Series) -> dict[str, int]:
    counts = Counter(labels.astype(str))
    return {f"label_{label}_n": int(count) for label, count in sorted(counts.items())}


def top_feature_text(
    feature_df: pd.DataFrame,
    counts: np.ndarray,
    feature_type: str,
    limit: int,
) -> str:
    typed = feature_df.index[feature_df["feature_type"].astype(str) == feature_type].to_numpy()
    if typed.size == 0:
        return ""
    typed_counts = counts[typed]
    nonzero = typed_counts > 0
    typed = typed[nonzero]
    typed_counts = typed_counts[nonzero]
    if typed.size == 0:
        return ""
    order = np.argsort(typed_counts)[::-1][:limit]
    parts = []
    for idx in order:
        feature_index = int(typed[idx])
        name = str(feature_df.iloc[feature_index]["feature_name"])
        parts.append(f"{name} ({int(typed_counts[idx])})")
    return " | ".join(parts)


def summarize_communities(
    case_meta: pd.DataFrame,
    selected_case_indices: np.ndarray,
    membership: np.ndarray,
    pagerank: list[float],
    feature_matrix: sp.csr_matrix,
    feature_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_meta = case_meta.set_index("case_index").loc[selected_case_indices].reset_index()
    selected_meta["community_id"] = membership
    selected_meta["pagerank"] = np.asarray(pagerank, dtype=float)

    profile_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    feature_df = feature_df.reset_index(drop=True)

    for community_id, group in selected_meta.groupby("community_id", sort=True):
        row_positions = group.index.to_numpy(dtype=np.int64)
        size = int(len(group))
        labels = group["target_label"].astype(str)
        preds = group["pred_label"].astype(str)
        confidences = pd.to_numeric(group["confidence"], errors="coerce")
        correct = labels == preds
        label_counts = label_count_columns(labels)
        dominant_label = labels.value_counts().idxmax()
        dominant_domain = group["domain_bucket"].value_counts().idxmax()
        feature_counts = np.asarray(feature_matrix[row_positions].sum(axis=0)).ravel()
        top_rep = group.sort_values("pagerank", ascending=False).head(args.representative_cases)
        representative_indices = "|".join(str(int(value)) for value in top_rep["case_index"])
        representative_cases = " | ".join(str(value) for value in top_rep["case_id"])

        base = {
            "community_id": int(community_id),
            "size": size,
            "dominant_label": dominant_label,
            "dominant_domain_bucket": dominant_domain,
            "accuracy": float(correct.mean()) if size else np.nan,
            "mean_confidence": float(confidences.mean()) if confidences.notna().any() else np.nan,
            "median_confidence": float(confidences.median()) if confidences.notna().any() else np.nan,
            "high_confidence_wrong_n": int(((~correct) & (confidences >= args.high_confidence_threshold)).sum()),
            "representative_case_indices": representative_indices,
            "representative_cases": representative_cases,
        }
        base.update(label_counts)
        for label, count in Counter(labels).items():
            base[f"label_{label}_rate"] = float(count / size)
        for split, count in Counter(group["split"].astype(str)).items():
            base[f"split_{split}_n"] = int(count)
        for feature_type in sorted(STRUCTURAL_FEATURE_TYPES):
            base[f"top_{feature_type}s"] = top_feature_text(
                feature_df,
                feature_counts,
                feature_type,
                args.top_features_per_type,
            )
        profile_rows.append(base)

        confusion = Counter(zip(labels, preds, strict=False))
        success = {
            "community_id": int(community_id),
            "size": size,
            "accuracy": base["accuracy"],
            "mean_confidence": base["mean_confidence"],
            "high_confidence_wrong_n": base["high_confidence_wrong_n"],
        }
        for (target, pred), count in sorted(confusion.items()):
            success[f"target_{target}_pred_{pred}_n"] = int(count)
        success_rows.append(success)

        nonzero = np.flatnonzero(feature_counts > 0)
        for feature_type in sorted(STRUCTURAL_FEATURE_TYPES):
            typed = [
                idx
                for idx in nonzero
                if str(feature_df.iloc[int(idx)]["feature_type"]) == feature_type
            ]
            typed.sort(key=lambda idx: feature_counts[int(idx)], reverse=True)
            for rank, idx in enumerate(typed[: args.top_features_per_type], start=1):
                feature = feature_df.iloc[int(idx)]
                corpus_count = int(feature["corpus_case_count"])
                community_count = int(feature_counts[int(idx)])
                feature_rows.append(
                    {
                        "community_id": int(community_id),
                        "feature_rank": rank,
                        "feature_type": feature_type,
                        "feature_name": feature["feature_name"],
                        "case_count_in_community": community_count,
                        "community_rate": float(community_count / size),
                        "corpus_case_count": corpus_count,
                        "corpus_rate": float(corpus_count / len(selected_meta)),
                        "enrichment": float(
                            (community_count / size)
                            / max(corpus_count / len(selected_meta), 1e-12)
                        ),
                        "idf": float(feature["idf"]),
                    }
                )

    profiles = pd.DataFrame(profile_rows).sort_values(
        ["size", "community_id"],
        ascending=[False, True],
    )
    feature_profiles = pd.DataFrame(feature_rows)
    success = pd.DataFrame(success_rows).sort_values(
        ["accuracy", "mean_confidence", "size"],
        ascending=[True, False, False],
    )

    case_rows = selected_meta.merge(
        profiles[
            [
                "community_id",
                "size",
                "dominant_label",
                "dominant_domain_bucket",
                "accuracy",
                "mean_confidence",
            ]
        ].rename(
            columns={
                "size": "community_size",
                "dominant_label": "community_dominant_label",
                "dominant_domain_bucket": "community_dominant_domain_bucket",
                "accuracy": "community_accuracy",
                "mean_confidence": "community_mean_confidence",
            }
        ),
        on="community_id",
        how="left",
    )
    return case_rows, profiles, feature_profiles, success


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    q_values = np.full_like(p_values, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return q_values
    valid_indices = np.flatnonzero(valid)
    p = p_values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = len(ranked)
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted[i] = running
    q_values[valid_indices[order]] = np.minimum(adjusted, 1.0)
    return q_values


def score_evidence_skew(
    support_path: Path,
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    require_file(support_path, "connected_case_label_distribution.csv")
    support = pd.read_csv(support_path)
    train = predictions[predictions["split"].astype(str) == "train"]
    if train.empty:
        raise ValueError("Predictions CSV contains no train split for base-rate skew scoring.")
    train_labels = train["target_label"].map(normalise_label)
    base_counts = Counter(train_labels)
    positive_label = "1"
    negative_label = "-1"
    base_pos = float(base_counts.get(positive_label, 0))
    base_neg = float(base_counts.get(negative_label, 0))
    if base_pos <= 0 or base_neg <= 0:
        raise ValueError("Both -1 and 1 train labels are required for skew scoring.")
    base_total = base_pos + base_neg
    base_pos_rate = base_pos / base_total
    base_neg_rate = base_neg / base_total

    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for _, row in support.iterrows():
        support_n = int(row.get("support_train_n") or 0)
        pos = float(row.get("support_positive_n") or row.get("support_label_1") or 0.0)
        neg = float(row.get("support_negative_n") or row.get("support_label_-1") or 0.0)
        if support_n <= 0:
            support_n = int(pos + neg)
        pos_rate = pos / support_n if support_n else np.nan
        neg_rate = neg / support_n if support_n else np.nan
        expected = np.asarray([support_n * base_neg_rate, support_n * base_pos_rate], dtype=float)
        observed = np.asarray([neg, pos], dtype=float)
        if support_n > 0 and np.all(expected > 0):
            stat, p_value = power_divergence(
                f_obs=observed,
                f_exp=expected,
                lambda_="log-likelihood",
            )
            stat = float(stat)
            p_value = float(p_value)
        else:
            stat = np.nan
            p_value = np.nan
        effect = float(pos_rate - base_pos_rate) if math.isfinite(pos_rate) else np.nan
        log_odds = float(
            math.log(((pos + 0.5) / (neg + 0.5)) / ((base_pos + 0.5) / (base_neg + 0.5)))
        )
        out = row.to_dict()
        out.update(
            {
                "train_base_positive_rate": base_pos_rate,
                "train_base_negative_rate": base_neg_rate,
                "g_test_statistic": stat,
                "g_test_p_value": p_value,
                "positive_rate_minus_base": effect,
                "negative_rate_minus_base": float(neg_rate - base_neg_rate)
                if math.isfinite(neg_rate)
                else np.nan,
                "log_odds_vs_base": log_odds,
                "skew_direction": "positive_label_1" if effect > 0 else "negative_label_-1",
            }
        )
        rows.append(out)
        p_values.append(p_value)

    result = pd.DataFrame(rows)
    result["g_test_q_value_bh"] = benjamini_hochberg(np.asarray(p_values, dtype=float))
    result["skew_class"] = "ambiguous"
    enough = result["support_train_n"].fillna(0).astype(float) >= args.skew_min_support
    significant = (
        enough
        & (result["g_test_q_value_bh"].astype(float) <= args.skew_alpha)
        & (result["positive_rate_minus_base"].abs() >= args.skew_effect_min)
    )
    neutral = enough & ~significant
    result.loc[significant, "skew_class"] = "label_discriminative"
    result.loc[neutral, "skew_class"] = "neutral"
    result = result.sort_values(
        ["skew_class", "g_test_q_value_bh", "support_train_n"],
        ascending=[True, True, False],
    )
    return result


def write_skew_augmented_explanations(
    explanation_dir: Path,
    output_dir: Path,
    skew: pd.DataFrame,
) -> None:
    top_path = explanation_dir / "case_top_explanations.csv"
    if not top_path.exists():
        return
    top = pd.read_csv(top_path)
    if "evidence_global_index" not in top.columns:
        return
    left = top.copy()
    right = skew.copy()
    left["evidence_global_index_key"] = pd.to_numeric(
        left["evidence_global_index"],
        errors="coerce",
    ).astype("Int64")
    right["evidence_global_index_key"] = pd.to_numeric(
        right["evidence_global_index"],
        errors="coerce",
    ).astype("Int64")
    keep_cols = [
        "evidence_type",
        "evidence_global_index_key",
        "g_test_p_value",
        "g_test_q_value_bh",
        "positive_rate_minus_base",
        "log_odds_vs_base",
        "skew_direction",
        "skew_class",
    ]
    merged = left.merge(
        right[keep_cols],
        on=["evidence_type", "evidence_global_index_key"],
        how="left",
    ).drop(columns=["evidence_global_index_key"])
    merged.to_csv(output_dir / "case_top_explanations_with_skew.csv", index=False)


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = infer_artifact_paths(
        args.explanation_dir,
        graph_cache=args.graph_cache,
        predictions_csv=args.predictions_csv,
    )
    data, metadata = load_graph(paths["graph_cache"])
    predictions = load_predictions(paths["predictions_csv"], expected_cases=int(data["case"].num_nodes))
    case_meta = build_case_metadata(data, predictions)
    selected_case_indices = select_case_indices(case_meta, args.split, args.case_limit)
    feature_types = {part.strip() for part in args.feature_types.split(",") if part.strip()}
    missing = feature_types - STRUCTURAL_FEATURE_TYPES
    if missing:
        raise ValueError(f"Unsupported structural feature types: {sorted(missing)}")

    print(f"[features] cases={len(selected_case_indices)} feature_types={sorted(feature_types)}", flush=True)
    feature_matrix, feature_df = build_case_feature_matrix(data, selected_case_indices, feature_types)
    feature_artifacts = save_case_feature_artifacts(
        args.output_dir,
        feature_matrix,
        feature_df,
        selected_case_indices,
    )
    print(f"[projection] matrix={feature_matrix.shape} nnz={feature_matrix.nnz}", flush=True)
    projection, kept_features = build_projection(feature_matrix, feature_df, args)
    sp.save_npz(args.output_dir / "case_case_projection.npz", projection)
    kept_features.to_csv(args.output_dir / "projection_feature_metadata.csv", index=False)
    print(f"[leiden] nodes={projection.shape[0]} undirected_edges={projection.nnz // 2}", flush=True)
    membership, pagerank = run_leiden(projection, args)
    print(f"[communities] n={len(set(membership.tolist()))}", flush=True)

    case_rows, profiles, feature_profiles, success = summarize_communities(
        case_meta,
        selected_case_indices,
        membership,
        pagerank,
        feature_matrix,
        feature_df,
        args,
    )
    case_rows.to_csv(args.output_dir / "case_communities.csv", index=False)
    profiles.to_csv(args.output_dir / "community_profiles.csv", index=False)
    feature_profiles.to_csv(args.output_dir / "community_feature_profiles.csv", index=False)
    success.to_csv(args.output_dir / "community_success_failure.csv", index=False)

    support_path = args.explanation_dir / "connected_case_label_distribution.csv"
    skew = score_evidence_skew(support_path, predictions, args)
    skew.to_csv(args.output_dir / "evidence_label_skew.csv", index=False)
    write_skew_augmented_explanations(args.explanation_dir, args.output_dir, skew)

    manifest = {
        **feature_artifacts,
        "case_case_projection": str(args.output_dir / "case_case_projection.npz"),
        "projection_feature_metadata": str(args.output_dir / "projection_feature_metadata.csv"),
        "case_communities": str(args.output_dir / "case_communities.csv"),
        "community_profiles": str(args.output_dir / "community_profiles.csv"),
        "community_feature_profiles": str(args.output_dir / "community_feature_profiles.csv"),
        "community_success_failure": str(args.output_dir / "community_success_failure.csv"),
        "evidence_label_skew": str(args.output_dir / "evidence_label_skew.csv"),
        "case_top_explanations_with_skew": str(
            args.output_dir / "case_top_explanations_with_skew.csv"
        ),
        "graph_cache": str(paths["graph_cache"]),
        "predictions_csv": str(paths["predictions_csv"]),
        "label_names": list(metadata.get("label_names", [])),
        "split": args.split,
        "case_limit": args.case_limit,
        "elapsed_seconds": time.time() - start,
    }
    dump_json(args.output_dir / "pattern_why_manifest.json", manifest)
    print(f"[done] out={args.output_dir} elapsed_min={(time.time() - start) / 60:.2f}", flush=True)


if __name__ == "__main__":
    main()
