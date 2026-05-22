#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from pattern_explainability_common import DEFAULT_PATTERN_OUTPUT_DIR, dump_json, require_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HDBSCAN on HGT case embeddings and compare embedding clusters to structural communities."
    )
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_OUTPUT_DIR)
    parser.add_argument("--embeddings-npz", type=Path, default=None)
    parser.add_argument("--case-communities-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATTERN_OUTPUT_DIR)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=80)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def normalize(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def label_counts(labels: pd.Series) -> dict[str, int]:
    return {f"label_{label}_n": int(count) for label, count in Counter(labels.astype(str)).items()}


def cluster_profiles(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cluster_id, group in cases.groupby("embedding_cluster_id", sort=True):
        labels = group["target_label"].astype(str)
        preds = group["pred_label"].astype(str)
        confidence = pd.to_numeric(group["confidence"], errors="coerce")
        communities = group["community_id"].astype(str)
        dominant_community = communities.value_counts().idxmax() if len(group) else ""
        dominant_label = labels.value_counts().idxmax() if len(group) else ""
        correct = labels == preds
        row = {
            "embedding_cluster_id": int(cluster_id),
            "is_noise": int(cluster_id) == -1,
            "size": int(len(group)),
            "dominant_label": dominant_label,
            "accuracy": float(correct.mean()) if len(group) else np.nan,
            "mean_confidence": float(confidence.mean()) if confidence.notna().any() else np.nan,
            "dominant_structural_community_id": dominant_community,
            "dominant_structural_community_rate": float(
                communities.value_counts().iloc[0] / len(group)
            )
            if len(group)
            else np.nan,
            "top_structural_communities": " | ".join(
                f"{idx} ({count})" for idx, count in communities.value_counts().head(8).items()
            ),
            "top_domain_buckets": " | ".join(
                f"{idx} ({count})"
                for idx, count in group["domain_bucket"].astype(str).value_counts().head(8).items()
            ),
        }
        row.update(label_counts(labels))
        for label, count in Counter(labels).items():
            row[f"label_{label}_rate"] = float(count / len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["is_noise", "size"], ascending=[True, False])


def community_splits(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for community_id, group in cases.groupby("community_id", sort=True):
        clusters = group["embedding_cluster_id"].astype(str).value_counts()
        labels = group["target_label"].astype(str)
        rows.append(
            {
                "community_id": community_id,
                "size": int(len(group)),
                "n_embedding_clusters": int(group["embedding_cluster_id"].nunique()),
                "noise_rate": float((group["embedding_cluster_id"] == -1).mean()),
                "dominant_embedding_cluster_id": clusters.index[0],
                "dominant_embedding_cluster_rate": float(clusters.iloc[0] / len(group)),
                "top_embedding_clusters": " | ".join(
                    f"{idx} ({count})" for idx, count in clusters.head(10).items()
                ),
                "label_distribution": " | ".join(
                    f"{idx} ({count})" for idx, count in labels.value_counts().items()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_embedding_clusters", "size"],
        ascending=[False, False],
    )


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.embeddings_npz or (args.pattern_dir / "hgt_case_embeddings.npz")
    communities_path = args.case_communities_csv or (args.pattern_dir / "case_communities.csv")
    require_file(embeddings_path, "HGT embeddings npz")
    require_file(communities_path, "case communities CSV")

    payload = np.load(embeddings_path, allow_pickle=True)
    embeddings = normalize(payload["embeddings"])
    if args.case_limit is not None:
        embeddings = embeddings[: args.case_limit]
    n_cases = embeddings.shape[0]
    cases = pd.DataFrame(
        {
            "case_index": payload["case_indices"][:n_cases].astype(np.int64),
            "case_id": payload["case_id"][:n_cases].astype(str),
            "file_name": payload["file_name"][:n_cases].astype(str),
            "split": payload["split"][:n_cases].astype(str),
            "target_label": payload["target_label"][:n_cases].astype(str),
            "pred_label": payload["pred_label"][:n_cases].astype(str),
            "confidence": payload["confidence"][:n_cases].astype(float),
        }
    )
    communities = pd.read_csv(communities_path)
    cases = cases.merge(
        communities[
            [
                "case_index",
                "community_id",
                "domain_bucket",
                "community_size",
                "community_accuracy",
                "community_dominant_label",
            ]
        ],
        on="case_index",
        how="inner",
    )
    if len(cases) != n_cases:
        raise ValueError("Embeddings and case communities do not cover the same selected cases.")

    print(f"[hdbscan] cases={len(cases)} dim={embeddings.shape[1]}", flush=True)
    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        metric="euclidean",
        n_jobs=args.n_jobs,
        copy=True,
    )
    labels = clusterer.fit_predict(embeddings)
    cases["embedding_cluster_id"] = labels.astype(int)
    if hasattr(clusterer, "probabilities_"):
        cases["embedding_cluster_probability"] = clusterer.probabilities_
    else:
        cases["embedding_cluster_probability"] = np.nan
    cases.to_csv(args.output_dir / "case_embedding_clusters.csv", index=False)

    profiles = cluster_profiles(cases)
    profiles.to_csv(args.output_dir / "embedding_cluster_profiles.csv", index=False)
    splits = community_splits(cases)
    splits.to_csv(args.output_dir / "community_embedding_splits.csv", index=False)

    structural = cases["community_id"].astype(str).to_numpy()
    embedding = cases["embedding_cluster_id"].astype(str).to_numpy()
    non_noise = cases["embedding_cluster_id"].to_numpy() != -1
    alignment = {
        "n_cases": int(len(cases)),
        "n_structural_communities": int(cases["community_id"].nunique()),
        "n_embedding_clusters_including_noise": int(cases["embedding_cluster_id"].nunique()),
        "noise_n": int((cases["embedding_cluster_id"] == -1).sum()),
        "noise_rate": float((cases["embedding_cluster_id"] == -1).mean()),
        "adjusted_rand_all": float(adjusted_rand_score(structural, embedding)),
        "normalized_mutual_info_all": float(normalized_mutual_info_score(structural, embedding)),
        "homogeneity_all": float(homogeneity_score(structural, embedding)),
        "completeness_all": float(completeness_score(structural, embedding)),
        "v_measure_all": float(v_measure_score(structural, embedding)),
    }
    if bool(non_noise.any()):
        alignment.update(
            {
                "adjusted_rand_non_noise": float(
                    adjusted_rand_score(structural[non_noise], embedding[non_noise])
                ),
                "normalized_mutual_info_non_noise": float(
                    normalized_mutual_info_score(structural[non_noise], embedding[non_noise])
                ),
                "homogeneity_non_noise": float(
                    homogeneity_score(structural[non_noise], embedding[non_noise])
                ),
                "completeness_non_noise": float(
                    completeness_score(structural[non_noise], embedding[non_noise])
                ),
                "v_measure_non_noise": float(v_measure_score(structural[non_noise], embedding[non_noise])),
            }
        )
    pd.DataFrame([alignment]).to_csv(args.output_dir / "structural_embedding_alignment.csv", index=False)
    dump_json(
        args.output_dir / "embedding_cluster_manifest.json",
        {
            "case_embedding_clusters": str(args.output_dir / "case_embedding_clusters.csv"),
            "embedding_cluster_profiles": str(args.output_dir / "embedding_cluster_profiles.csv"),
            "community_embedding_splits": str(args.output_dir / "community_embedding_splits.csv"),
            "structural_embedding_alignment": str(
                args.output_dir / "structural_embedding_alignment.csv"
            ),
            "embeddings_npz": str(embeddings_path),
            "case_communities_csv": str(communities_path),
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "case_limit": args.case_limit,
            "elapsed_seconds": time.time() - start,
        },
    )
    print(f"[done] clusters={cases['embedding_cluster_id'].nunique()} out={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
