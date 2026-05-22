#!/usr/bin/env python
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from pattern_explainability_common import (
    DEFAULT_PATTERN_OUTPUT_DIR,
    dump_json,
    load_case_feature_artifacts,
    parse_gpu_list,
    require_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find nearest opposite-label training cases in HGT embedding space and compare evidence neighborhoods."
    )
    parser.add_argument("--pattern-dir", type=Path, default=DEFAULT_PATTERN_OUTPUT_DIR)
    parser.add_argument("--embeddings-npz", type=Path, default=None)
    parser.add_argument("--case-feature-matrix", type=Path, default=None)
    parser.add_argument("--case-feature-metadata", type=Path, default=None)
    parser.add_argument("--case-feature-case-index", type=Path, default=None)
    parser.add_argument("--evidence-skew-csv", type=Path, default=None)
    parser.add_argument("--case-communities-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATTERN_OUTPUT_DIR)
    parser.add_argument("--query-split", default="test")
    parser.add_argument("--candidate-split", default="train")
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--gpus", default="all")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-diff-features", type=int, default=8)
    return parser


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    require_file(path, "HGT embeddings npz")
    payload = np.load(path, allow_pickle=True)
    return {key: payload[key] for key in payload.files}


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def worker_nearest(
    gpu_id: int,
    query_rows: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    candidate_rows: np.ndarray,
    batch_size: int,
    queue: mp.Queue,
) -> None:
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    emb = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    candidate_labels = labels[candidate_rows]
    unique_labels = sorted(set(candidate_labels.tolist()))
    candidate_by_label = {
        label: torch.as_tensor(
            candidate_rows[candidate_labels != label],
            dtype=torch.long,
            device=device,
        )
        for label in unique_labels
    }
    out: list[tuple[int, int, float]] = []
    with torch.no_grad():
        for start in range(0, len(query_rows), batch_size):
            batch_rows = query_rows[start : start + batch_size]
            for label in sorted(set(labels[batch_rows].tolist())):
                local_query_rows = batch_rows[labels[batch_rows] == label]
                candidates = candidate_by_label.get(label)
                if candidates is None or int(candidates.numel()) == 0:
                    continue
                q = emb[torch.as_tensor(local_query_rows, dtype=torch.long, device=device)]
                c = emb[candidates]
                scores = q @ c.T
                values, positions = torch.max(scores, dim=1)
                nearest_rows = candidates[positions].detach().cpu().numpy()
                values_np = values.detach().cpu().numpy()
                for query_row, nearest_row, value in zip(
                    local_query_rows.tolist(),
                    nearest_rows.tolist(),
                    values_np.tolist(),
                    strict=False,
                ):
                    out.append((int(query_row), int(nearest_row), float(value)))
    queue.put(out)


def find_nearest_opposite(
    embeddings: np.ndarray,
    labels: np.ndarray,
    query_rows: np.ndarray,
    candidate_rows: np.ndarray,
    gpus: list[int],
    batch_size: int,
) -> list[tuple[int, int, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU nearest-neighbor search requires CUDA.")
    chunks = [chunk for chunk in np.array_split(query_rows, len(gpus)) if len(chunk)]
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = []
    for gpu_id, chunk in zip(gpus, chunks, strict=False):
        process = ctx.Process(
            target=worker_nearest,
            args=(gpu_id, chunk, embeddings, labels, candidate_rows, batch_size, queue),
        )
        process.start()
        processes.append(process)
    results: list[tuple[int, int, float]] = []
    for _ in processes:
        results.extend(queue.get())
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {process.exitcode}.")
    results.sort(key=lambda row: row[0])
    return results


def skew_lookup(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    require_file(path, "evidence label skew CSV")
    skew = pd.read_csv(path)
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in skew.iterrows():
        key = (str(row["evidence_type"]), str(row["evidence_name"]).lower())
        lookup[key] = row.to_dict()
    return lookup


def feature_rank_key(feature: pd.Series, skew: dict[str, Any] | None) -> tuple[float, float]:
    skew_score = 0.0
    if skew and str(skew.get("skew_class")) == "label_discriminative":
        try:
            skew_score = abs(float(skew.get("log_odds_vs_base")))
        except Exception:
            skew_score = 0.0
    return (skew_score, float(feature.get("idf", 0.0)))


def top_difference_features(
    diff: set[int],
    feature_df: pd.DataFrame,
    skew_by_feature: dict[tuple[str, str], dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_idx in diff:
        feature = feature_df.iloc[int(feature_idx)]
        key = (str(feature["feature_type"]), str(feature["feature_name"]).lower())
        skew = skew_by_feature.get(key)
        row = {
            "feature_index": int(feature_idx),
            "feature_type": str(feature["feature_type"]),
            "feature_name": str(feature["feature_name"]),
            "idf": float(feature.get("idf", 0.0)),
            "corpus_case_count": int(feature.get("corpus_case_count", 0)),
            "skew_class": skew.get("skew_class") if skew else "",
            "skew_direction": skew.get("skew_direction") if skew else "",
            "log_odds_vs_base": float(skew.get("log_odds_vs_base")) if skew else np.nan,
            "g_test_q_value_bh": float(skew.get("g_test_q_value_bh")) if skew else np.nan,
            "_rank": feature_rank_key(feature, skew),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["_rank"], reverse=True)
    for row in rows:
        row.pop("_rank", None)
    return rows[:limit]


def compact_feature_text(features: list[dict[str, Any]]) -> str:
    return " | ".join(f"{row['feature_type']}:{row['feature_name']}" for row in features)


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = args.embeddings_npz or (args.pattern_dir / "hgt_case_embeddings.npz")
    matrix_path = args.case_feature_matrix or (args.pattern_dir / "case_feature_matrix.npz")
    feature_path = args.case_feature_metadata or (args.pattern_dir / "case_feature_metadata.csv")
    case_feature_case_index = args.case_feature_case_index or (
        args.pattern_dir / "case_feature_case_index.csv"
    )
    skew_path = args.evidence_skew_csv or (args.pattern_dir / "evidence_label_skew.csv")
    communities_path = args.case_communities_csv or (args.pattern_dir / "case_communities.csv")

    payload = load_embeddings(embeddings_path)
    embeddings = normalize_embeddings(payload["embeddings"])
    case_indices = payload["case_indices"].astype(np.int64)
    splits = payload["split"].astype(str)
    labels = payload["target_label"].astype(str)
    pred_labels = payload["pred_label"].astype(str)
    confidences = payload["confidence"].astype(float)
    case_ids = payload["case_id"].astype(str)
    file_names = payload["file_name"].astype(str)
    matrix, feature_df, feature_case_rows = load_case_feature_artifacts(
        matrix_path,
        feature_path,
        case_feature_case_index,
    )
    feature_df = feature_df.reset_index(drop=True)
    if not np.array_equal(feature_case_rows["case_index"].to_numpy(dtype=np.int64), case_indices):
        raise ValueError(
            "Embeddings and case feature artifacts must contain the same cases in the same order."
        )
    communities = pd.read_csv(require_file(communities_path, "case communities CSV"))
    communities = communities.set_index("case_index")
    skew_by_feature = skew_lookup(skew_path)

    query_rows = np.where(splits == args.query_split)[0]
    candidate_rows = np.where(splits == args.candidate_split)[0]
    if args.case_limit is not None:
        query_rows = query_rows[: args.case_limit]
    if len(query_rows) == 0 or len(candidate_rows) == 0:
        raise ValueError("No query or candidate rows selected.")
    gpu_ids = parse_gpu_list(args.gpus)
    print(
        f"[nearest] queries={len(query_rows)} candidates={len(candidate_rows)} gpus={gpu_ids}",
        flush=True,
    )
    nearest = find_nearest_opposite(
        embeddings,
        labels,
        query_rows.astype(np.int64),
        candidate_rows.astype(np.int64),
        gpu_ids,
        args.batch_size,
    )

    matrix = matrix.tocsr()
    summary_rows: list[dict[str, Any]] = []
    diff_rows: list[dict[str, Any]] = []
    for query_row, opposite_row, similarity in nearest:
        q_features = set(matrix[query_row].indices.tolist())
        o_features = set(matrix[opposite_row].indices.tolist())
        query_only = top_difference_features(
            q_features - o_features,
            feature_df,
            skew_by_feature,
            args.top_diff_features,
        )
        opposite_only = top_difference_features(
            o_features - q_features,
            feature_df,
            skew_by_feature,
            args.top_diff_features,
        )
        shared = q_features & o_features
        q_case_index = int(case_indices[query_row])
        o_case_index = int(case_indices[opposite_row])
        q_comm = communities.loc[q_case_index].to_dict() if q_case_index in communities.index else {}
        o_comm = communities.loc[o_case_index].to_dict() if o_case_index in communities.index else {}
        narrative = (
            f"Case {q_case_index} (label {labels[query_row]}) is closest to opposite-label "
            f"training case {o_case_index} (label {labels[opposite_row]}) with cosine "
            f"{similarity:.4f}. Query-only evidence: {compact_feature_text(query_only)}. "
            f"Opposite-only evidence: {compact_feature_text(opposite_only)}."
        )
        summary_rows.append(
            {
                "case_index": q_case_index,
                "case_id": case_ids[query_row],
                "file_name": file_names[query_row],
                "split": splits[query_row],
                "target_label": labels[query_row],
                "pred_label": pred_labels[query_row],
                "confidence": confidences[query_row],
                "community_id": q_comm.get("community_id", ""),
                "nearest_opposite_case_index": o_case_index,
                "nearest_opposite_case_id": case_ids[opposite_row],
                "nearest_opposite_file_name": file_names[opposite_row],
                "nearest_opposite_split": splits[opposite_row],
                "nearest_opposite_target_label": labels[opposite_row],
                "nearest_opposite_pred_label": pred_labels[opposite_row],
                "nearest_opposite_confidence": confidences[opposite_row],
                "nearest_opposite_community_id": o_comm.get("community_id", ""),
                "cosine_similarity": similarity,
                "shared_feature_count": len(shared),
                "query_only_feature_count": len(q_features - o_features),
                "opposite_only_feature_count": len(o_features - q_features),
                "top_query_only_features": compact_feature_text(query_only),
                "top_opposite_only_features": compact_feature_text(opposite_only),
                "narrative": narrative,
            }
        )
        for side, features in (("query_only", query_only), ("opposite_only", opposite_only)):
            for rank, feature in enumerate(features, start=1):
                row = {
                    "case_index": q_case_index,
                    "nearest_opposite_case_index": o_case_index,
                    "side": side,
                    "rank": rank,
                }
                row.update(feature)
                diff_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(diff_rows)
    summary.to_csv(args.output_dir / "counterfactual_neighborhoods.csv", index=False)
    details.to_csv(args.output_dir / "counterfactual_neighborhood_feature_differences.csv", index=False)
    dump_json(
        args.output_dir / "counterfactual_neighborhood_manifest.json",
        {
            "counterfactual_neighborhoods": str(args.output_dir / "counterfactual_neighborhoods.csv"),
            "counterfactual_neighborhood_feature_differences": str(
                args.output_dir / "counterfactual_neighborhood_feature_differences.csv"
            ),
            "embeddings_npz": str(embeddings_path),
            "query_split": args.query_split,
            "candidate_split": args.candidate_split,
            "case_limit": args.case_limit,
            "gpus": gpu_ids,
            "elapsed_seconds": time.time() - start,
        },
    )
    print(f"[done] rows={len(summary)} out={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
