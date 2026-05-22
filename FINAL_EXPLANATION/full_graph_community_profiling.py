#!/usr/bin/env python
"""Profile communities discovered in the full heterogeneous graph.

Reads the long-form membership table from `full_graph_community_detection.py`,
selects a single resolution, then summarises each community along three axes:

* composition — node-type counts, label balance, domain-bucket distribution,
  representative cases.
* dominant authorities — top statutes, provisions, precedents inside the
  community by case-count.
* prediction behaviour — overall and test-only accuracy, confidence, and the
  high/low-confidence × correct/wrong four-bucket breakdown that flags whether
  errors cluster in specific communities.

Also produces a per-case "neighborhood mixedness" table (community entropy of
the case's authority neighborhood) and flags boundary cases — cases whose
authority neighbours span multiple communities, an explanation for why some
predictions are inherently uncertain.
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pattern_explainability_common import (
    DEFAULT_EXPLANATION_DIR,
    DEFAULT_PATTERN_OUTPUT_DIR,
    LEGAL_AUTHORITY_TYPES,
    SECTION_NODE_TYPES,
    build_case_metadata,
    display_node_id,
    dump_json,
    infer_artifact_paths,
    load_graph,
    load_predictions,
    require_file,
)


DEFAULT_FULL_GRAPH_DIR = DEFAULT_PATTERN_OUTPUT_DIR.parent / "pattern_why_full_graph"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile full-graph Leiden communities at a chosen resolution: "
            "composition, top authorities, accuracy, and boundary cases."
        )
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--communities-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--top-features-per-type", type=int, default=10)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.80)
    parser.add_argument(
        "--boundary-min-degree",
        type=int,
        default=3,
        help="Minimum authority degree for a case to be eligible as a boundary case.",
    )
    parser.add_argument(
        "--boundary-min-entropy",
        type=float,
        default=0.5,
        help="Normalised neighborhood entropy threshold for boundary flagging.",
    )
    parser.add_argument("--representative-cases", type=int, default=5)
    parser.add_argument(
        "--no-section-bridges",
        dest="include_section_bridges",
        action="store_false",
        help="Disable case→authority routing through section nodes (defaults on).",
    )
    parser.set_defaults(include_section_bridges=True)
    return parser


def load_membership(path: Path, resolution: float) -> pd.DataFrame:
    require_file(path, "full_graph_node_communities_long.csv")
    df = pd.read_csv(path)
    available = sorted(df["resolution"].unique().tolist())
    matching = df[np.isclose(df["resolution"], resolution)]
    if matching.empty:
        raise ValueError(
            f"Resolution {resolution} not found in {path}. Available: {available}."
        )
    return matching.reset_index(drop=True)


def collect_case_authorities(
    data: Any,
    include_section_bridges: bool,
) -> dict[int, list[tuple[str, int]]]:
    case_authorities: dict[int, list[tuple[str, int]]] = defaultdict(list)
    section_to_cases: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if src_type == "case" and dst_type in LEGAL_AUTHORITY_TYPES:
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for c_idx, a_idx in zip(edge_index[0], edge_index[1], strict=False):
                case_authorities[int(c_idx)].append((dst_type, int(a_idx)))
        if include_section_bridges and src_type == "case" and dst_type in SECTION_NODE_TYPES:
            edge_index = data[edge_type].edge_index.cpu().numpy()
            section_map = section_to_cases[dst_type]
            for c_idx, sec_idx in zip(edge_index[0], edge_index[1], strict=False):
                section_map[int(sec_idx)].append(int(c_idx))
    if include_section_bridges:
        for edge_type in data.edge_types:
            src_type, relation, dst_type = edge_type
            if relation.startswith("rev_"):
                continue
            if (
                src_type in SECTION_NODE_TYPES
                and dst_type in LEGAL_AUTHORITY_TYPES
                and relation.startswith("cites_")
            ):
                section_map = section_to_cases.get(src_type, {})
                edge_index = data[edge_type].edge_index.cpu().numpy()
                for sec_idx, a_idx in zip(edge_index[0], edge_index[1], strict=False):
                    cases = section_map.get(int(sec_idx), [])
                    for c_idx in cases:
                        case_authorities[c_idx].append((dst_type, int(a_idx)))
    return case_authorities


def authority_id_lookup(data: Any) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    for ntype in LEGAL_AUTHORITY_TYPES:
        if ntype in data.node_types and hasattr(data[ntype], "node_id"):
            lookup[ntype] = [str(v) for v in data[ntype].node_id]
    return lookup


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = infer_artifact_paths(
        args.explanation_dir,
        graph_cache=args.graph_cache,
        predictions_csv=args.predictions_csv,
    )
    data, _ = load_graph(paths["graph_cache"])
    predictions = load_predictions(paths["predictions_csv"], expected_cases=int(data["case"].num_nodes))
    case_meta = build_case_metadata(data, predictions)
    membership = load_membership(
        args.communities_dir / "full_graph_node_communities_long.csv",
        args.resolution,
    )

    case_to_comm: dict[int, int] = {}
    auth_to_comm: dict[tuple[str, int], int] = {}
    nodes_per_comm: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for _, row in membership.iterrows():
        ntype = str(row["node_type"])
        lidx = int(row["local_index"])
        cid = int(row["community_id"])
        nodes_per_comm[cid][ntype].append(lidx)
        if ntype == "case":
            case_to_comm[lidx] = cid
        else:
            auth_to_comm[(ntype, lidx)] = cid

    print(
        f"[membership] communities={len(nodes_per_comm)} cases_assigned={len(case_to_comm)}",
        flush=True,
    )

    case_authorities = collect_case_authorities(data, args.include_section_bridges)
    auth_id_table = authority_id_lookup(data)

    # Per-case neighborhood mixedness table.
    nbr_rows: list[dict[str, Any]] = []
    for c_idx, auths in case_authorities.items():
        case_comm = case_to_comm.get(c_idx)
        if case_comm is None:
            continue
        comm_counts: Counter[int] = Counter()
        for atype, aidx in auths:
            ac = auth_to_comm.get((atype, aidx))
            if ac is None:
                continue
            comm_counts[ac] += 1
        total = sum(comm_counts.values())
        if total == 0:
            continue
        same = comm_counts.get(case_comm, 0)
        n_touched = len(comm_counts)
        max_share = max(comm_counts.values()) / total
        probs = np.asarray(
            [v / total for v in comm_counts.values()], dtype=float
        )
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        norm_entropy = entropy / np.log(n_touched) if n_touched > 1 else 0.0
        nbr_rows.append(
            {
                "case_index": int(c_idx),
                "case_community": int(case_comm),
                "n_authorities": int(total),
                "n_communities_touched": int(n_touched),
                "same_community_share": float(same / total),
                "max_community_share": float(max_share),
                "neighborhood_entropy": entropy,
                "neighborhood_normalized_entropy": float(norm_entropy),
            }
        )

    case_nbr_df = pd.DataFrame(nbr_rows)
    keep_meta_cols = [
        "case_index",
        "case_id",
        "split",
        "target_label",
        "pred_label",
        "confidence",
        "correct",
        "domain_bucket",
    ]
    case_nbr_df = case_nbr_df.merge(case_meta[keep_meta_cols], on="case_index", how="left")
    boundary_df = case_nbr_df[
        (case_nbr_df["n_authorities"] >= args.boundary_min_degree)
        & (case_nbr_df["neighborhood_normalized_entropy"] >= args.boundary_min_entropy)
    ].sort_values("neighborhood_normalized_entropy", ascending=False)

    profile_rows: list[dict[str, Any]] = []
    authority_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    case_meta_indexed = case_meta.set_index("case_index")

    for comm_id, type_to_indices in sorted(nodes_per_comm.items()):
        case_indices = type_to_indices.get("case", [])
        if not case_indices:
            continue
        present = case_meta_indexed.index.intersection(case_indices)
        if present.empty:
            continue
        meta_subset = case_meta_indexed.loc[present].reset_index()
        size = len(meta_subset)
        labels = meta_subset["target_label"].astype(str)
        preds = meta_subset["pred_label"].astype(str)
        confs = pd.to_numeric(meta_subset["confidence"], errors="coerce")
        correct = labels == preds
        is_test = meta_subset["split"].astype(str) == "test"

        label_share = labels.value_counts(normalize=True).to_dict()
        domain_share = meta_subset["domain_bucket"].value_counts(normalize=True).to_dict()
        dominant_label = labels.value_counts().idxmax() if size else ""
        dominant_domain = meta_subset["domain_bucket"].value_counts().idxmax() if size else ""

        # Top representative cases: highest community-internal authority share.
        if not case_nbr_df.empty:
            comm_nbr = case_nbr_df[case_nbr_df["case_community"] == comm_id]
            top_rep = comm_nbr.sort_values("same_community_share", ascending=False).head(
                args.representative_cases
            )
            rep_indices = "|".join(str(int(v)) for v in top_rep["case_index"])
            rep_cases = " | ".join(str(v) for v in top_rep["case_id"])
        else:
            rep_indices = ""
            rep_cases = ""

        profile = {
            "community_id": int(comm_id),
            "n_cases": int(size),
            "n_test_cases": int(is_test.sum()),
            "n_statutes": int(len(type_to_indices.get("statute", []))),
            "n_provisions": int(len(type_to_indices.get("provision", []))),
            "n_precedents": int(len(type_to_indices.get("precedent", []))),
            "dominant_label": dominant_label,
            "dominant_domain_bucket": dominant_domain,
            "accuracy": float(correct.mean()) if size else np.nan,
            "accuracy_test": float(correct[is_test].mean()) if is_test.any() else np.nan,
            "mean_confidence": float(confs.mean()) if confs.notna().any() else np.nan,
            "median_confidence": float(confs.median()) if confs.notna().any() else np.nan,
            "high_conf_wrong_n": int(
                ((~correct) & (confs >= args.high_confidence_threshold)).sum()
            ),
            "representative_case_indices": rep_indices,
            "representative_cases": rep_cases,
        }
        for label, share in label_share.items():
            profile[f"label_{label}_rate"] = float(share)
        for domain, share in domain_share.items():
            profile[f"domain_{domain}_rate"] = float(share)
        for split, count in meta_subset["split"].value_counts().items():
            profile[f"split_{split}_n"] = int(count)
        profile_rows.append(profile)

        # Top authorities aggregated by case occurrence inside this community.
        auth_count: Counter[tuple[str, int]] = Counter()
        for c_idx in case_indices:
            for auth in case_authorities.get(c_idx, []):
                auth_count[auth] += 1
        for atype in sorted(LEGAL_AUTHORITY_TYPES):
            typed = [(k, v) for k, v in auth_count.items() if k[0] == atype]
            typed.sort(key=lambda kv: kv[1], reverse=True)
            for rank, ((_, aidx), cnt) in enumerate(typed[: args.top_features_per_type], start=1):
                ids = auth_id_table.get(atype, [])
                name = (
                    display_node_id(atype, ids[aidx], max_len=200)
                    if aidx < len(ids)
                    else f"{atype}::{aidx}"
                )
                authority_rows.append(
                    {
                        "community_id": int(comm_id),
                        "feature_type": atype,
                        "feature_rank": rank,
                        "feature_local_index": int(aidx),
                        "feature_name": name,
                        "case_count_in_community": int(cnt),
                        "community_rate": float(cnt / size),
                    }
                )

        # Four-bucket prediction breakdown.
        hi = confs >= args.high_confidence_threshold
        bucket = pd.Series("low_conf_correct", index=meta_subset.index)
        bucket[hi & correct] = "high_conf_correct"
        bucket[hi & ~correct] = "high_conf_wrong"
        bucket[~hi & ~correct] = "low_conf_wrong"
        counts = bucket.value_counts()
        total = int(counts.sum())
        prediction_rows.append(
            {
                "community_id": int(comm_id),
                "n_cases": total,
                "n_test_cases": int(is_test.sum()),
                "accuracy": float(correct.mean()) if size else np.nan,
                "accuracy_test": float(correct[is_test].mean()) if is_test.any() else np.nan,
                "high_conf_correct_n": int(counts.get("high_conf_correct", 0)),
                "high_conf_correct_rate": float(counts.get("high_conf_correct", 0) / max(total, 1)),
                "high_conf_wrong_n": int(counts.get("high_conf_wrong", 0)),
                "high_conf_wrong_rate": float(counts.get("high_conf_wrong", 0) / max(total, 1)),
                "low_conf_correct_n": int(counts.get("low_conf_correct", 0)),
                "low_conf_correct_rate": float(counts.get("low_conf_correct", 0) / max(total, 1)),
                "low_conf_wrong_n": int(counts.get("low_conf_wrong", 0)),
                "low_conf_wrong_rate": float(counts.get("low_conf_wrong", 0) / max(total, 1)),
                "mean_confidence_correct": float(confs[correct].mean())
                if correct.any()
                else np.nan,
                "mean_confidence_wrong": float(confs[~correct].mean())
                if (~correct).any()
                else np.nan,
            }
        )

    profile_df = pd.DataFrame(profile_rows).sort_values("n_cases", ascending=False)
    authority_df = pd.DataFrame(authority_rows)
    prediction_df = pd.DataFrame(prediction_rows).sort_values("accuracy")

    suffix = f"res_{args.resolution:.2f}"
    profile_path = args.output_dir / f"full_graph_community_profiles_{suffix}.csv"
    authority_path = args.output_dir / f"full_graph_community_authorities_{suffix}.csv"
    prediction_path = args.output_dir / f"full_graph_community_predictions_{suffix}.csv"
    nbr_path = args.output_dir / f"full_graph_case_neighborhood_{suffix}.csv"
    boundary_path = args.output_dir / f"full_graph_boundary_cases_{suffix}.csv"

    profile_df.to_csv(profile_path, index=False)
    authority_df.to_csv(authority_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)
    case_nbr_df.to_csv(nbr_path, index=False)
    boundary_df.to_csv(boundary_path, index=False)

    manifest = {
        "resolution": args.resolution,
        "communities_dir": str(args.communities_dir),
        "graph_cache": str(paths["graph_cache"]),
        "predictions_csv": str(paths["predictions_csv"]),
        "high_confidence_threshold": args.high_confidence_threshold,
        "boundary_min_degree": args.boundary_min_degree,
        "boundary_min_entropy": args.boundary_min_entropy,
        "include_section_bridges": bool(args.include_section_bridges),
        "n_communities_with_cases": int(profile_df.shape[0]),
        "n_boundary_cases": int(boundary_df.shape[0]),
        "outputs": {
            "profiles": profile_path.name,
            "authorities": authority_path.name,
            "predictions": prediction_path.name,
            "case_neighborhoods": nbr_path.name,
            "boundary_cases": boundary_path.name,
        },
        "elapsed_seconds": time.time() - start,
    }
    dump_json(
        args.output_dir / f"full_graph_profiling_manifest_{suffix}.json", manifest
    )
    print(
        f"[done] out={args.output_dir} elapsed_min={(time.time() - start) / 60:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
