#!/usr/bin/env python
"""Analyze hierarchical relationships between Leiden community partitions.

For every pair of resolutions in the sweep produced by
`full_graph_community_detection.py`, computes the dominant parent of each
finer-resolution community in the coarser partition. Outputs lineage tables
that let you read the partition like a tree: broad legal domains at low
resolution split into specific themes (e.g. tax → income tax / GST) at
higher resolution.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pattern_explainability_common import (
    DEFAULT_PATTERN_OUTPUT_DIR,
    dump_json,
    require_file,
)


DEFAULT_FULL_GRAPH_DIR = DEFAULT_PATTERN_OUTPUT_DIR.parent / "pattern_why_full_graph"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute parent-child community lineage across Leiden resolutions "
            "from a long-form membership table."
        )
    )
    parser.add_argument("--communities-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument(
        "--restrict-node-type",
        default="case",
        help="Restrict alignment statistics to this node type (default: case).",
    )
    parser.add_argument(
        "--min-overlap-share",
        type=float,
        default=0.10,
        help="Drop child→parent links covering less than this share of the child's nodes.",
    )
    return parser


def majority_parent_table(
    long_df: pd.DataFrame,
    coarse_res: float,
    fine_res: float,
    min_overlap_share: float,
) -> pd.DataFrame:
    coarse = long_df[long_df["resolution"] == coarse_res][
        ["global_id", "community_id"]
    ].rename(columns={"community_id": "parent_community"})
    fine = long_df[long_df["resolution"] == fine_res][
        ["global_id", "community_id"]
    ].rename(columns={"community_id": "child_community"})
    merged = fine.merge(coarse, on="global_id", how="inner")
    if merged.empty:
        return pd.DataFrame()
    counts = (
        merged.groupby(["child_community", "parent_community"])
        .size()
        .reset_index(name="overlap_n")
    )
    child_totals = (
        merged.groupby("child_community").size().reset_index(name="child_size")
    )
    counts = counts.merge(child_totals, on="child_community", how="left")
    counts["overlap_share"] = counts["overlap_n"] / counts["child_size"]
    counts = counts[counts["overlap_share"] >= min_overlap_share]
    counts["coarse_resolution"] = coarse_res
    counts["fine_resolution"] = fine_res
    counts = counts.sort_values(
        ["child_community", "overlap_n"], ascending=[True, False]
    )
    return counts


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.communities_dir / "full_graph_node_communities_long.csv"
    require_file(long_path, "full_graph_node_communities_long.csv")
    long_df = pd.read_csv(long_path)
    if args.restrict_node_type:
        long_df = long_df[long_df["node_type"] == args.restrict_node_type].reset_index(
            drop=True
        )
    if long_df.empty:
        raise ValueError(
            f"No nodes of type {args.restrict_node_type!r} in membership table."
        )

    resolutions = sorted(long_df["resolution"].unique().tolist())
    print(f"[hierarchy] resolutions={resolutions}", flush=True)

    lineage_frames: list[pd.DataFrame] = []
    pairwise_rows: list[dict[str, Any]] = []
    for i, coarse in enumerate(resolutions):
        for fine in resolutions[i + 1 :]:
            lineage = majority_parent_table(long_df, coarse, fine, args.min_overlap_share)
            if not lineage.empty:
                lineage_frames.append(lineage)
            joined = (
                long_df[long_df["resolution"] == coarse]
                .rename(columns={"community_id": "coarse_id"})[["global_id", "coarse_id"]]
                .merge(
                    long_df[long_df["resolution"] == fine].rename(
                        columns={"community_id": "fine_id"}
                    )[["global_id", "fine_id"]],
                    on="global_id",
                    how="inner",
                )
            )
            if joined.empty:
                continue
            ari = float(
                adjusted_rand_score(joined["coarse_id"], joined["fine_id"])
            )
            nmi = float(
                normalized_mutual_info_score(joined["coarse_id"], joined["fine_id"])
            )
            pairwise_rows.append(
                {
                    "coarse_resolution": coarse,
                    "fine_resolution": fine,
                    "n_overlap_nodes": int(joined.shape[0]),
                    "n_coarse_communities": int(joined["coarse_id"].nunique()),
                    "n_fine_communities": int(joined["fine_id"].nunique()),
                    "adjusted_rand_index": ari,
                    "normalized_mutual_info": nmi,
                }
            )

    lineage_path = args.output_dir / "community_lineage_pairs.csv"
    if lineage_frames:
        pd.concat(lineage_frames, ignore_index=True).to_csv(lineage_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "child_community",
                "parent_community",
                "overlap_n",
                "child_size",
                "overlap_share",
                "coarse_resolution",
                "fine_resolution",
            ]
        ).to_csv(lineage_path, index=False)

    pairwise_path = args.output_dir / "resolution_pairwise_alignment.csv"
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)

    chains: list[dict[str, Any]] = []
    if len(resolutions) >= 2:
        finest_res = resolutions[-1]
        finest = long_df[long_df["resolution"] == finest_res][
            ["global_id", "community_id"]
        ].rename(columns={"community_id": f"comm_res_{finest_res:.2f}"})
        chain_df = finest
        for res in resolutions[:-1]:
            other = long_df[long_df["resolution"] == res][
                ["global_id", "community_id"]
            ].rename(columns={"community_id": f"comm_res_{res:.2f}"})
            chain_df = chain_df.merge(other, on="global_id", how="left")
        # Aggregate to leaf community → most common ancestor at each coarser resolution
        leaf_col = f"comm_res_{finest_res:.2f}"
        for leaf, group in chain_df.groupby(leaf_col):
            row: dict[str, Any] = {
                "leaf_community": int(leaf),
                "leaf_resolution": finest_res,
                "leaf_size": int(len(group)),
            }
            for res in resolutions[:-1]:
                col = f"comm_res_{res:.2f}"
                top = group[col].value_counts()
                if top.empty:
                    continue
                row[f"parent_res_{res:.2f}"] = int(top.index[0])
                row[f"parent_share_res_{res:.2f}"] = float(top.iloc[0] / len(group))
            chains.append(row)
    chain_path = args.output_dir / "community_lineage_chains.csv"
    pd.DataFrame(chains).to_csv(chain_path, index=False)

    manifest = {
        "communities_dir": str(args.communities_dir),
        "restrict_node_type": args.restrict_node_type,
        "min_overlap_share": args.min_overlap_share,
        "resolutions": resolutions,
        "outputs": {
            "lineage_pairs": "community_lineage_pairs.csv",
            "lineage_chains": "community_lineage_chains.csv",
            "pairwise_alignment": "resolution_pairwise_alignment.csv",
        },
        "elapsed_seconds": time.time() - start,
    }
    dump_json(args.output_dir / "community_hierarchy_manifest.json", manifest)
    print(
        f"[done] out={args.output_dir} elapsed_min={(time.time() - start) / 60:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
