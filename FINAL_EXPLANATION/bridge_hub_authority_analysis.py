#!/usr/bin/env python
"""Classify legal authorities as core / bridge / hub based on community spread.

For each statute / provision / precedent, looks at the set of cases that cite
it and the Leiden community memberships of those cases. An authority is:

* core   — almost all citing cases live in one community (max share ≥ τ_core).
* bridge — concentrated in a small handful of communities (n_communities small,
           but max share below τ_bridge).
* hub    — spread across many communities with no dominant home.
* rare   — too few citing cases to classify confidently.

Outputs per-authority classification, a bridge-pair table (which community
pairs are most often connected by bridge authorities), and per-type role
summaries. Reads memberships produced by `full_graph_community_detection.py`.
"""
from __future__ import annotations

import argparse
import math
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
            "Score statutes, provisions and precedents as core / bridge / hub "
            "based on the community spread of the cases that cite them."
        )
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--communities-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_GRAPH_DIR)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--core-min-share",
        type=float,
        default=0.80,
        help="Top-community share at or above this threshold ⇒ 'core'.",
    )
    parser.add_argument(
        "--bridge-max-share",
        type=float,
        default=0.60,
        help="Below this top share, an authority is bridge or hub.",
    )
    parser.add_argument(
        "--bridge-max-communities",
        type=int,
        default=3,
        help="Authorities touching ≥ this many communities (and not core) become 'hub'.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=5,
        help="Authorities cited by fewer than this many cases are tagged 'rare'.",
    )
    parser.add_argument(
        "--no-section-bridges",
        dest="include_section_bridges",
        action="store_false",
        help="Disable case→authority routing through section nodes (defaults on).",
    )
    parser.set_defaults(include_section_bridges=True)
    return parser


def collect_authority_to_cases(
    data: Any,
    include_section_bridges: bool,
) -> dict[tuple[str, int], list[int]]:
    auth_to_cases: dict[tuple[str, int], list[int]] = defaultdict(list)
    section_to_cases: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if src_type == "case" and dst_type in LEGAL_AUTHORITY_TYPES:
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for c_idx, a_idx in zip(edge_index[0], edge_index[1], strict=False):
                auth_to_cases[(dst_type, int(a_idx))].append(int(c_idx))
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
                    if cases:
                        auth_to_cases[(dst_type, int(a_idx))].extend(cases)
    return auth_to_cases


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

    long_path = args.communities_dir / "full_graph_node_communities_long.csv"
    require_file(long_path, "full_graph_node_communities_long.csv")
    membership = pd.read_csv(long_path)
    membership = membership[np.isclose(membership["resolution"], args.resolution)]
    if membership.empty:
        available = sorted(pd.read_csv(long_path)["resolution"].unique().tolist())
        raise ValueError(
            f"Resolution {args.resolution} not found. Available: {available}."
        )

    case_to_comm: dict[int, int] = {}
    for _, row in membership[membership["node_type"] == "case"].iterrows():
        case_to_comm[int(row["local_index"])] = int(row["community_id"])

    correct_lookup = case_meta.set_index("case_index")["correct"].to_dict()
    confidence_lookup = (
        case_meta.set_index("case_index")["confidence"].astype(float).to_dict()
    )
    test_lookup = (
        case_meta.set_index("case_index")["split"].astype(str) == "test"
    ).to_dict()

    auth_id_table = authority_id_lookup(data)
    auth_to_cases = collect_authority_to_cases(data, args.include_section_bridges)
    print(f"[authorities] candidates={len(auth_to_cases)}", flush=True)

    auth_rows: list[dict[str, Any]] = []
    bridge_pair_counter: Counter[tuple[int, int]] = Counter()

    for (atype, aidx), case_list in auth_to_cases.items():
        comms = [case_to_comm.get(int(c)) for c in case_list]
        comms_clean = [c for c in comms if c is not None]
        n_cases = len(comms_clean)
        if n_cases == 0:
            continue
        comm_counts = Counter(comms_clean)
        n_communities = len(comm_counts)
        max_count = max(comm_counts.values())
        max_share = max_count / n_cases
        home_community = comm_counts.most_common(1)[0][0]
        probs = np.asarray([v / n_cases for v in comm_counts.values()], dtype=float)
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
        max_entropy = math.log(max(n_communities, 1))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        if n_cases < args.min_cases:
            role = "rare"
        elif max_share >= args.core_min_share:
            role = "core"
        elif n_communities >= args.bridge_max_communities and max_share < args.bridge_max_share:
            role = "hub"
        else:
            role = "bridge"

        if role == "bridge":
            top2 = comm_counts.most_common(2)
            if len(top2) == 2:
                pair = tuple(sorted([int(top2[0][0]), int(top2[1][0])]))
                bridge_pair_counter[pair] += 1

        # Citing-case prediction behaviour: how often is the model right when
        # this authority is in the receptive field?
        cited_cases = list({int(c) for c in case_list})
        correct_flags = [bool(correct_lookup.get(c)) for c in cited_cases if c in correct_lookup]
        confs = [
            float(confidence_lookup.get(c))
            for c in cited_cases
            if c in confidence_lookup and not np.isnan(float(confidence_lookup.get(c)))
        ]
        test_flags = [bool(test_lookup.get(c)) for c in cited_cases if c in test_lookup]

        ids = auth_id_table.get(atype, [])
        name = (
            display_node_id(atype, ids[aidx], max_len=200)
            if aidx < len(ids)
            else f"{atype}::{aidx}"
        )

        auth_rows.append(
            {
                "feature_type": atype,
                "feature_local_index": int(aidx),
                "feature_name": name,
                "n_cases": int(n_cases),
                "n_unique_cases": int(len(cited_cases)),
                "n_test_cases": int(sum(test_flags)),
                "n_communities_touched": int(n_communities),
                "home_community": int(home_community),
                "max_community_share": float(max_share),
                "normalized_entropy": float(norm_entropy),
                "role": role,
                "top_communities": "|".join(
                    f"{cid}:{cnt}" for cid, cnt in comm_counts.most_common(3)
                ),
                "model_accuracy_on_citing_cases": float(np.mean(correct_flags))
                if correct_flags
                else np.nan,
                "mean_confidence_on_citing_cases": float(np.mean(confs))
                if confs
                else np.nan,
            }
        )

    auth_df = pd.DataFrame(auth_rows).sort_values(
        ["role", "n_cases"], ascending=[True, False]
    )

    bridge_pairs_df = pd.DataFrame(
        [
            {"community_a": a, "community_b": b, "n_bridge_authorities": cnt}
            for (a, b), cnt in bridge_pair_counter.most_common()
        ]
    )

    role_summary = (
        auth_df.groupby(["feature_type", "role"]).size().reset_index(name="n_authorities")
    )
    role_summary = role_summary.sort_values(["feature_type", "role"])

    suffix = f"res_{args.resolution:.2f}"
    role_path = args.output_dir / f"authority_role_classification_{suffix}.csv"
    pairs_path = args.output_dir / f"bridge_authority_pairs_{suffix}.csv"
    summary_path = args.output_dir / f"authority_role_summary_{suffix}.csv"

    auth_df.to_csv(role_path, index=False)
    bridge_pairs_df.to_csv(pairs_path, index=False)
    role_summary.to_csv(summary_path, index=False)

    manifest = {
        "resolution": args.resolution,
        "communities_dir": str(args.communities_dir),
        "graph_cache": str(paths["graph_cache"]),
        "predictions_csv": str(paths["predictions_csv"]),
        "thresholds": {
            "core_min_share": args.core_min_share,
            "bridge_max_share": args.bridge_max_share,
            "bridge_max_communities": args.bridge_max_communities,
            "min_cases": args.min_cases,
        },
        "include_section_bridges": bool(args.include_section_bridges),
        "n_authorities_classified": int(auth_df.shape[0]),
        "outputs": {
            "authority_role_classification": role_path.name,
            "bridge_authority_pairs": pairs_path.name,
            "authority_role_summary": summary_path.name,
        },
        "elapsed_seconds": time.time() - start,
    }
    dump_json(args.output_dir / f"bridge_hub_manifest_{suffix}.json", manifest)
    print(
        f"[done] out={args.output_dir} elapsed_min={(time.time() - start) / 60:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
