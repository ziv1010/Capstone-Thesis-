#!/usr/bin/env python
"""Run Leiden community detection on the full heterogeneous legal graph.

Constructs an undirected multi-typed graph from the cached PyG HeteroData bundle
(cases + statutes + provisions + precedents + optionally judges/courts), runs
Leiden at one or more resolution parameters, and saves per-resolution node
memberships. Downstream profiling and bridge/hub scripts consume the long-form
membership table this writes.

Differs from `structural_why_analysis.py` in that Leiden operates on the
heterogeneous graph topology directly rather than a case-case projection.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd

from pattern_explainability_common import (
    DEFAULT_EXPLANATION_DIR,
    DEFAULT_PATTERN_OUTPUT_DIR,
    LEGAL_AUTHORITY_TYPES,
    SECTION_NODE_TYPES,
    build_case_metadata,
    dump_json,
    infer_artifact_paths,
    load_graph,
    load_predictions,
)


DEFAULT_FULL_GRAPH_OUTPUT_DIR = DEFAULT_PATTERN_OUTPUT_DIR.parent / "pattern_why_full_graph"
DEFAULT_RESOLUTIONS = "0.4,0.7,1.0,1.4,2.0"
DEFAULT_NODE_TYPES = "case,statute,provision,precedent"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Leiden on the full heterogeneous legal graph at multiple "
            "resolutions. Produces long- and wide-form community memberships."
        )
    )
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FULL_GRAPH_OUTPUT_DIR)
    parser.add_argument(
        "--node-types",
        default=DEFAULT_NODE_TYPES,
        help="Comma-separated node types to include in the full graph.",
    )
    parser.add_argument(
        "--no-section-bridges",
        dest="include_section_bridges",
        action="store_false",
        help=(
            "Disable case→authority edges routed through section nodes. "
            "Sections (arguments, facts, etc.) are where cites_* edges originate; "
            "without bridges the case nodes would be isolated."
        ),
    )
    parser.set_defaults(include_section_bridges=True)
    parser.add_argument(
        "--resolutions",
        default=DEFAULT_RESOLUTIONS,
        help="Comma-separated Leiden resolution parameters for sweep.",
    )
    parser.add_argument("--leiden-iterations", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--edge-weighting",
        choices=["binary", "log_inverse_degree"],
        default="binary",
        help="binary=1.0; log_inverse_degree downweights edges to hub authorities.",
    )
    parser.add_argument(
        "--max-authority-degree",
        type=int,
        default=0,
        help=(
            "Drop authority nodes with case-degree above this cap. "
            "0 disables. Cases are never dropped."
        ),
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Smoke-test cap on cases (random sample).",
    )
    return parser


def parse_resolutions(spec: str) -> list[float]:
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    values: list[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Bad resolution value: {part!r}") from exc
    if not values:
        raise ValueError("--resolutions must have at least one value")
    return values


def build_full_graph(
    data: Any,
    args: argparse.Namespace,
) -> tuple[ig.Graph, pd.DataFrame]:
    keep_node_types = {p.strip() for p in args.node_types.split(",") if p.strip()}
    valid = set(data.node_types)
    missing = keep_node_types - valid
    if missing:
        raise ValueError(f"Requested node types not in graph: {sorted(missing)}")

    type_offsets: dict[str, int] = {}
    node_records: list[dict[str, Any]] = []
    next_id = 0
    keep_case_set: set[int] | None = None
    if args.case_limit is not None and "case" in keep_node_types:
        n_cases_total = int(data["case"].num_nodes)
        rng = np.random.default_rng(args.seed)
        sample = rng.choice(n_cases_total, size=min(args.case_limit, n_cases_total), replace=False)
        keep_case_set = set(int(v) for v in sample.tolist())

    for ntype in sorted(keep_node_types):
        type_offsets[ntype] = next_id
        n_local = int(data[ntype].num_nodes)
        if hasattr(data[ntype], "node_id"):
            local_ids = [str(v) for v in data[ntype].node_id]
        else:
            local_ids = [f"{ntype}::{i}" for i in range(n_local)]
        for local_idx in range(n_local):
            keep = True
            if ntype == "case" and keep_case_set is not None:
                keep = local_idx in keep_case_set
            node_records.append(
                {
                    "global_id": next_id,
                    "node_type": ntype,
                    "local_index": local_idx,
                    "node_id": local_ids[local_idx],
                    "_keep": keep,
                }
            )
            next_id += 1

    node_df = pd.DataFrame(node_records)

    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []

    def add_edge(src_type: str, src_local: int, dst_type: str, dst_local: int) -> None:
        if src_type not in keep_node_types or dst_type not in keep_node_types:
            return
        if keep_case_set is not None:
            if src_type == "case" and src_local not in keep_case_set:
                return
            if dst_type == "case" and dst_local not in keep_case_set:
                return
        s = type_offsets[src_type] + src_local
        d = type_offsets[dst_type] + dst_local
        if s == d:
            return
        edge_pairs.append((s, d))
        edge_weights.append(1.0)

    section_to_cases: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    if args.include_section_bridges:
        for edge_type in data.edge_types:
            src_type, relation, dst_type = edge_type
            if relation.startswith("rev_"):
                continue
            if src_type == "case" and dst_type in SECTION_NODE_TYPES:
                edge_index = data[edge_type].edge_index.cpu().numpy()
                section_map = section_to_cases[dst_type]
                for c_idx, sec_idx in zip(edge_index[0], edge_index[1], strict=False):
                    if keep_case_set is not None and int(c_idx) not in keep_case_set:
                        continue
                    section_map[int(sec_idx)].append(int(c_idx))

    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        if relation.startswith("rev_"):
            continue
        if src_type in keep_node_types and dst_type in keep_node_types:
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for s_idx, d_idx in zip(edge_index[0], edge_index[1], strict=False):
                add_edge(src_type, int(s_idx), dst_type, int(d_idx))
        if (
            args.include_section_bridges
            and src_type in SECTION_NODE_TYPES
            and dst_type in keep_node_types
            and relation.startswith("cites_")
        ):
            section_map = section_to_cases.get(src_type, {})
            if not section_map:
                continue
            edge_index = data[edge_type].edge_index.cpu().numpy()
            for sec_idx, d_idx in zip(edge_index[0], edge_index[1], strict=False):
                cases = section_map.get(int(sec_idx), [])
                for c_idx in cases:
                    add_edge("case", int(c_idx), dst_type, int(d_idx))

    if not edge_pairs:
        raise ValueError("No edges constructed for full graph community detection.")

    graph = ig.Graph(n=len(node_df), edges=edge_pairs, directed=False)
    graph.es["weight"] = edge_weights

    keep_mask = node_df["_keep"].to_numpy()
    drop_idx = np.where(~keep_mask)[0].tolist()

    if args.max_authority_degree and args.max_authority_degree > 0:
        degrees = np.asarray(graph.degree())
        for idx in np.where(degrees > args.max_authority_degree)[0]:
            ntype = str(node_df.iloc[int(idx)]["node_type"])
            if ntype != "case" and keep_mask[int(idx)]:
                drop_idx.append(int(idx))
                keep_mask[int(idx)] = False

    if drop_idx:
        graph.delete_vertices(drop_idx)
        node_df = node_df.loc[keep_mask].reset_index(drop=True)
        node_df["global_id"] = node_df.index.astype(np.int64)

    graph = graph.simplify(multiple=True, loops=True, combine_edges=dict(weight="sum"))

    if args.edge_weighting == "log_inverse_degree":
        degrees = np.asarray(graph.degree(), dtype=float)
        new_weights = np.empty(graph.ecount(), dtype=float)
        edge_list = graph.get_edgelist()
        for ei, (s, d) in enumerate(edge_list):
            min_deg = min(degrees[s], degrees[d])
            new_weights[ei] = 1.0 / max(np.log1p(min_deg), 1.0)
        graph.es["weight"] = new_weights.tolist()

    node_df = node_df.drop(columns=["_keep"])
    return graph, node_df


def run_leiden(
    graph: ig.Graph,
    resolution: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=args.leiden_iterations,
        seed=args.seed,
    )
    membership = np.asarray(partition.membership, dtype=np.int64)
    quality = float(partition.quality())
    return membership, quality


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
    print(f"[graph] heterogeneous node_types={list(data.node_types)}", flush=True)

    graph, node_df = build_full_graph(data, args)
    print(f"[graph] |V|={graph.vcount()} |E|={graph.ecount()}", flush=True)
    type_counts = node_df["node_type"].value_counts().to_dict()
    print(f"[graph] node_type_counts={type_counts}", flush=True)

    resolutions = parse_resolutions(args.resolutions)
    sweep_rows: list[dict[str, Any]] = []
    membership_wide = node_df.copy()
    long_frames: list[pd.DataFrame] = []

    for resolution in resolutions:
        t0 = time.time()
        membership, quality = run_leiden(graph, resolution, args)
        elapsed = time.time() - t0
        sizes = pd.Series(membership).value_counts()
        n_communities = int(sizes.size)
        sweep_rows.append(
            {
                "resolution": resolution,
                "n_communities": n_communities,
                "quality": quality,
                "max_community_size": int(sizes.max()),
                "median_community_size": float(sizes.median()),
                "min_community_size": int(sizes.min()),
                "elapsed_seconds": elapsed,
            }
        )
        col = f"community_res_{resolution:.2f}"
        membership_wide[col] = membership
        long_frames.append(
            pd.DataFrame(
                {
                    "resolution": resolution,
                    "global_id": node_df["global_id"].to_numpy(),
                    "node_type": node_df["node_type"].to_numpy(),
                    "local_index": node_df["local_index"].to_numpy(),
                    "community_id": membership,
                }
            )
        )
        print(
            f"[leiden] res={resolution} comms={n_communities} quality={quality:.4f} t={elapsed:.1f}s",
            flush=True,
        )

    membership_wide.to_csv(
        args.output_dir / "full_graph_node_communities_wide.csv", index=False
    )
    pd.concat(long_frames, ignore_index=True).to_csv(
        args.output_dir / "full_graph_node_communities_long.csv", index=False
    )
    pd.DataFrame(sweep_rows).to_csv(
        args.output_dir / "resolution_sweep_summary.csv", index=False
    )

    n_test = int((case_meta["split"].astype(str) == "test").sum())
    manifest = {
        "graph_cache": str(paths["graph_cache"]),
        "predictions_csv": str(paths["predictions_csv"]),
        "node_types": [t.strip() for t in args.node_types.split(",") if t.strip()],
        "include_section_bridges": bool(args.include_section_bridges),
        "edge_weighting": args.edge_weighting,
        "max_authority_degree": int(args.max_authority_degree),
        "case_limit": args.case_limit,
        "n_nodes": int(graph.vcount()),
        "n_edges": int(graph.ecount()),
        "node_type_counts": {k: int(v) for k, v in type_counts.items()},
        "resolutions": resolutions,
        "n_test_cases": n_test,
        "label_names": list(metadata.get("label_names", [])),
        "outputs": {
            "node_communities_wide": "full_graph_node_communities_wide.csv",
            "node_communities_long": "full_graph_node_communities_long.csv",
            "resolution_sweep_summary": "resolution_sweep_summary.csv",
        },
        "elapsed_seconds": time.time() - start,
    }
    dump_json(args.output_dir / "full_graph_manifest.json", manifest)
    print(
        f"[done] out={args.output_dir} elapsed_min={(time.time() - start) / 60:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
