#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/"
    "outputs/timed_bucket_runs/fin_fraud_timed_mistral"
)
DEFAULT_NODE_TYPES = ("preamble", "statute", "provision", "precedent")


NodeRef = tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise how strongly selected node types connect to cases of each "
            "class in a section_GNN HeteroData graph."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Output run directory. If --graph-cache is omitted, the matching "
            "data/.../graph_cache directory is inferred from this path."
        ),
    )
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=None,
        help="Explicit .pt graph cache to load.",
    )
    parser.add_argument(
        "--graph-name-contains",
        default="global_graph",
        help="When inferring a graph cache, prefer .pt files containing this string.",
    )
    parser.add_argument(
        "--node-types",
        nargs="+",
        default=list(DEFAULT_NODE_TYPES),
        help="Node types to analyse.",
    )
    parser.add_argument(
        "--degree-thresholds",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Direct-neighbour degree thresholds to report.",
    )
    parser.add_argument(
        "--degree-op",
        choices=("gt", "ge"),
        default="gt",
        help="Use 'gt' for degree > threshold or 'ge' for degree >= threshold.",
    )
    parser.add_argument(
        "--case-hop-depth",
        type=int,
        default=3,
        help=(
            "Max undirected hops from a node to nearby case nodes. Use 2 for "
            "authority -> arguments -> case, 3 to also catch statute <- provision "
            "-> arguments -> case links."
        ),
    )
    parser.add_argument(
        "--shared-min-cases",
        type=int,
        default=2,
        help="Minimum distinct connected cases for a node to count as cross-case/shared.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top connected nodes to print for each class/node type.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV path for the summary rows.",
    )
    parser.add_argument(
        "--include-local",
        action="store_true",
        help=(
            "Include local case-scoped nodes too. By default, summary rows are "
            "restricted to nodes connected to at least --shared-min-cases cases."
        ),
    )
    return parser.parse_args()


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required to load the graph cache. Run this with the thesis "
            "environment, for example:\n"
            "  micromamba run -n thesis_work python "
            "section_GNN/src/scripts/analyze_shared_node_connectivity.py"
        ) from exc
    return torch


def infer_graph_cache(run_dir: Path, graph_name_contains: str) -> Path:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    parts = list(run_dir.resolve().parts)
    try:
        outputs_idx = parts.index("outputs")
    except ValueError as exc:
        raise ValueError(
            "Could not infer graph cache because run-dir does not contain an "
            "'outputs' path component. Pass --graph-cache explicitly."
        ) from exc

    parts[outputs_idx] = "data"
    graph_cache_dir = Path(*parts) / "graph_cache"
    if not graph_cache_dir.exists():
        raise FileNotFoundError(f"Inferred graph_cache directory does not exist: {graph_cache_dir}")

    candidates = sorted(graph_cache_dir.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt graph caches found in {graph_cache_dir}")

    preferred = [path for path in candidates if graph_name_contains in path.name]
    if preferred:
        baseline_like = [
            path
            for path in preferred
            if not any(token in path.name for token in ("party_args", "section_sep", "no_names", "no_cross", "text_only", "case_node_minimised", "hierarchical"))
        ]
        return (baseline_like or preferred)[0]
    return candidates[0]


def load_graph(path: Path) -> tuple[Any, dict[str, Any]]:
    torch = require_torch()
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        return blob["data"], dict(blob.get("metadata", {}) or {})
    return blob, {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def label_names_for(data: Any, metadata: dict[str, Any]) -> dict[int, str]:
    names = metadata.get("label_names")
    if isinstance(names, list) and names:
        return {idx: str(name) for idx, name in enumerate(names)}

    y_values = as_list(getattr(data["case"], "y", []))
    raw_labels = [str(item) for item in as_list(getattr(data["case"], "raw_label", []))]
    mapping: dict[int, str] = {}
    for idx, y_value in enumerate(y_values):
        label_idx = int(y_value)
        if label_idx not in mapping:
            mapping[label_idx] = raw_labels[idx] if idx < len(raw_labels) else str(label_idx)
    return mapping


def build_adjacency(data: Any) -> dict[NodeRef, set[NodeRef]]:
    adjacency: dict[NodeRef, set[NodeRef]] = defaultdict(set)
    for src_type, relation, dst_type in data.edge_types:
        edge_index = data[(src_type, relation, dst_type)].edge_index
        src_indices = edge_index[0].tolist()
        dst_indices = edge_index[1].tolist()
        for src_idx, dst_idx in zip(src_indices, dst_indices):
            src = (src_type, int(src_idx))
            dst = (dst_type, int(dst_idx))
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def case_sets_within_hops(
    start: NodeRef,
    adjacency: dict[NodeRef, set[NodeRef]],
    max_hops: int,
) -> set[int]:
    seen = {start}
    queue: deque[tuple[NodeRef, int]] = deque([(start, 0)])
    cases: set[int] = set()
    while queue:
        node, depth = queue.popleft()
        if node[0] == "case" and node != start:
            cases.add(node[1])
        if depth >= max_hops:
            continue
        for neighbor in adjacency.get(node, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, depth + 1))
    return cases


def node_display_name(data: Any, node_type: str, node_idx: int) -> str:
    node_ids = as_list(getattr(data[node_type], "node_id", []))
    node_id = str(node_ids[node_idx]) if node_idx < len(node_ids) else f"{node_type}:{node_idx}"
    shared_prefix = f"{node_type}::"
    local_marker = f"::{node_type}::"
    if node_id.startswith(shared_prefix):
        return node_id[len(shared_prefix):]
    if local_marker in node_id:
        return node_id.split(local_marker, 1)[1]
    if node_id.startswith("case::"):
        return node_id.split("::")[-1]
    return node_id


def qualifies(degree: int, threshold: int, op: str) -> bool:
    if op == "ge":
        return degree >= threshold
    return degree > threshold


def summarise(
    data: Any,
    metadata: dict[str, Any],
    node_types: list[str],
    thresholds: list[int],
    degree_op: str,
    case_hop_depth: int,
    shared_min_cases: int,
    include_local: bool,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], list[dict[str, Any]]]]:
    adjacency = build_adjacency(data)
    label_names = label_names_for(data, metadata)
    case_labels = [int(item) for item in as_list(getattr(data["case"], "y", []))]
    case_idx_to_class = {
        idx: label_names.get(label_idx, str(label_idx))
        for idx, label_idx in enumerate(case_labels)
    }
    classes = sorted(set(case_idx_to_class.values()))

    summary_rows: list[dict[str, Any]] = []
    top_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}

    for node_type in node_types:
        if node_type not in data.node_types:
            print(f"[warn] node type not present in graph: {node_type}", file=sys.stderr)
            continue

        node_records: list[dict[str, Any]] = []
        for node_idx in range(int(data[node_type].num_nodes)):
            ref = (node_type, node_idx)
            connected_cases = case_sets_within_hops(ref, adjacency, case_hop_depth)
            if not connected_cases:
                continue
            global_case_count = len(connected_cases)
            is_shared = global_case_count >= shared_min_cases
            if not include_local and not is_shared:
                continue
            class_case_counts = Counter(
                case_idx_to_class[case_idx]
                for case_idx in connected_cases
                if case_idx in case_idx_to_class
            )
            direct_degree = len(adjacency.get(ref, ()))
            node_records.append(
                {
                    "node_type": node_type,
                    "node_index": node_idx,
                    "node_id": str(as_list(getattr(data[node_type], "node_id", []))[node_idx]),
                    "name": node_display_name(data, node_type, node_idx),
                    "direct_degree": direct_degree,
                    "global_case_count": global_case_count,
                    "is_shared": is_shared,
                    "class_case_counts": class_case_counts,
                }
            )

        for threshold in thresholds:
            threshold_records = [
                row for row in node_records if qualifies(int(row["direct_degree"]), threshold, degree_op)
            ]
            for class_name in classes:
                class_records = [
                    row for row in threshold_records if row["class_case_counts"].get(class_name, 0) > 0
                ]
                total_class_connections = sum(
                    int(row["class_case_counts"].get(class_name, 0)) for row in class_records
                )
                shared_records = [row for row in class_records if row["is_shared"]]
                summary_rows.append(
                    {
                        "class": class_name,
                        "node_type": node_type,
                        "degree_filter": f"degree {'>=' if degree_op == 'ge' else '>'} {threshold}",
                        "nodes": len(class_records),
                        "shared_nodes": len(shared_records),
                        "case_connections": total_class_connections,
                        "mean_class_cases_per_node": (
                            total_class_connections / len(class_records) if class_records else 0.0
                        ),
                    }
                )
                ranked = sorted(
                    class_records,
                    key=lambda row: (
                        int(row["class_case_counts"].get(class_name, 0)),
                        int(row["global_case_count"]),
                        int(row["direct_degree"]),
                    ),
                    reverse=True,
                )[:top_k]
                top_rows[(class_name, node_type, threshold)] = ranked

    return summary_rows, top_rows


def print_summary(
    graph_cache: Path,
    summary_rows: list[dict[str, Any]],
    top_rows: dict[tuple[str, str, int], list[dict[str, Any]]],
    degree_op: str,
    top_k: int,
) -> None:
    print(f"Graph cache: {graph_cache}")
    print()
    print("Summary")
    print("class,node_type,degree_filter,nodes,shared_nodes,case_connections,mean_class_cases_per_node")
    for row in summary_rows:
        print(
            f"{row['class']},{row['node_type']},{row['degree_filter']},"
            f"{row['nodes']},{row['shared_nodes']},{row['case_connections']},"
            f"{row['mean_class_cases_per_node']:.3f}"
        )

    print()
    print(f"Top {top_k} nodes per class/type/threshold")
    for key in sorted(top_rows):
        class_name, node_type, threshold = key
        ranked = top_rows[key]
        op_text = ">=" if degree_op == "ge" else ">"
        print()
        print(f"[class={class_name} node_type={node_type} degree {op_text} {threshold}]")
        if not ranked:
            print("  no nodes")
            continue
        print("  rank,class_case_connections,global_case_count,direct_degree,shared,name")
        for rank, row in enumerate(ranked, start=1):
            print(
                "  "
                f"{rank},{row['class_case_counts'].get(class_name, 0)},"
                f"{row['global_case_count']},{row['direct_degree']},"
                f"{int(bool(row['is_shared']))},{row['name'][:180]}"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class",
        "node_type",
        "degree_filter",
        "nodes",
        "shared_nodes",
        "case_connections",
        "mean_class_cases_per_node",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    graph_cache = args.graph_cache or infer_graph_cache(args.run_dir, args.graph_name_contains)
    data, metadata = load_graph(graph_cache)
    rows, top_rows = summarise(
        data=data,
        metadata=metadata,
        node_types=[str(item) for item in args.node_types],
        thresholds=[int(item) for item in args.degree_thresholds],
        degree_op=str(args.degree_op),
        case_hop_depth=int(args.case_hop_depth),
        shared_min_cases=int(args.shared_min_cases),
        include_local=bool(args.include_local),
        top_k=int(args.top_k),
    )
    print_summary(graph_cache, rows, top_rows, str(args.degree_op), int(args.top_k))
    if args.csv_out:
        write_csv(args.csv_out, rows)
        print()
        print(f"Wrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
