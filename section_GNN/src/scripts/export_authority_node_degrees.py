#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/"
    "outputs/timed_bucket_runs/fin_fraud_timed_mistral"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/"
    "outputs/timed_bucket_runs"
)
DEFAULT_DATA_ROOT = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/"
    "data/timed_bucket_runs"
)
DEFAULT_BUCKETS = (
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
)
DEFAULT_NODE_TYPES = ("statute", "provision", "precedent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export statute/provision/precedent node degrees as CSV files."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Output run directory. Used to infer graph cache and default CSV output directory.",
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
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write CSVs. Defaults to <run-dir>/authority_node_degrees "
            "for one bucket, or <output-root>/authority_node_degrees_all_buckets "
            "for --all-buckets aggregate CSVs."
        ),
    )
    parser.add_argument(
        "--all-buckets",
        action="store_true",
        help="Run all non-cross timed buckets and write per-bucket plus aggregate CSVs.",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=list(DEFAULT_BUCKETS),
        help="Bucket names to use with --all-buckets. Cross bucket is intentionally omitted by default.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root containing timed_bucket_runs graph caches.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root containing timed_bucket_runs output directories.",
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=2,
        help="Minimum degree to export. Default 2 means degree > 1.",
    )
    parser.add_argument(
        "--node-types",
        nargs="+",
        default=list(DEFAULT_NODE_TYPES),
        help="Node types to export.",
    )
    return parser.parse_args()


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required to load the graph cache. Run with:\n"
            "  micromamba run -n thesis_work python "
            "section_GNN/src/scripts/export_authority_node_degrees.py"
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
            if not any(
                token in path.name
                for token in (
                    "party_args",
                    "section_sep",
                    "no_names",
                    "no_cross",
                    "text_only",
                    "case_node_minimised",
                    "hierarchical",
                )
            )
        ]
        return (baseline_like or preferred)[0]
    return candidates[0]


def infer_bucket_graph_cache(data_root: Path, bucket: str, graph_name_contains: str) -> Path:
    graph_cache_dir = data_root / bucket / "graph_cache"
    if not graph_cache_dir.exists():
        raise FileNotFoundError(f"Graph cache directory does not exist for {bucket}: {graph_cache_dir}")

    exact = graph_cache_dir / f"case_star_global_graph_{bucket}.reasoning_focused.pt"
    if exact.exists():
        return exact

    candidates = sorted(graph_cache_dir.glob("*.pt"))
    preferred = [path for path in candidates if graph_name_contains in path.name]
    if preferred:
        baseline_like = [
            path
            for path in preferred
            if not any(
                token in path.name
                for token in (
                    "party_args",
                    "section_sep",
                    "no_names",
                    "no_cross",
                    "text_only",
                    "case_node_minimised",
                    "hierarchical",
                )
            )
        ]
        return (baseline_like or preferred)[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .pt graph caches found in {graph_cache_dir}")


def load_graph(path: Path) -> Any:
    torch = require_torch()
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    return blob["data"] if isinstance(blob, dict) else blob


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def node_name(data: Any, node_type: str, node_idx: int) -> str:
    node_ids = as_list(getattr(data[node_type], "node_id", []))
    node_id = str(node_ids[node_idx]) if node_idx < len(node_ids) else f"{node_type}:{node_idx}"
    shared_prefix = f"{node_type}::"
    local_marker = f"::{node_type}::"
    if node_id.startswith(shared_prefix):
        return node_id[len(shared_prefix):]
    if local_marker in node_id:
        return node_id.split(local_marker, 1)[1]
    return node_id.split("::")[-1]


def compute_direct_degrees(data: Any) -> dict[tuple[str, int], int]:
    degrees: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for src_type, relation, dst_type in data.edge_types:
        edge_index = data[(src_type, relation, dst_type)].edge_index
        for src_idx, dst_idx in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            src = (src_type, int(src_idx))
            dst = (dst_type, int(dst_idx))
            degrees[src].add(dst)
            degrees[dst].add(src)
    return {node: len(neighbors) for node, neighbors in degrees.items()}


def export_node_type(
    data: Any,
    node_type: str,
    degrees: dict[tuple[str, int], int],
    min_degree: int,
    out_dir: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if node_type not in data.node_types:
        out_path = out_dir / f"{node_type}s_degree_gt_1.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["name", "count"])
            writer.writeheader()
        return out_path, rows

    for node_idx in range(int(data[node_type].num_nodes)):
        count = int(degrees.get((node_type, node_idx), 0))
        if count < min_degree:
            continue
        rows.append({"name": node_name(data, node_type, node_idx), "count": count})

    rows.sort(key=lambda row: (-int(row["count"]), str(row["name"])))
    out_path = out_dir / f"{node_type}s_degree_gt_1.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path, rows


def write_aggregate_csv(
    node_type: str,
    aggregate_counts: dict[str, int],
    out_dir: Path,
) -> Path:
    rows = [
        {"name": name, "count": count}
        for name, count in aggregate_counts.items()
    ]
    rows.sort(key=lambda row: (-int(row["count"]), str(row["name"])))
    out_path = out_dir / f"{node_type}s_degree_gt_1_all_buckets.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def write_per_bucket_combined_csv(
    node_type: str,
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    out_path = out_dir / f"{node_type}s_degree_gt_1_by_bucket.csv"
    rows.sort(key=lambda row: (str(row["bucket"]), -int(row["count"]), str(row["name"])))
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["bucket", "name", "count"])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def run_one_graph(
    graph_cache: Path,
    out_dir: Path,
    node_types: list[str],
    min_degree: int,
) -> dict[str, list[dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_graph(graph_cache)
    degrees = compute_direct_degrees(data)

    print(f"Graph cache: {graph_cache}")
    print(f"Output dir : {out_dir}")
    exported: dict[str, list[dict[str, Any]]] = {}
    for node_type in node_types:
        out_path, rows = export_node_type(
            data=data,
            node_type=node_type,
            degrees=degrees,
            min_degree=min_degree,
            out_dir=out_dir,
        )
        exported[node_type] = rows
        print(f"Wrote {node_type}: {out_path} ({len(rows)} rows)")
    return exported


def run_all_buckets(args: argparse.Namespace) -> None:
    node_types = [str(item) for item in args.node_types]
    aggregate: dict[str, dict[str, int]] = {node_type: defaultdict(int) for node_type in node_types}
    by_bucket_rows: dict[str, list[dict[str, Any]]] = {node_type: [] for node_type in node_types}

    for bucket in [str(item) for item in args.buckets]:
        if bucket == "cross_bucket_total_dataset":
            print("Skipping cross_bucket_total_dataset")
            continue
        graph_cache = infer_bucket_graph_cache(args.data_root, bucket, args.graph_name_contains)
        bucket_out_dir = args.output_root / bucket / "authority_node_degrees"
        exported = run_one_graph(
            graph_cache=graph_cache,
            out_dir=bucket_out_dir,
            node_types=node_types,
            min_degree=int(args.min_degree),
        )
        for node_type, rows in exported.items():
            for row in rows:
                name = str(row["name"])
                count = int(row["count"])
                aggregate[node_type][name] += count
                by_bucket_rows[node_type].append(
                    {"bucket": bucket, "name": name, "count": count}
                )
        print()

    aggregate_out_dir = args.out_dir or (args.output_root / "authority_node_degrees_all_buckets")
    aggregate_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Aggregate output dir: {aggregate_out_dir}")
    for node_type in node_types:
        aggregate_path = write_aggregate_csv(node_type, aggregate[node_type], aggregate_out_dir)
        by_bucket_path = write_per_bucket_combined_csv(
            node_type,
            by_bucket_rows[node_type],
            aggregate_out_dir,
        )
        print(f"Wrote aggregate {node_type}: {aggregate_path}")
        print(f"Wrote by-bucket {node_type}: {by_bucket_path}")


def main() -> None:
    args = parse_args()
    if args.all_buckets:
        run_all_buckets(args)
        return

    graph_cache = args.graph_cache or infer_graph_cache(args.run_dir, args.graph_name_contains)
    out_dir = args.out_dir or (args.run_dir / "authority_node_degrees")
    run_one_graph(
        graph_cache=graph_cache,
        out_dir=out_dir,
        node_types=[str(item) for item in args.node_types],
        min_degree=int(args.min_degree),
    )


if __name__ == "__main__":
    main()
