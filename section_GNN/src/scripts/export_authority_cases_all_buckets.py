#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


SECTION_GNN_ROOT = Path("/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN")
DEFAULT_DATA_ROOT = SECTION_GNN_ROOT / "data/timed_bucket_runs"
DEFAULT_OUTPUT_ROOT = SECTION_GNN_ROOT / "outputs/timed_bucket_runs"
DEFAULT_BUCKETS = (
    "family_matrimonial_timed_mistral",
    "fin_fraud_timed_mistral",
    "land_property_timed_mistral",
    "motor_accidents_timed_mistral",
    "sexual_offences_timed_mistral",
)


NodeRef = tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all cases connected to an authority node across non-cross bucket graphs."
    )
    parser.add_argument("--node-type", default="statute", help="Node type to search.")
    parser.add_argument("--name", required=True, help="Exact canonical node name to search.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--buckets", nargs="+", default=list(DEFAULT_BUCKETS))
    parser.add_argument("--max-hops", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Combined CSV path. Defaults to <output-root>/authority_cases_all_buckets/<name>_cases.csv",
    )
    return parser.parse_args()


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required to load graph caches. Run with:\n"
            "  micromamba run -n thesis_work python "
            "section_GNN/src/scripts/export_authority_cases_all_buckets.py --name 'food adulteration act'"
        ) from exc
    return torch


def slugify(text: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in text.strip()]
    return "_".join("".join(chars).split("_")).strip("_") or "authority_node"


def graph_cache_for_bucket(data_root: Path, bucket: str) -> Path:
    graph_cache_dir = data_root / bucket / "graph_cache"
    preferred = graph_cache_dir / f"case_star_global_graph_{bucket}.reasoning_focused.pt"
    if preferred.exists():
        return preferred

    candidates = sorted(graph_cache_dir.glob("*global_graph*.pt"))
    baseline_like = [
        path
        for path in candidates
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
    if baseline_like:
        return baseline_like[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No baseline/global graph cache found for bucket: {bucket}")


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


def clean_node_name(node_type: str, node_id: Any) -> str:
    text = str(node_id)
    prefix = f"{node_type}::"
    marker = f"::{node_type}::"
    if text.startswith(prefix):
        return text[len(prefix):]
    if marker in text:
        return text.split(marker, 1)[1]
    return text.split("::")[-1]


def find_exact_node(data: Any, node_type: str, name: str) -> tuple[int, str] | None:
    if node_type not in data.node_types:
        return None
    for idx, node_id in enumerate(as_list(getattr(data[node_type], "node_id", []))):
        if clean_node_name(node_type, node_id) == name:
            return idx, str(node_id)
    return None


def build_adjacency(data: Any) -> dict[NodeRef, set[NodeRef]]:
    adjacency: dict[NodeRef, set[NodeRef]] = defaultdict(set)
    for src_type, relation, dst_type in data.edge_types:
        edge_index = data[(src_type, relation, dst_type)].edge_index
        for src_idx, dst_idx in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            src = (src_type, int(src_idx))
            dst = (dst_type, int(dst_idx))
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def case_paths_within_hops(
    start: NodeRef,
    adjacency: dict[NodeRef, set[NodeRef]],
    max_hops: int,
) -> dict[int, list[NodeRef]]:
    seen = {start}
    parent: dict[NodeRef, NodeRef] = {}
    queue: deque[tuple[NodeRef, int]] = deque([(start, 0)])
    case_paths: dict[int, list[NodeRef]] = {}

    while queue:
        node, depth = queue.popleft()
        if node[0] == "case" and node != start:
            path: list[NodeRef] = []
            cur = node
            while True:
                path.append(cur)
                if cur == start:
                    break
                cur = parent[cur]
            case_paths[node[1]] = list(reversed(path))
            continue
        if depth >= max_hops:
            continue
        for neighbor in sorted(adjacency.get(node, ()), key=lambda item: (item[0], item[1])):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            parent[neighbor] = node
            queue.append((neighbor, depth + 1))

    return case_paths


def label_names_for(data: Any, metadata: dict[str, Any]) -> dict[int, str]:
    names = metadata.get("label_names")
    if isinstance(names, list) and names:
        return {idx: str(name) for idx, name in enumerate(names)}

    y_values = as_list(getattr(data["case"], "y", []))
    raw_labels = [str(item) for item in as_list(getattr(data["case"], "raw_label", []))]
    labels: dict[int, str] = {}
    for idx, y_value in enumerate(y_values):
        label_idx = int(y_value)
        labels.setdefault(label_idx, raw_labels[idx] if idx < len(raw_labels) else str(label_idx))
    return labels


def label_for_case(data: Any, metadata: dict[str, Any], case_idx: int) -> str:
    y_values = [int(item) for item in as_list(getattr(data["case"], "y", []))]
    label_names = label_names_for(data, metadata)
    if case_idx < len(y_values):
        return label_names.get(y_values[case_idx], str(y_values[case_idx]))
    return ""


def path_to_text(data: Any, path: list[NodeRef]) -> str:
    parts: list[str] = []
    for node_type, node_idx in path:
        if node_type == "case":
            case_ids = as_list(getattr(data["case"], "case_id", []))
            parts.append(f"case::{case_ids[node_idx]}")
            continue
        node_ids = as_list(getattr(data[node_type], "node_id", []))
        parts.append(f"{node_type}::{clean_node_name(node_type, node_ids[node_idx])}")
    return " -> ".join(parts)


def rows_for_bucket(
    bucket: str,
    graph_cache: Path,
    node_type: str,
    name: str,
    max_hops: int,
) -> list[dict[str, Any]]:
    data, metadata = load_graph(graph_cache)
    match = find_exact_node(data, node_type, name)
    if match is None:
        return []

    node_idx, raw_node_id = match
    adjacency = build_adjacency(data)
    case_paths = case_paths_within_hops((node_type, node_idx), adjacency, max_hops=max_hops)

    case_ids = as_list(getattr(data["case"], "case_id", []))
    file_names = as_list(getattr(data["case"], "file_name", []))
    raw_labels = as_list(getattr(data["case"], "raw_label", []))

    rows: list[dict[str, Any]] = []
    for case_idx, path in sorted(case_paths.items(), key=lambda item: str(case_ids[item[0]])):
        rows.append(
            {
                "bucket": bucket,
                "authority_node_type": node_type,
                "authority_name": name,
                "authority_node_id": raw_node_id,
                "case_id": str(case_ids[case_idx]),
                "file_name": str(file_names[case_idx]) if case_idx < len(file_names) else "",
                "class": label_for_case(data, metadata, case_idx),
                "raw_label": str(raw_labels[case_idx]) if case_idx < len(raw_labels) else "",
                "hops_from_authority": len(path) - 1,
                "path": path_to_text(data, path),
                "graph_cache": str(graph_cache),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket",
        "authority_node_type",
        "authority_name",
        "authority_node_id",
        "case_id",
        "file_name",
        "class",
        "raw_label",
        "hops_from_authority",
        "path",
        "graph_cache",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_path = args.out or (
        args.output_root
        / "authority_cases_all_buckets"
        / f"{slugify(args.name)}_{args.node_type}_cases.csv"
    )

    all_rows: list[dict[str, Any]] = []
    for bucket in args.buckets:
        graph_cache = graph_cache_for_bucket(args.data_root, bucket)
        rows = rows_for_bucket(
            bucket=bucket,
            graph_cache=graph_cache,
            node_type=str(args.node_type),
            name=str(args.name),
            max_hops=int(args.max_hops),
        )
        all_rows.extend(rows)
        class_counts = dict(Counter(row["class"] for row in rows))
        print(f"{bucket}: {len(rows)} cases {class_counts}")

    write_csv(out_path, all_rows)
    print(f"Wrote CSV: {out_path}")
    print(f"Total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
