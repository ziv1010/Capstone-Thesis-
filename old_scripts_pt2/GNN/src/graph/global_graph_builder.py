from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.graph.schema import CaseGraphSpec, GraphNode


def merge_case_graphs_into_global_graph(case_graphs: list[CaseGraphSpec]) -> dict[str, Any]:
    node_registry: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    edge_registry: dict[tuple[str, str, str], set[tuple[str, str]]] = defaultdict(set)
    case_summaries: list[dict[str, Any]] = []

    for case_graph in case_graphs:
        case_summaries.append(
            {
                "case_id": case_graph.case_id,
                "file_name": case_graph.file_name,
                "raw_label": case_graph.raw_label,
                "metadata": case_graph.metadata,
            }
        )

        for node in case_graph.nodes:
            registry = node_registry[node.node_type]
            if node.node_key not in registry:
                registry[node.node_key] = {
                    "node_type": node.node_type,
                    "node_key": node.node_key,
                    "text": node.text,
                    "metadata": dict(node.metadata),
                    "share_across_cases": node.share_across_cases,
                    "case_ids": {case_graph.case_id},
                }
                continue

            existing = registry[node.node_key]
            existing["case_ids"].add(case_graph.case_id)
            existing["text"] = existing["text"] or node.text
            existing["share_across_cases"] = existing["share_across_cases"] or node.share_across_cases
            for field, value in node.metadata.items():
                if field == "mention_count":
                    existing["metadata"][field] = int(existing["metadata"].get(field, 0)) + int(value or 0)
                elif field not in existing["metadata"] or existing["metadata"].get(field) in (None, "", 0):
                    existing["metadata"][field] = value

        for edge in case_graph.edges:
            edge_registry[(edge.src_type, edge.relation, edge.dst_type)].add((edge.src_key, edge.dst_key))

    nodes_by_type: dict[str, list[dict[str, Any]]] = {}
    node_key_to_index: dict[str, dict[str, int]] = {}
    node_stats: dict[str, Any] = {}
    for node_type, keyed_nodes in node_registry.items():
        ordered_nodes = sorted(keyed_nodes.values(), key=lambda item: item["node_key"])
        for item in ordered_nodes:
            item["metadata"]["global_case_frequency"] = len(item.pop("case_ids"))
        nodes_by_type[node_type] = ordered_nodes
        node_key_to_index[node_type] = {
            node["node_key"]: idx for idx, node in enumerate(ordered_nodes)
        }
        node_stats[node_type] = len(ordered_nodes)

    edges_by_type: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    relation_stats: dict[str, int] = {}
    node_degree: dict[tuple[str, str], int] = defaultdict(int)
    for edge_type, keyed_edges in edge_registry.items():
        src_type, relation, dst_type = edge_type
        pairs: list[tuple[int, int]] = []
        for src_key, dst_key in sorted(keyed_edges):
            src_idx = node_key_to_index[src_type][src_key]
            dst_idx = node_key_to_index[dst_type][dst_key]
            pairs.append((src_idx, dst_idx))
            node_degree[(src_type, src_key)] += 1
            node_degree[(dst_type, dst_key)] += 1
        edges_by_type[edge_type] = pairs
        relation_stats["|".join(edge_type)] = len(pairs)

    for node_type, nodes in nodes_by_type.items():
        for node in nodes:
            node["metadata"]["degree"] = node_degree[(node_type, node["node_key"])]

    return {
        "nodes_by_type": nodes_by_type,
        "node_key_to_index": node_key_to_index,
        "edges_by_type": edges_by_type,
        "case_summaries": case_summaries,
        "node_stats": node_stats,
        "relation_stats": relation_stats,
    }
