from __future__ import annotations

import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.graph.case_star_builder import build_case_star_graph
from src.graph.schema import CleanedCase
from src.utils.io import ensure_dir

NODE_COLORS = {
    "case": "#1f2937",
    "preamble": "#d97706",
    "facts": "#059669",
    "arguments": "#2563eb",
    "petitioner": "#7c3aed",
    "respondent": "#dc2626",
    "court": "#0f766e",
    "judge": "#6d28d9",
    "lawyer": "#0ea5e9",
    "statute": "#ca8a04",
    "provision": "#65a30d",
    "precedent": "#db2777",
    "org": "#64748b",
    "gpe": "#0891b2",
    "date": "#475569",
    "case_number": "#334155",
    "aggregate": "#94a3b8",
}

DISPLAY_NAMES = {
    "case_number": "Case Number",
}


def _display_name(node_type: str) -> str:
    return DISPLAY_NAMES.get(node_type, node_type.replace("_", " ").title())


def _shorten(text: str, width: int = 20, max_lines: int = 3) -> str:
    normalized = " ".join(str(text).replace("_", " ").split())
    if not normalized:
        return ""
    wrapped = textwrap.wrap(normalized, width=width)
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1][: max(0, width - 3)] + "..."
    return "\n".join(wrapped)


def _human_case_id(case_id: str) -> str:
    return _shorten(case_id.replace("_", " "), width=24, max_lines=4)


def _node_label_from_key(node_type: str, node_key: str) -> str:
    if node_type == "case":
        if node_key.startswith("case::"):
            return _human_case_id(node_key.split("case::", 1)[1])
        return _human_case_id(node_key)
    if node_type in {"preamble", "facts", "arguments"}:
        return _display_name(node_type)
    raw = node_key
    if "::" in raw:
        raw = raw.rsplit("::", 1)[-1]
    return f"{_display_name(node_type)}\n{_shorten(raw, width=18, max_lines=2)}"


def _type_legend_handles(node_types: list[str]) -> list[Line2D]:
    handles: list[Line2D] = []
    for node_type in node_types:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=_display_name(node_type),
                markerfacecolor=NODE_COLORS.get(node_type, "#9ca3af"),
                markersize=10,
            )
        )
    return handles


def _set_axis_clean(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_schema_view(data: Any, output_path: str | Path) -> None:
    graph = nx.DiGraph()
    edge_labels: dict[tuple[str, str], str] = {}

    node_counts = {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}
    for node_type in data.node_types:
        graph.add_node(
            node_type,
            label=f"{_display_name(node_type)}\nN={node_counts[node_type]}",
            color=NODE_COLORS.get(node_type, "#9ca3af"),
        )

    for src_type, relation, dst_type in data.edge_types:
        if relation.startswith("rev_"):
            continue
        edge_count = int(data[(src_type, relation, dst_type)].edge_index.shape[1])
        graph.add_edge(src_type, dst_type)
        edge_labels[(src_type, dst_type)] = f"{relation}\nE={edge_count}"

    positions = {
        "case": (0.0, 0.0),
        "preamble": (-1.6, 1.4),
        "facts": (0.0, 1.7),
        "arguments": (1.6, 1.4),
        "petitioner": (-2.6, 0.4),
        "respondent": (2.6, 0.4),
        "court": (-2.3, -0.8),
        "judge": (0.0, -1.0),
        "lawyer": (2.3, -0.8),
        "statute": (-1.3, -2.0),
        "provision": (0.0, -2.3),
        "precedent": (1.3, -2.0),
        "org": (-2.8, -2.2),
        "gpe": (2.8, -2.2),
        "date": (2.9, 1.9),
        "case_number": (-2.9, 1.9),
    }

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[graph.nodes[node]["color"] for node in graph.nodes],
        node_size=4300,
        alpha=0.95,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: graph.nodes[node]["label"] for node in graph.nodes},
        font_size=10,
        font_color="white",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        edge_color="#475569",
        ax=ax,
        connectionstyle="arc3,rad=0.06",
    )
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=8, ax=ax, rotate=False)
    ax.set_title("Case Star + Global Authority Graph Schema", fontsize=16, pad=18)
    _set_axis_clean(ax)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _stacked_positions(anchor: tuple[float, float], count: int, axis: str = "y", spacing: float = 0.34) -> list[tuple[float, float]]:
    if count <= 1:
        return [anchor]
    start = -spacing * (count - 1) / 2.0
    coords: list[tuple[float, float]] = []
    for idx in range(count):
        offset = start + idx * spacing
        if axis == "x":
            coords.append((anchor[0] + offset, anchor[1]))
        else:
            coords.append((anchor[0], anchor[1] + offset))
    return coords


def save_local_case_star_view(cleaned_case: CleanedCase, graph_cfg: dict[str, Any], output_path: str | Path) -> dict[str, int]:
    case_graph = build_case_star_graph(cleaned_case, graph_cfg)
    graph = nx.DiGraph()
    edge_labels: dict[tuple[str, str], str] = {}

    type_counts = Counter(node.node_type for node in case_graph.nodes)
    for node in case_graph.nodes:
        graph.add_node(
            node.node_key,
            node_type=node.node_type,
            label=(
                _human_case_id(cleaned_case.case_id)
                if node.node_type == "case"
                else (_display_name(node.node_type) if node.node_type in {"preamble", "facts", "arguments"} else f"{_display_name(node.node_type)}\n{_shorten(node.text, width=18, max_lines=2)}")
            ),
            color=NODE_COLORS.get(node.node_type, "#9ca3af"),
        )
    for edge in case_graph.edges:
        graph.add_edge(edge.src_key, edge.dst_key)
        edge_labels[(edge.src_key, edge.dst_key)] = edge.relation

    grouped_nodes: dict[str, list[str]] = defaultdict(list)
    for node_key, attrs in graph.nodes(data=True):
        grouped_nodes[str(attrs["node_type"])].append(node_key)

    anchor_map = {
        "case": ((0.0, 0.0), "y"),
        "preamble": ((-1.3, 1.6), "x"),
        "facts": ((0.0, 1.9), "x"),
        "arguments": ((1.3, 1.6), "x"),
        "petitioner": ((-2.8, 0.6), "y"),
        "respondent": ((2.8, 0.6), "y"),
        "court": ((-2.8, -0.5), "y"),
        "judge": ((0.0, -0.7), "x"),
        "lawyer": ((2.8, -0.5), "y"),
        "statute": ((-1.8, -2.1), "y"),
        "provision": ((0.0, -2.5), "y"),
        "precedent": ((1.8, -2.1), "y"),
        "org": ((-3.7, -1.8), "y"),
        "gpe": ((3.7, -1.8), "y"),
        "date": ((3.8, 1.9), "y"),
        "case_number": ((-3.8, 1.9), "y"),
    }
    positions: dict[str, tuple[float, float]] = {}
    for node_type, node_keys in grouped_nodes.items():
        anchor, axis = anchor_map.get(node_type, ((4.0, 0.0), "y"))
        for node_key, position in zip(sorted(node_keys), _stacked_positions(anchor, len(node_keys), axis=axis)):
            positions[node_key] = position

    fig, ax = plt.subplots(figsize=(18, 13))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[graph.nodes[node]["color"] for node in graph.nodes],
        node_size=[3600 if graph.nodes[node]["node_type"] == "case" else 2600 for node in graph.nodes],
        alpha=0.96,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: graph.nodes[node]["label"] for node in graph.nodes},
        font_size=8,
        font_color="white",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        arrowstyle="-|>",
        arrowsize=14,
        width=1.2,
        edge_color="#64748b",
        ax=ax,
        connectionstyle="arc3,rad=0.03",
    )
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7, ax=ax, rotate=False)
    ax.legend(
        handles=_type_legend_handles(sorted(type_counts.keys())),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Node Types",
    )
    ax.set_title(f"Local Case Star Graph\n{cleaned_case.file_name}", fontsize=15, pad=18)
    _set_axis_clean(ax)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return dict(type_counts)


def _get_out_neighbors(data: Any, src_type: str, src_idx: int) -> list[tuple[tuple[str, str, str], int]]:
    neighbors: list[tuple[tuple[str, str, str], int]] = []
    for edge_type in data.edge_types:
        if edge_type[0] != src_type:
            continue
        edge_index = data[edge_type].edge_index
        matches = (edge_index[0] == src_idx).nonzero(as_tuple=True)[0].tolist()
        for match_idx in matches:
            neighbors.append((edge_type, int(edge_index[1, match_idx].item())))
    return neighbors


def _select_with_limit(items: list[tuple[str, int, str]], limit: int) -> tuple[list[tuple[str, int, str]], int]:
    selected = items[:limit]
    hidden_count = max(0, len(items) - limit)
    return selected, hidden_count


def _global_node_label(data: Any, node_type: str, idx: int) -> str:
    if node_type == "case":
        return _human_case_id(str(data["case"].case_id[idx]))
    if node_type in {"preamble", "facts", "arguments"}:
        return _display_name(node_type)
    node_key = str(data[node_type].node_id[idx])
    return _node_label_from_key(node_type, node_key)


def _global_node_key(data: Any, node_type: str, idx: int) -> str:
    if node_type == "case":
        return str(data["case"].case_id[idx])
    return str(data[node_type].node_id[idx])


def save_two_layer_receptive_field_view(
    data: Any,
    case_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    case_ids = list(data["case"].case_id)
    if case_id not in case_ids:
        raise KeyError(f"Case {case_id} not found in current graph cache.")
    case_idx = case_ids.index(case_id)

    graph = nx.DiGraph()
    positions: dict[str, tuple[float, float]] = {}
    edge_labels: dict[tuple[str, str], str] = {}

    central_key = f"case::{case_id}"
    graph.add_node(
        central_key,
        node_type="case",
        label=_human_case_id(case_id),
        color=NODE_COLORS["case"],
        layer=0,
    )
    positions[central_key] = (0.0, 0.0)

    direct_limits = {
        "lawyer": 3,
        "date": 2,
        "case_number": 2,
        "court": 2,
        "gpe": 2,
        "preamble": 1,
        "facts": 1,
        "arguments": 1,
        "petitioner": 1,
        "respondent": 1,
        "judge": 1,
        "org": 1,
    }
    direct_neighbors_by_type: dict[str, list[tuple[tuple[str, str, str], int]]] = defaultdict(list)
    for edge_type, dst_idx in _get_out_neighbors(data, "case", case_idx):
        direct_neighbors_by_type[edge_type[2]].append((edge_type, dst_idx))

    layer1_nodes: list[tuple[str, str, int, str]] = []
    hidden_direct: dict[str, int] = {}
    for node_type, neighbors in sorted(direct_neighbors_by_type.items()):
        limit = direct_limits.get(node_type, 2)
        selected, hidden_count = _select_with_limit(
            [(node_type, dst_idx, edge_type[1]) for edge_type, dst_idx in neighbors],
            limit=limit,
        )
        hidden_direct[node_type] = hidden_count
        for _, dst_idx, relation in selected:
            node_key = f"{node_type}::{_global_node_key(data, node_type, dst_idx)}::{dst_idx}"
            layer1_nodes.append((node_key, node_type, dst_idx, relation))

    layer1_positions = {
        "preamble": (-2.2, 2.3),
        "facts": (0.0, 2.6),
        "arguments": (2.2, 2.3),
        "petitioner": (-4.0, 0.8),
        "respondent": (4.0, 0.8),
        "court": (-3.5, -0.8),
        "judge": (0.0, -1.0),
        "lawyer": (3.5, -0.8),
        "date": (3.8, 2.4),
        "case_number": (-3.8, 2.4),
        "gpe": (4.4, -2.0),
        "org": (-4.4, -2.0),
    }
    layer1_buckets: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for node_key, node_type, dst_idx, relation in layer1_nodes:
        layer1_buckets[node_type].append((node_key, dst_idx, relation))

    for node_type, node_entries in layer1_buckets.items():
        anchor = layer1_positions.get(node_type, (5.0, 0.0))
        for (node_key, dst_idx, relation), pos in zip(
            node_entries,
            _stacked_positions(anchor, len(node_entries), axis="y", spacing=0.38),
        ):
            graph.add_node(
                node_key,
                node_type=node_type,
                label=_global_node_label(data, node_type, dst_idx),
                color=NODE_COLORS.get(node_type, "#9ca3af"),
                layer=1,
            )
            positions[node_key] = pos
            graph.add_edge(central_key, node_key)
            edge_labels[(central_key, node_key)] = relation

        hidden = hidden_direct.get(node_type, 0)
        if hidden > 0:
            aggregate_key = f"aggregate::{node_type}"
            graph.add_node(
                aggregate_key,
                node_type="aggregate",
                label=f"{_display_name(node_type)}\n+{hidden} more",
                color=NODE_COLORS["aggregate"],
                layer=1,
            )
            positions[aggregate_key] = (anchor[0], anchor[1] - 0.6 - 0.38 * len(node_entries))
            graph.add_edge(central_key, aggregate_key)
            edge_labels[(central_key, aggregate_key)] = "omitted"

    layer2_summary: dict[str, int] = Counter()

    # Layer 2a: local citation nodes reachable via the arguments node in two hops.
    argument_entries = layer1_buckets.get("arguments", [])
    if argument_entries:
        argument_key, argument_idx, _ = argument_entries[0]
        citation_limits = {"statute": 2, "provision": 3, "precedent": 2}
        citation_neighbors: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
        for edge_type, dst_idx in _get_out_neighbors(data, "arguments", argument_idx):
            if edge_type[2] in {"statute", "provision", "precedent"}:
                citation_neighbors[edge_type[2]].append((edge_type[2], dst_idx, edge_type[1]))
        citation_positions = {"statute": (-1.5, 4.2), "provision": (0.0, 4.5), "precedent": (1.5, 4.2)}
        for node_type, items in citation_neighbors.items():
            selected, hidden_count = _select_with_limit(items, citation_limits.get(node_type, 2))
            layer2_summary[node_type] += len(selected)
            for (_, dst_idx, relation), pos in zip(
                selected,
                _stacked_positions(citation_positions[node_type], len(selected), axis="y", spacing=0.34),
            ):
                node_key = f"{node_type}::{_global_node_key(data, node_type, dst_idx)}::{dst_idx}"
                graph.add_node(
                    node_key,
                    node_type=node_type,
                    label=_global_node_label(data, node_type, dst_idx),
                    color=NODE_COLORS.get(node_type, "#9ca3af"),
                    layer=2,
                )
                positions[node_key] = pos
                graph.add_edge(argument_key, node_key)
                edge_labels[(argument_key, node_key)] = relation
            if hidden_count > 0:
                aggregate_key = f"aggregate::{node_type}::citations"
                graph.add_node(
                    aggregate_key,
                    node_type="aggregate",
                    label=f"{_display_name(node_type)}\n+{hidden_count} more",
                    color=NODE_COLORS["aggregate"],
                    layer=2,
                )
                positions[aggregate_key] = (
                    citation_positions[node_type][0],
                    citation_positions[node_type][1] - 0.55 - 0.34 * len(selected),
                )
                graph.add_edge(argument_key, aggregate_key)
                edge_labels[(argument_key, aggregate_key)] = "omitted"

    # Layer 2b: other cases reachable through shared direct neighbors after two GNN layers.
    shared_bridge_types = {"court", "judge", "lawyer", "org", "gpe", "date", "case_number"}
    bridge_case_limits = 2
    for node_type, node_entries in layer1_buckets.items():
        if node_type not in shared_bridge_types:
            continue
        for node_key, node_idx, _ in node_entries:
            other_case_indices: list[int] = []
            for edge_type, dst_idx in _get_out_neighbors(data, node_type, node_idx):
                if edge_type[2] == "case" and dst_idx != case_idx:
                    other_case_indices.append(dst_idx)
            other_case_indices = sorted(dict.fromkeys(other_case_indices))
            selected_indices = other_case_indices[:bridge_case_limits]
            layer2_summary["other_case"] += len(selected_indices)

            base_x, base_y = positions[node_key]
            for offset, other_case_idx in enumerate(selected_indices):
                other_case_key = f"case::{data['case'].case_id[other_case_idx]}"
                if other_case_key not in graph:
                    graph.add_node(
                        other_case_key,
                        node_type="case",
                        label=_human_case_id(str(data["case"].case_id[other_case_idx])),
                        color="#475569",
                        layer=2,
                    )
                    positions[other_case_key] = (base_x + (offset - 0.5) * 0.9, base_y - 2.0)
                graph.add_edge(node_key, other_case_key)
                edge_labels[(node_key, other_case_key)] = "shared via\n2 layers"

            hidden_cases = max(0, len(other_case_indices) - len(selected_indices))
            if hidden_cases > 0:
                aggregate_key = f"aggregate::{node_type}::{node_idx}::cases"
                graph.add_node(
                    aggregate_key,
                    node_type="aggregate",
                    label=f"Other Cases\n+{hidden_cases} more",
                    color=NODE_COLORS["aggregate"],
                    layer=2,
                )
                positions[aggregate_key] = (base_x, base_y - 2.6)
                graph.add_edge(node_key, aggregate_key)
                edge_labels[(node_key, aggregate_key)] = "shared"

    fig, ax = plt.subplots(figsize=(18, 13))
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=[graph.nodes[node]["color"] for node in graph.nodes],
        node_size=[
            3800 if graph.nodes[node]["layer"] == 0 else 2500 if graph.nodes[node]["layer"] == 1 else 2200
            for node in graph.nodes
        ],
        alpha=0.96,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: graph.nodes[node]["label"] for node in graph.nodes},
        font_size=8,
        font_color="white",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        arrowstyle="-|>",
        arrowsize=14,
        width=1.3,
        edge_color="#64748b",
        ax=ax,
        connectionstyle="arc3,rad=0.04",
    )
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_size=7, ax=ax, rotate=False)
    ax.axhline(1.1, color="#cbd5e1", linestyle="--", linewidth=1.0)
    ax.axhline(3.35, color="#cbd5e1", linestyle="--", linewidth=1.0)
    ax.text(-5.2, -0.15, "Layer 0: Case node", fontsize=10, color="#334155")
    ax.text(-5.2, 1.25, "Layer 1: Direct neighbors", fontsize=10, color="#334155")
    ax.text(-5.2, 3.5, "Layer 2: Receptive field after 2 HGT layers", fontsize=10, color="#334155")
    shown_types = sorted({attrs["node_type"] for _, attrs in graph.nodes(data=True)})
    ax.legend(
        handles=_type_legend_handles(shown_types),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        title="Node Types",
    )
    ax.set_title(
        f"Two-Layer Receptive Field Around Case Node\n{case_id}",
        fontsize=15,
        pad=18,
    )
    _set_axis_clean(ax)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"layer2_summary": dict(layer2_summary), "num_nodes_shown": graph.number_of_nodes()}


def save_model_flow_view(
    model_cfg: dict[str, Any],
    output_path: str | Path,
) -> None:
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    num_layers = int(model_cfg.get("num_layers", 2))
    num_heads = int(model_cfg.get("num_heads", 4))
    architecture = str(model_cfg.get("architecture", "hgt")).upper()

    fig, ax = plt.subplots(figsize=(15, 6))
    _set_axis_clean(ax)

    boxes = [
        ((0.3, 2.2), 2.6, 1.2, "Leakage-Safe Inputs", "preamble + facts + arguments\nentities + counts"),
        ((3.5, 2.2), 2.6, 1.2, "HeteroData Graph", "case star graphs merged\nthrough shared authority nodes"),
        ((6.8, 2.2), 2.8, 1.2, f"{architecture} Layers", f"{num_layers} layers, {num_heads} heads\nhidden dim {hidden_dim}"),
        ((10.3, 2.2), 2.4, 1.2, "Case Embeddings", "final `case` node state"),
        ((13.2, 2.2), 1.9, 1.2, "MLP Head", "binary or multiclass"),
        ((15.8, 2.2), 1.9, 1.2, "Outcome", "logits -> prediction"),
    ]

    for (x, y), w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor="#334155",
            facecolor="#e2e8f0",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.78, title, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x + w / 2, y + 0.35, body, ha="center", va="center", fontsize=9, color="#334155")

    for start_x, end_x in [(2.9, 3.5), (6.1, 6.8), (9.6, 10.3), (12.7, 13.2), (15.1, 15.8)]:
        arrow = FancyArrowPatch((start_x, 2.8), (end_x, 2.8), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#475569")
        ax.add_patch(arrow)

    ax.text(
        8.1,
        1.4,
        "How message passing works:\nLayer 1 = case aggregates direct neighbors.\nLayer 2 = case aggregates neighbors-of-neighbors,\nincluding shared authority context via court/judge/lawyer nodes.",
        fontsize=10,
        color="#1e293b",
        ha="center",
        va="center",
    )
    ax.set_xlim(0, 18.2)
    ax.set_ylim(0.5, 4.5)
    ax.set_title("Current Heterogeneous GNN Flow", fontsize=16, pad=16)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_visualization_summary(
    output_path: str | Path,
    selected_case_id: str,
    node_counts: dict[str, int],
    local_case_counts: dict[str, int],
    receptive_field_stats: dict[str, Any],
    model_cfg: dict[str, Any],
) -> None:
    lines = [
        "# Graph Visualization Summary",
        "",
        f"Selected case: `{selected_case_id}`",
        "",
        "## What each figure shows",
        "",
        "- `graph_schema.png`: the heterogeneous schema used by the PyG graph.",
        "- `case_star_<case>.png`: the local star graph for one case before global merging.",
        "- `receptive_field_2layer_<case>.png`: the nodes that can influence the selected case embedding after 2 HGT layers.",
        "- `model_flow.png`: the end-to-end model flow from cleaned inputs to case-level prediction.",
        "",
        "## Current global graph size",
        "",
    ]
    for node_type, count in sorted(node_counts.items()):
        lines.append(f"- `{node_type}`: {count}")
    lines.extend(
        [
            "",
            "## Selected local case graph",
            "",
        ]
    )
    for node_type, count in sorted(local_case_counts.items()):
        lines.append(f"- `{node_type}`: {count}")
    lines.extend(
        [
            "",
            "## Two-layer receptive field notes",
            "",
            "- With 2 HGT layers, the `case` node directly aggregates its immediate neighbors in layer 1.",
            "- In layer 2, it can absorb information from local citation nodes through `arguments`, and from other cases through shared direct authority/context nodes such as `court`, `judge`, and `lawyer`.",
            f"- Nodes shown in the receptive-field figure: {receptive_field_stats.get('num_nodes_shown', 0)}",
            f"- Layer-2 node summary: {receptive_field_stats.get('layer2_summary', {})}",
            "",
            "## Current model",
            "",
            f"- Architecture: `{model_cfg.get('architecture', 'hgt')}`",
            f"- Hidden dim: `{model_cfg.get('hidden_dim', 128)}`",
            f"- Layers: `{model_cfg.get('num_layers', 2)}`",
            f"- Heads: `{model_cfg.get('num_heads', 4)}`",
        ]
    )
    ensure_dir(Path(output_path).parent)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_schema_simple_view(data: Any, output_path: str | Path) -> None:
    node_counts = {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}
    fig, ax = plt.subplots(figsize=(15, 8))
    _set_axis_clean(ax)

    boxes = [
        ((6.0, 3.0), 3.0, 1.4, "Case Node", f"`case`\nN={node_counts.get('case', 0)}"),
        (
            (0.7, 4.9),
            3.8,
            1.6,
            "Pre-Judgment Text",
            f"preamble ({node_counts.get('preamble', 0)})\nfacts ({node_counts.get('facts', 0)})\narguments ({node_counts.get('arguments', 0)})",
        ),
        (
            (0.7, 1.9),
            3.8,
            2.2,
            "Parties + Bench",
            f"petitioner ({node_counts.get('petitioner', 0)})\nrespondent ({node_counts.get('respondent', 0)})\ncourt ({node_counts.get('court', 0)})\njudge ({node_counts.get('judge', 0)})\nlawyer ({node_counts.get('lawyer', 0)})",
        ),
        (
            (10.0, 4.7),
            3.8,
            1.8,
            "Citations via Arguments",
            f"statute ({node_counts.get('statute', 0)})\nprovision ({node_counts.get('provision', 0)})\nprecedent ({node_counts.get('precedent', 0)})",
        ),
        (
            (10.0, 1.9),
            3.8,
            2.2,
            "Context / Shared Nodes",
            f"org ({node_counts.get('org', 0)})\ngpe ({node_counts.get('gpe', 0)})\ndate ({node_counts.get('date', 0)})\ncase_number ({node_counts.get('case_number', 0)})",
        ),
    ]

    for (x, y), w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#334155",
            facecolor="#e2e8f0",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.15, body, ha="center", va="center", fontsize=10, color="#334155")

    arrows = [
        ((4.5, 5.6), (6.0, 4.0)),
        ((4.5, 2.9), (6.0, 3.4)),
        ((9.0, 4.0), (10.0, 5.3)),
        ((9.0, 3.4), (10.0, 3.0)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20, linewidth=1.4, color="#475569"))

    ax.text(
        7.5,
        0.9,
        "Interpretation:\n1. Every case is a center node.\n2. Local text and entity nodes attach around it.\n3. Some entity/context nodes are shared across cases, which creates the global graph.",
        ha="center",
        va="center",
        fontsize=10,
        color="#1e293b",
    )
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.2)
    ax.set_title("Simplified Graph Schema", fontsize=16, pad=16)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_case_star_aggregated_view(cleaned_case: CleanedCase, graph_cfg: dict[str, Any], output_path: str | Path) -> dict[str, dict[str, Any]]:
    case_graph = build_case_star_graph(cleaned_case, graph_cfg)
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in case_graph.nodes:
        if node.node_type == "case":
            continue
        if node.node_type in {"preamble", "facts", "arguments"}:
            grouped[node.node_type].append(f"{len(node.text)} chars")
        else:
            grouped[node.node_type].append(node.text)

    summary: dict[str, dict[str, Any]] = {}
    for node_type, values in grouped.items():
        summary[node_type] = {
            "count": len(values),
            "examples": values[:3],
        }

    fig, ax = plt.subplots(figsize=(16, 10))
    _set_axis_clean(ax)

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=11, fontweight="bold", color="white")
        ax.text(x + w / 2, y + h / 2 - 0.05, body, ha="center", va="center", fontsize=9, color="white")

    draw_box(
        6.0,
        3.3,
        3.2,
        1.5,
        "Selected Case",
        _human_case_id(cleaned_case.case_id),
        NODE_COLORS["case"],
    )

    placements = {
        "preamble": (1.1, 5.4),
        "facts": (6.0, 5.6),
        "arguments": (10.9, 5.4),
        "petitioner": (1.1, 3.2),
        "respondent": (10.9, 3.2),
        "court": (1.1, 1.0),
        "judge": (6.0, 0.8),
        "lawyer": (10.9, 1.0),
        "statute": (4.3, 7.6),
        "provision": (6.0, 7.9),
        "precedent": (7.7, 7.6),
    }

    ordered_types = ["preamble", "facts", "arguments", "petitioner", "respondent", "court", "judge", "lawyer", "statute", "provision", "precedent"]
    for node_type in ordered_types:
        if node_type not in summary:
            continue
        x, y = placements[node_type]
        count = summary[node_type]["count"]
        examples = summary[node_type]["examples"]
        if node_type in {"preamble", "facts", "arguments"}:
            body = f"{count} node\n{examples[0]}"
        else:
            body = f"{count} nodes\n" + "\n".join(_shorten(example, width=20, max_lines=1) for example in examples[:2])
        draw_box(x, y, 3.0, 1.3, _display_name(node_type), body, NODE_COLORS.get(node_type, "#64748b"))
        if node_type in {"statute", "provision", "precedent"}:
            ax.add_patch(FancyArrowPatch((12.4, 6.0), (x + 1.5, y), arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color="#94a3b8"))
        else:
            ax.add_patch(FancyArrowPatch((7.6, 4.8), (x + 1.5, y + 0.65), arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color="#64748b"))

    ax.text(
        7.6,
        6.9,
        "Local meaning:\nThe case node connects directly to text, parties, bench, and context.\nCitation nodes sit one step farther out through the `arguments` node.",
        ha="center",
        va="center",
        fontsize=10,
        color="#1e293b",
    )
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.5)
    ax.set_title(f"Simplified Local Case Star Graph\n{cleaned_case.file_name}", fontsize=16, pad=16)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def _bridge_stats_for_case(data: Any, case_id: str) -> dict[str, dict[str, int]]:
    case_ids = list(data["case"].case_id)
    case_idx = case_ids.index(case_id)
    direct_nodes_by_type: dict[str, list[int]] = defaultdict(list)
    for edge_type in data.edge_types:
        if edge_type[0] != "case" or edge_type[1].startswith("rev_"):
            continue
        edge_index = data[edge_type].edge_index
        matches = (edge_index[0] == case_idx).nonzero(as_tuple=True)[0].tolist()
        for match_idx in matches:
            direct_nodes_by_type[edge_type[2]].append(int(edge_index[1, match_idx].item()))

    result: dict[str, dict[str, int]] = {}
    for node_type in ["court", "judge", "lawyer", "gpe", "date", "case_number"]:
        unique_cases: set[str] = set()
        local_nodes = direct_nodes_by_type.get(node_type, [])
        for node_idx in local_nodes:
            for edge_type in data.edge_types:
                if edge_type[0] != node_type or edge_type[2] != "case":
                    continue
                edge_index = data[edge_type].edge_index
                matches = (edge_index[0] == node_idx).nonzero(as_tuple=True)[0].tolist()
                for match_idx in matches:
                    other_case_idx = int(edge_index[1, match_idx].item())
                    if other_case_idx != case_idx:
                        unique_cases.add(str(data["case"].case_id[other_case_idx]))
        result[node_type] = {
            "local_nodes": len(local_nodes),
            "other_cases": len(unique_cases),
        }
    return result


def save_shared_bridges_simple_view(data: Any, case_id: str, output_path: str | Path) -> dict[str, dict[str, int]]:
    bridge_stats = _bridge_stats_for_case(data, case_id)
    focus_types = ["court", "judge", "lawyer", "gpe"]

    fig, ax = plt.subplots(figsize=(14, 8))
    _set_axis_clean(ax)

    def draw_box(x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.5,
            edgecolor="#334155",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=11, fontweight="bold", color="white")
        ax.text(x + w / 2, y + h / 2 - 0.02, body, ha="center", va="center", fontsize=10, color="white")

    draw_box(0.8, 2.7, 3.2, 1.8, "Selected Case", _human_case_id(case_id), NODE_COLORS["case"])
    draw_box(10.1, 2.7, 3.0, 1.8, "Other Cases", "Cases reachable through\nshared nodes after 2 layers", "#475569")

    y_positions = [5.3, 4.0, 2.7, 1.4]
    for node_type, y in zip(focus_types, y_positions):
        stats = bridge_stats[node_type]
        body = f"{stats['local_nodes']} local nodes\n{stats['other_cases']} unique other cases"
        draw_box(5.1, y, 3.2, 1.0, _display_name(node_type), body, NODE_COLORS.get(node_type, "#64748b"))
        ax.add_patch(FancyArrowPatch((4.0, 3.6), (5.1, y + 0.5), arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color="#64748b"))
        ax.add_patch(FancyArrowPatch((8.3, y + 0.5), (10.1, 3.6), arrowstyle="-|>", mutation_scale=18, linewidth=1.3, color="#64748b"))

    ax.text(
        7.0,
        0.5,
        "This is the main cross-case pathway in the current 2-layer model:\ncase -> shared node -> other cases",
        ha="center",
        va="center",
        fontsize=10,
        color="#1e293b",
    )
    ax.set_xlim(0, 14)
    ax.set_ylim(0.0, 6.7)
    ax.set_title(f"Simplified Global Sharing View\n{case_id}", fontsize=16, pad=16)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return bridge_stats


def save_message_passing_simple_view(
    case_id: str,
    bridge_stats: dict[str, dict[str, int]],
    output_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Step 1: Build the Case Star", "Step 2: HGT Layer 1", "Step 3: HGT Layer 2"]
    for ax, title in zip(axes, titles):
        _set_axis_clean(ax)
        ax.set_title(title, fontsize=13, pad=10)

    # Panel 1
    axes[0].add_patch(FancyBboxPatch((0.35, 0.38), 0.3, 0.2, boxstyle="round,pad=0.03", facecolor=NODE_COLORS["case"], edgecolor="#334155"))
    axes[0].text(0.5, 0.48, "Case", ha="center", va="center", color="white", fontweight="bold")
    satellites = [
        ("Preamble", NODE_COLORS["preamble"], (0.18, 0.78)),
        ("Facts", NODE_COLORS["facts"], (0.5, 0.84)),
        ("Arguments", NODE_COLORS["arguments"], (0.82, 0.78)),
        ("Court", NODE_COLORS["court"], (0.18, 0.18)),
        ("Judge", NODE_COLORS["judge"], (0.5, 0.12)),
        ("Lawyer", NODE_COLORS["lawyer"], (0.82, 0.18)),
    ]
    for label, color, center in satellites:
        axes[0].add_patch(plt.Circle(center, 0.085, color=color, ec="#334155", lw=1.2))
        axes[0].text(center[0], center[1], label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        axes[0].add_patch(FancyArrowPatch(center, (0.5, 0.48), arrowstyle="-|>", mutation_scale=15, linewidth=1.2, color="#64748b"))
    axes[0].text(0.5, 0.95, "Each case starts as a local star graph.", ha="center", va="center", fontsize=10, color="#1e293b")

    # Panel 2
    axes[1].add_patch(FancyBboxPatch((0.35, 0.38), 0.3, 0.2, boxstyle="round,pad=0.03", facecolor=NODE_COLORS["case"], edgecolor="#334155"))
    axes[1].text(0.5, 0.48, "Case\nembedding", ha="center", va="center", color="white", fontweight="bold")
    axes[1].text(
        0.5,
        0.83,
        "Layer 1 mixes direct neighbors:\ntext nodes, parties, bench,\ncontext nodes.",
        ha="center",
        va="center",
        fontsize=10,
        color="#1e293b",
    )
    axes[1].add_patch(FancyArrowPatch((0.5, 0.74), (0.5, 0.59), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#475569"))

    # Panel 3
    axes[2].add_patch(FancyBboxPatch((0.35, 0.38), 0.3, 0.2, boxstyle="round,pad=0.03", facecolor=NODE_COLORS["case"], edgecolor="#334155"))
    axes[2].text(0.5, 0.48, "Updated case\nembedding", ha="center", va="center", color="white", fontweight="bold")
    bridge_text = "\n".join(
        f"{_display_name(node_type)} -> {stats['other_cases']} other cases"
        for node_type, stats in bridge_stats.items()
        if node_type in {"court", "judge", "lawyer"} and stats["other_cases"] > 0
    )
    axes[2].text(
        0.5,
        0.83,
        "Layer 2 adds neighbors-of-neighbors:\nlocal citation info via arguments\nand shared cross-case context via:\n" + bridge_text,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#1e293b",
    )
    axes[2].add_patch(FancyArrowPatch((0.5, 0.74), (0.5, 0.59), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="#475569"))

    fig.suptitle(f"How Information Moves Through the Current GNN\n{case_id}", fontsize=16, y=1.02)
    fig.tight_layout()
    ensure_dir(Path(output_path).parent)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_visual_guide_readme(
    output_path: str | Path,
    selected_case_id: str,
    aggregated_case_summary: dict[str, dict[str, Any]],
    bridge_stats: dict[str, dict[str, int]],
    model_cfg: dict[str, Any],
) -> None:
    lines = [
        "# Easy Guide to the Current GNN Graph",
        "",
        f"Selected example case: `{selected_case_id}`",
        "",
        "This guide explains the current graph in plain language.",
        "",
        "## Big picture",
        "",
        "Every legal case becomes one central `case` node.",
        "",
        "Around that case node, the pipeline attaches three kinds of information:",
        "",
        "1. pre-judgment text nodes: `preamble`, `facts`, `arguments`",
        "2. local entity nodes: petitioner, respondent, court, judge, lawyer",
        "3. legal/context nodes: statute, provision, precedent, org, gpe, date, case number",
        "",
        "Some nodes are shared across many cases. That is what turns many separate case stars into one global graph.",
        "",
        "## What the simplified images mean",
        "",
        "### 1. `01_schema_simple.png`",
        "",
        "This is the cleanest map of the whole design.",
        "",
        "- center: one `case` node",
        "- left/top: the text summaries available before judgment",
        "- left/bottom: parties and bench information",
        "- right/top: citations connected through `arguments`",
        "- right/bottom: context nodes that can also be shared across cases",
        "",
        "### 2. `02_case_star_simple_<case>.png`",
        "",
        "This is one real case, simplified.",
        "",
        "- it does **not** show every raw node individually",
        "- it groups nodes by type and shows counts/examples",
        "- it is the easiest way to see what one training example looks like before global merging",
        "",
        "### 3. `03_shared_bridges_<case>.png`",
        "",
        "This explains cross-case communication.",
        "",
        "- the selected case sits on the left",
        "- shared node types sit in the middle",
        "- other cases reachable through those shared nodes sit on the right",
        "",
        "This is the main reason a GNN can use global legal structure instead of treating each case in isolation.",
        "",
        "### 4. `04_message_passing_steps.png`",
        "",
        "This explains what the current 2-layer HGT actually does.",
        "",
        "- Layer 1: the case node mixes its direct neighbors",
        "- Layer 2: the case node can now absorb neighbors-of-neighbors",
        "- that includes local citation information through `arguments`",
        "- and cross-case information through shared direct bridges like `court`, `judge`, and `lawyer`",
        "",
        "## Important subtle point",
        "",
        "Not all shared nodes create cross-case influence equally fast.",
        "",
        "- `court`, `judge`, `lawyer`, `gpe`, `date`, `case_number` are directly attached to the `case` node",
        "- `statute`, `provision`, `precedent` are attached through the `arguments` node",
        "",
        "So with only 2 GNN layers:",
        "",
        "- local statute/provision/precedent information can influence the case",
        "- but cross-case sharing through statutes is weaker than sharing through directly attached nodes",
        "",
        "## Current selected case summary",
        "",
    ]
    for node_type, payload in sorted(aggregated_case_summary.items()):
        examples = ", ".join(str(example) for example in payload["examples"][:2])
        lines.append(f"- `{node_type}`: {payload['count']} nodes. Examples: {examples}")
    lines.extend(
        [
            "",
            "## Current shared-bridge strength for the selected case",
            "",
        ]
    )
    for node_type, payload in bridge_stats.items():
        lines.append(
            f"- `{node_type}`: {payload['local_nodes']} local nodes of this type connect the case to {payload['other_cases']} unique other cases"
        )
    lines.extend(
        [
            "",
            "## Current model",
            "",
            f"- architecture: `{model_cfg.get('architecture', 'hgt')}`",
            f"- hidden dim: `{model_cfg.get('hidden_dim', 128)}`",
            f"- layers: `{model_cfg.get('num_layers', 2)}`",
            f"- heads: `{model_cfg.get('num_heads', 4)}`",
            "",
            "## How to read this in practice",
            "",
            "If you want to understand one prediction, start in this order:",
            "",
            "1. open `02_case_star_simple_<case>.png` to see what information exists for that case",
            "2. open `03_shared_bridges_<case>.png` to see where cross-case influence could come from",
            "3. open `04_message_passing_steps.png` to understand what the 2-layer HGT can actually mix",
            "",
            "This is a much better mental model than looking at the raw dense graph directly.",
        ]
    )
    ensure_dir(Path(output_path).parent)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _lighten_hex(color: str, blend_with_white: float = 0.82) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return "#e2e8f0"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    red = int(red + (255 - red) * blend_with_white)
    green = int(green + (255 - green) * blend_with_white)
    blue = int(blue + (255 - blue) * blend_with_white)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _draw_story_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    *,
    header_height: float = 0.34,
    fontsize: float = 10.0,
    body_color: str = "#0f172a",
    title_color: str = "white",
    rounded: float = 0.14,
    linewidth: float = 1.4,
    zorder: int = 2,
) -> None:
    outer = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.03,rounding_size={rounded}",
        linewidth=linewidth,
        edgecolor="#334155",
        facecolor=_lighten_hex(color, 0.87),
        zorder=zorder,
    )
    ax.add_patch(outer)
    header = FancyBboxPatch(
        (x, y + h - header_height),
        w,
        header_height,
        boxstyle=f"round,pad=0.03,rounding_size={rounded}",
        linewidth=0,
        facecolor=color,
        zorder=zorder + 0.1,
    )
    ax.add_patch(header)
    ax.text(
        x + 0.18,
        y + h - header_height / 2,
        title,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=title_color,
        zorder=zorder + 0.2,
    )
    ax.text(
        x + 0.18,
        y + h - header_height - 0.12,
        body,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=body_color,
        zorder=zorder + 0.2,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#475569",
    linewidth: float = 1.5,
    mutation_scale: float = 18.0,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def _summarize_case_graph(cleaned_case: CleanedCase, graph_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    case_graph = build_case_star_graph(cleaned_case, graph_cfg)
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in case_graph.nodes:
        if node.node_type == "case":
            continue
        grouped[node.node_type].append(node.text)

    summary: dict[str, dict[str, Any]] = {}
    for node_type, values in grouped.items():
        summary[node_type] = {
            "count": len(values),
            "examples": values[:3],
        }
    return summary


def _summary_examples(summary: dict[str, dict[str, Any]], node_type: str) -> list[str]:
    payload = summary.get(node_type, {})
    return [str(item) for item in payload.get("examples", [])]


def _summary_count(summary: dict[str, dict[str, Any]], node_type: str) -> int:
    payload = summary.get(node_type, {})
    return int(payload.get("count", 0))


def _format_list_examples(values: list[str], *, width: int = 26, limit: int = 2) -> str:
    if not values:
        return "None"
    parts = [_shorten(value, width=width, max_lines=1) for value in values[:limit]]
    return "\n".join(parts)


def _top_bridge_rows_for_case(
    data: Any,
    case_id: str,
    *,
    per_type_limits: dict[str, int] | None = None,
    other_case_limit: int = 2,
) -> list[dict[str, Any]]:
    if per_type_limits is None:
        per_type_limits = {"court": 2, "judge": 1, "lawyer": 1}

    case_ids = list(data["case"].case_id)
    case_idx = case_ids.index(case_id)
    direct_nodes_by_type: dict[str, list[int]] = defaultdict(list)
    for edge_type in data.edge_types:
        if edge_type[0] != "case" or edge_type[1].startswith("rev_"):
            continue
        edge_index = data[edge_type].edge_index
        matches = (edge_index[0] == case_idx).nonzero(as_tuple=True)[0].tolist()
        for match_idx in matches:
            direct_nodes_by_type[edge_type[2]].append(int(edge_index[1, match_idx].item()))

    rows: list[dict[str, Any]] = []
    for node_type in ("court", "judge", "lawyer"):
        type_rows: list[dict[str, Any]] = []
        for node_idx in direct_nodes_by_type.get(node_type, []):
            other_case_indices: list[int] = []
            for edge_type in data.edge_types:
                if edge_type[0] != node_type or edge_type[2] != "case":
                    continue
                edge_index = data[edge_type].edge_index
                matches = (edge_index[0] == node_idx).nonzero(as_tuple=True)[0].tolist()
                for match_idx in matches:
                    other_idx = int(edge_index[1, match_idx].item())
                    if other_idx != case_idx:
                        other_case_indices.append(other_idx)
            other_case_indices = list(dict.fromkeys(other_case_indices))
            if not other_case_indices:
                continue
            type_rows.append(
                {
                    "node_type": node_type,
                    "shared_node_label": _global_node_label(data, node_type, node_idx).replace("\n", " "),
                    "other_case_count": len(other_case_indices),
                    "sample_other_cases": [
                        _human_case_id(str(data["case"].case_id[other_idx]))
                        for other_idx in other_case_indices[:other_case_limit]
                    ],
                }
            )
        type_rows.sort(key=lambda row: (-int(row["other_case_count"]), str(row["shared_node_label"])))
        rows.extend(type_rows[: int(per_type_limits.get(node_type, 0))])
    return rows


def save_case_representation_story_view(
    cleaned_case: CleanedCase,
    graph_cfg: dict[str, Any],
    output_path: str | Path,
) -> dict[str, dict[str, Any]]:
    summary = _summarize_case_graph(cleaned_case, graph_cfg)
    preamble_len = len(cleaned_case.texts.get("preamble", ""))
    facts_len = len(cleaned_case.texts.get("facts", ""))
    arguments_len = len(cleaned_case.texts.get("arguments", ""))

    fig, ax = plt.subplots(figsize=(18, 12))
    _set_axis_clean(ax)

    ax.text(
        9.0,
        12.25,
        "One Real Case As a Graph",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        9.0,
        11.75,
        "This figure shows exactly how one selected case is represented before it is merged into the full global graph.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )
    _draw_story_box(
        ax,
        6.5,
        5.25,
        5.0,
        1.8,
        "Case Node",
        _human_case_id(cleaned_case.case_id),
        NODE_COLORS["case"],
        fontsize=11,
    )

    _draw_story_box(
        ax,
        0.8,
        8.2,
        4.0,
        1.55,
        "Preamble",
        f"1 text node\n{preamble_len} characters kept\nPre-judgment introduction and setup",
        NODE_COLORS["preamble"],
    )
    _draw_story_box(
        ax,
        6.5,
        8.75,
        5.0,
        1.55,
        "Facts",
        f"1 text node\n{facts_len} characters kept\nBackground facts before the ruling",
        NODE_COLORS["facts"],
    )
    _draw_story_box(
        ax,
        13.2,
        8.2,
        4.0,
        1.55,
        "Arguments",
        f"1 text node\n{arguments_len} characters kept\nThis node links the case to legal citations",
        NODE_COLORS["arguments"],
    )

    _draw_story_box(
        ax,
        0.6,
        5.0,
        4.3,
        1.55,
        "Petitioner",
        f"{_summary_count(summary, 'petitioner')} node\n{_format_list_examples(_summary_examples(summary, 'petitioner'), width=30, limit=1)}",
        NODE_COLORS["petitioner"],
    )
    _draw_story_box(
        ax,
        13.1,
        5.0,
        4.3,
        1.55,
        "Respondent",
        f"{_summary_count(summary, 'respondent')} node\n{_format_list_examples(_summary_examples(summary, 'respondent'), width=30, limit=1)}",
        NODE_COLORS["respondent"],
    )

    _draw_story_box(
        ax,
        0.8,
        2.0,
        4.0,
        1.75,
        "Court Nodes",
        f"{_summary_count(summary, 'court')} nodes\n{_format_list_examples(_summary_examples(summary, 'court'), width=24, limit=2)}"
        + ("\n+ more" if _summary_count(summary, "court") > 2 else ""),
        NODE_COLORS["court"],
    )
    _draw_story_box(
        ax,
        6.5,
        1.25,
        5.0,
        1.45,
        "Judge Nodes",
        f"{_summary_count(summary, 'judge')} node\n{_format_list_examples(_summary_examples(summary, 'judge'), width=30, limit=1)}",
        NODE_COLORS["judge"],
    )
    _draw_story_box(
        ax,
        13.2,
        2.0,
        4.0,
        1.75,
        "Lawyer Nodes",
        f"{_summary_count(summary, 'lawyer')} nodes\n{_format_list_examples(_summary_examples(summary, 'lawyer'), width=22, limit=2)}"
        + ("\n+ more" if _summary_count(summary, "lawyer") > 2 else ""),
        NODE_COLORS["lawyer"],
    )

    _draw_story_box(
        ax,
        5.75,
        0.05,
        6.5,
        0.95,
        "Other Direct Context",
        "GPE "
        + str(_summary_count(summary, "gpe"))
        + " | Date "
        + str(_summary_count(summary, "date"))
        + " | Case Number "
        + str(_summary_count(summary, "case_number"))
        + " | Org "
        + str(_summary_count(summary, "org"))
        + "\nExamples: "
        + _shorten(", ".join(_summary_examples(summary, "gpe")[:2] + _summary_examples(summary, "case_number")[:1]), width=65, max_lines=2),
        NODE_COLORS["org"],
        header_height=0.28,
        fontsize=9.6,
    )

    _draw_story_box(
        ax,
        10.9,
        10.45,
        3.0,
        0.98,
        "Statutes",
        f"{_summary_count(summary, 'statute')} nodes\n{_format_list_examples(_summary_examples(summary, 'statute'), width=20, limit=2)}",
        NODE_COLORS["statute"],
        header_height=0.28,
        fontsize=9.0,
    )
    _draw_story_box(
        ax,
        14.05,
        10.45,
        3.0,
        0.98,
        "Provisions",
        f"{_summary_count(summary, 'provision')} nodes\n"
        f"{_shorten(', '.join(_summary_examples(summary, 'provision')[:2]), width=20, max_lines=1)}"
        + (f"\n+{max(0, _summary_count(summary, 'provision') - 2)} more" if _summary_count(summary, "provision") > 2 else ""),
        NODE_COLORS["provision"],
        header_height=0.28,
        fontsize=9.0,
    )
    _draw_story_box(
        ax,
        7.75,
        10.45,
        3.0,
        0.98,
        "Precedents",
        f"{_summary_count(summary, 'precedent')} nodes\n2 cited cases",
        NODE_COLORS["precedent"],
        header_height=0.28,
        fontsize=9.0,
    )

    case_center = (9.0, 6.15)
    _arrow(ax, (2.8, 8.2), (7.3, 7.05))
    _arrow(ax, (9.0, 8.75), (9.0, 7.05))
    _arrow(ax, (15.2, 8.2), (10.8, 7.05))
    _arrow(ax, (4.9, 5.8), (6.5, 6.0))
    _arrow(ax, (13.1, 5.8), (11.5, 6.0))
    _arrow(ax, (2.8, 3.75), (7.0, 5.2))
    _arrow(ax, (9.0, 2.7), (9.0, 5.2))
    _arrow(ax, (15.2, 3.75), (11.0, 5.2))
    _arrow(ax, (9.0, 1.0), (9.0, 5.2))

    _arrow(ax, (15.2, 9.75), (12.55, 10.8), color=NODE_COLORS["arguments"])
    _arrow(ax, (15.2, 9.75), (15.55, 10.8), color=NODE_COLORS["arguments"])
    _arrow(ax, (15.2, 9.75), (9.25, 10.8), color=NODE_COLORS["arguments"], connectionstyle="arc3,rad=0.08")

    ax.text(
        1.0,
        10.95,
        "The center `case` node is the object the model will finally classify.",
        ha="left",
        va="center",
        fontsize=10.3,
        color="#334155",
    )
    ax.text(
        1.0,
        0.65,
        "This example case has direct party, bench, and context nodes.\nIts legal citations are one step farther away through the `arguments` node.",
        ha="left",
        va="center",
        fontsize=10.3,
        color="#334155",
    )

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12.8)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return summary


def save_layer_connection_story_view(
    cleaned_case: CleanedCase,
    graph_cfg: dict[str, Any],
    data: Any,
    case_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    summary = _summarize_case_graph(cleaned_case, graph_cfg)
    bridge_stats = _bridge_stats_for_case(data, case_id)

    fig, ax = plt.subplots(figsize=(20, 8.5))
    _set_axis_clean(ax)

    panels = [
        (0.4, 0.55, 5.8, 7.2, "1. Local graph before message passing"),
        (7.1, 0.55, 5.8, 7.2, "2. After HGT Layer 1"),
        (13.8, 0.55, 5.8, 7.2, "3. After HGT Layer 2"),
    ]
    for x, y, w, h, title in panels:
        panel = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            linewidth=1.4,
            edgecolor="#cbd5e1",
            facecolor="#f8fafc",
        )
        ax.add_patch(panel)
        ax.text(x + 0.22, y + h - 0.22, title, ha="left", va="top", fontsize=13, fontweight="bold", color="#0f172a")

    _draw_story_box(
        ax,
        2.15,
        3.1,
        2.3,
        1.15,
        "Case",
        "The selected\ncase node",
        NODE_COLORS["case"],
        header_height=0.28,
        fontsize=10,
    )
    _draw_story_box(ax, 0.9, 5.6, 1.75, 0.95, "Text", "Preamble\nFacts\nArguments", NODE_COLORS["arguments"], header_height=0.24, fontsize=9.2)
    _draw_story_box(ax, 0.9, 1.75, 1.75, 0.95, "Bench", "Court\nJudge\nLawyer", NODE_COLORS["court"], header_height=0.24, fontsize=9.2)
    _draw_story_box(ax, 3.95, 5.55, 1.75, 0.95, "Parties", "Petitioner\nRespondent", NODE_COLORS["petitioner"], header_height=0.24, fontsize=9.2)
    _draw_story_box(ax, 4.1, 1.75, 1.6, 0.95, "Context", "GPE\nDate\nCase no.\nOrg", NODE_COLORS["org"], header_height=0.24, fontsize=8.8)
    _draw_story_box(ax, 2.15, 6.2, 2.3, 0.95, "Citations", "Statute\nProvision\nPrecedent", NODE_COLORS["statute"], header_height=0.24, fontsize=9.0)
    _arrow(ax, (1.8, 5.6), (2.6, 4.25))
    _arrow(ax, (4.85, 5.55), (4.0, 4.25))
    _arrow(ax, (1.8, 2.7), (2.6, 3.1))
    _arrow(ax, (4.9, 2.7), (4.0, 3.1))
    _arrow(ax, (3.3, 6.2), (3.3, 4.25), linestyle="--", color=NODE_COLORS["arguments"])
    ax.text(
        3.3,
        0.95,
        "This is the graph structure around one case.\nThe `case` node sits in the middle.",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )

    _draw_story_box(
        ax,
        8.55,
        2.8,
        2.9,
        1.5,
        "Case Embedding",
        "After layer 1, the case node has mixed information from all direct neighbors.",
        NODE_COLORS["case"],
        fontsize=10.2,
    )
    _draw_story_box(
        ax,
        7.55,
        5.4,
        4.7,
        1.25,
        "What layer 1 can use",
        "Text nodes: preamble, facts, arguments\nParties and bench: petitioner, respondent, court, judge, lawyer\nDirect context: GPE, date, case number, org",
        NODE_COLORS["arguments"],
        header_height=0.28,
        fontsize=9.4,
    )
    _arrow(ax, (9.9, 5.4), (9.9, 4.3), color="#475569")
    ax.text(
        10.0,
        1.0,
        "Layer 1 = the case node receives 1-hop information only.",
        ha="center",
        va="center",
        fontsize=10.4,
        color="#334155",
    )

    _draw_story_box(
        ax,
        15.1,
        2.65,
        3.1,
        1.7,
        "Updated Case Embedding",
        "After layer 2, the case node also sees second-hop nodes.",
        NODE_COLORS["case"],
        fontsize=10.2,
    )
    _draw_story_box(
        ax,
        14.15,
        5.55,
        2.15,
        1.15,
        "Local 2-hop path",
        f"case -> arguments -> citations\nStatute {_summary_count(summary, 'statute')}\nProvision {_summary_count(summary, 'provision')}\nPrecedent {_summary_count(summary, 'precedent')}",
        NODE_COLORS["statute"],
        header_height=0.26,
        fontsize=8.9,
    )
    _draw_story_box(
        ax,
        16.65,
        5.55,
        2.55,
        1.15,
        "Global 2-hop path",
        "case -> court/judge/lawyer -> other cases\n"
        f"Court {bridge_stats.get('court', {}).get('other_cases', 0)}\n"
        f"Judge {bridge_stats.get('judge', {}).get('other_cases', 0)}\n"
        f"Lawyer {bridge_stats.get('lawyer', {}).get('other_cases', 0)}",
        NODE_COLORS["court"],
        header_height=0.26,
        fontsize=8.8,
    )
    _arrow(ax, (15.25, 5.55), (16.0, 4.35), color=NODE_COLORS["statute"])
    _arrow(ax, (17.9, 5.55), (17.0, 4.35), color=NODE_COLORS["court"])
    _draw_story_box(
        ax,
        14.2,
        0.9,
        5.05,
        0.95,
        "Still NOT reachable in 2 layers",
        "Cross-case sharing through statutes/provisions is farther away:\ncase -> arguments -> statute -> arguments(other case) -> other case",
        "#b91c1c",
        header_height=0.24,
        fontsize=8.9,
    )
    ax.text(
        16.7,
        7.05,
        "Layer 2 adds 2-hop information.\nThat includes both local legal citations and some shared cross-case context.",
        ha="center",
        va="top",
        fontsize=10,
        color="#334155",
    )

    _arrow(ax, (6.25, 4.15), (7.05, 4.15), color="#475569", mutation_scale=22)
    _arrow(ax, (12.95, 4.15), (13.75, 4.15), color="#475569", mutation_scale=22)
    ax.text(6.65, 4.45, "HGT\nLayer 1", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#334155")
    ax.text(13.35, 4.45, "HGT\nLayer 2", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#334155")

    ax.text(
        10.0,
        8.1,
        "How the current 2-layer HGT sees the selected case",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        10.0,
        7.7,
        "The key idea: each new GNN layer lets the case node absorb information one graph-hop farther away.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.4)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "bridge_stats": bridge_stats,
        "summary": summary,
    }


def save_cross_case_connection_story_view(
    data: Any,
    case_id: str,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    bridge_rows = _top_bridge_rows_for_case(data, case_id)

    fig, ax = plt.subplots(figsize=(18, 10))
    _set_axis_clean(ax)

    ax.text(
        9.0,
        9.5,
        "How This One Case Connects To Other Cases",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        9.0,
        9.0,
        "The selected case is not isolated. It shares normalized nodes with many other cases in the global graph.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    _draw_story_box(
        ax,
        0.7,
        3.55,
        3.9,
        2.0,
        "Selected Case",
        _human_case_id(case_id),
        NODE_COLORS["case"],
        fontsize=11,
    )

    row_y = [7.0, 5.2, 3.4, 1.6]
    for row, y in zip(bridge_rows, row_y):
        node_type = str(row["node_type"])
        label = str(row["shared_node_label"])
        other_cases = list(row["sample_other_cases"])
        extra_cases = max(0, int(row["other_case_count"]) - len(other_cases))
        other_body = "\n".join(other_cases)
        if extra_cases > 0:
            other_body = (other_body + "\n" if other_body else "") + f"+{extra_cases} more related cases"

        _draw_story_box(
            ax,
            6.2,
            y,
            4.0,
            1.2,
            f"Shared {_display_name(node_type)} Node",
            f"{_shorten(label, width=28, max_lines=2)}\n{row['other_case_count']} other cases connected here",
            NODE_COLORS.get(node_type, "#64748b"),
            header_height=0.28,
            fontsize=9.3,
        )
        _draw_story_box(
            ax,
            12.0,
            y,
            4.8,
            1.2,
            "Example Other Cases",
            other_body,
            "#475569",
            header_height=0.28,
            fontsize=8.8,
        )
        _arrow(ax, (4.6, 4.55), (6.2, y + 0.6), color=NODE_COLORS.get(node_type, "#64748b"))
        _arrow(ax, (10.2, y + 0.6), (12.0, y + 0.6), color=NODE_COLORS.get(node_type, "#64748b"))

    ax.text(
        9.0,
        0.7,
        "This is the global graph effect: the same shared node object is reused across many case stars.\nThat lets the GNN pass information from one case neighborhood to another through courts, judges, lawyers, and other shared entities.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#334155",
    )

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return bridge_rows


def save_story_readme(
    output_path: str | Path,
    selected_case_id: str,
    case_summary: dict[str, dict[str, Any]],
    layer_summary: dict[str, Any],
    bridge_rows: list[dict[str, Any]],
    model_cfg: dict[str, Any],
) -> None:
    bridge_stats = layer_summary.get("bridge_stats", {})
    lines = [
        "# Understanding the Current Case-Star GNN",
        "",
        f"Selected example case: `{selected_case_id}`",
        "",
        "This folder is meant to answer three simple questions:",
        "",
        "1. What does one case become inside the graph?",
        "2. What do the GNN layers actually do to that case?",
        "3. How does that case become connected to other cases?",
        "",
        "Look at the files in this order:",
        "",
        "1. `01_one_case_representation.png`",
        "2. `02_how_two_hgt_layers_work.png`",
        "3. `03_how_this_case_connects_to_other_cases.png`",
        "",
        "## 1. What one case becomes in the graph",
        "",
        "Each case is turned into one central `case` node. That is the node the classifier will finally predict on.",
        "",
        "Around that node, the pipeline attaches pre-judgment information only:",
        "",
        "- text nodes: `preamble`, `facts`, `arguments`",
        "- party nodes: `petitioner`, `respondent`",
        "- bench and process nodes: `court`, `judge`, `lawyer`",
        "- contextual nodes: `gpe`, `date`, `case_number`, `org`",
        "- legal citation nodes: `statute`, `provision`, `precedent`",
        "",
        "For the selected case, the current local graph contains:",
        "",
    ]
    for node_type in ("preamble", "facts", "arguments", "petitioner", "respondent", "court", "judge", "lawyer", "statute", "provision", "precedent", "gpe", "date", "case_number", "org"):
        if node_type not in case_summary:
            continue
        count = int(case_summary[node_type]["count"])
        examples = ", ".join(str(item) for item in case_summary[node_type]["examples"][:2])
        if node_type in {"preamble", "facts", "arguments"}:
            lines.append(f"- `{node_type}`: {count} node, kept as cleaned pre-judgment text")
        else:
            lines.append(f"- `{node_type}`: {count} nodes. Examples: {examples}")

    lines.extend(
        [
            "",
            "The important structural point is this:",
            "",
            "- most nodes are attached directly to the `case` node",
            "- `statute`, `provision`, and `precedent` are attached through the `arguments` node",
            "",
            "So this graph is not a bag of text. It is a structured legal neighborhood around the selected case.",
            "",
            "## 2. What the two GNN layers actually do",
            "",
            "The current model uses:",
            "",
            f"- architecture: `{model_cfg.get('architecture', 'hgt')}`",
            f"- hidden dim: `{model_cfg.get('hidden_dim', 128)}`",
            f"- layers: `{model_cfg.get('num_layers', 2)}`",
            f"- heads: `{model_cfg.get('num_heads', 4)}`",
            "",
            "Think of each HGT layer as allowing the `case` node to absorb information one graph-hop farther away.",
            "",
            "### Before message passing",
            "",
            "- every node starts with an initial feature vector",
            "- text nodes start from text embeddings",
            "- entity nodes start from normalized-name features and metadata",
            "",
            "### After layer 1",
            "",
            "- the `case` node can mix information from every direct neighbor",
            "- that includes preamble, facts, arguments, petitioner, respondent, court, judge, lawyer, and the directly attached context nodes",
            "",
            "### After layer 2",
            "",
            "- the `case` node can now absorb information from neighbors-of-neighbors",
            "- local legal citation information becomes reachable through the path `case -> arguments -> statute/provision/precedent`",
            "- cross-case information becomes reachable through direct shared nodes such as `court`, `judge`, and `lawyer`",
            "",
            "For this selected case, the strongest current cross-case paths are:",
            "",
            f"- via `court`: {bridge_stats.get('court', {}).get('other_cases', 0)} other cases",
            f"- via `judge`: {bridge_stats.get('judge', {}).get('other_cases', 0)} other cases",
            f"- via `lawyer`: {bridge_stats.get('lawyer', {}).get('other_cases', 0)} other cases",
            "",
            "## 3. What is not reachable in only two layers",
            "",
            "This point is easy to miss and very important.",
            "",
            "The model can use this case's own statutes and provisions in two layers, but it cannot yet use other cases that only overlap through statutes or provisions.",
            "",
            "Why not?",
            "",
            "Because the cross-case path is longer:",
            "",
            "`case -> arguments -> statute -> arguments(other case) -> case(other case)`",
            "",
            "That is effectively a 4-hop path, so the current 2-layer model does not propagate that far.",
            "",
            "## 4. How this case becomes connected to other cases",
            "",
            "The global graph is created by reusing the same normalized shared node across many cases.",
            "",
            "Example:",
            "",
            "- if two cases mention the same normalized court name, both case stars connect to the same `court` node",
            "- if two cases share the same normalized judge name, both connect to the same `judge` node",
            "- the same idea holds for lawyers and other shareable authority/context nodes",
            "",
            "The selected case currently has these visible shared bridges:",
            "",
        ]
    )
    for row in bridge_rows:
        samples = ", ".join(str(case_name).replace("\n", " ") for case_name in row["sample_other_cases"])
        lines.append(
            f"- `{row['node_type']}` node `{row['shared_node_label']}` connects this case to {row['other_case_count']} other cases. Sample linked cases: {samples}"
        )
    lines.extend(
        [
            "",
            "This does not mean the model copies labels from those cases. It means the selected case embedding is updated using messages that flow through those shared legal-context nodes.",
            "",
            "## 5. What the final classifier actually sees",
            "",
            "After the two HGT layers finish, the model takes the final embedding of the selected `case` node and sends only that vector into the classifier head.",
            "",
            "So the prediction is not made from one text field in isolation. It is made from a learned summary of:",
            "",
            "- the case's own pre-judgment text",
            "- its parties, bench, and context",
            "- its local citations",
            "- the nearby cross-case structure reachable through shared nodes",
            "",
            "## 6. One-sentence mental model",
            "",
            "A case starts as a small legal star graph, then the GNN lets that case absorb information from its own neighborhood and from nearby cases that share courts, judges, lawyers, and other normalized authority nodes.",
        ]
    )
    ensure_dir(Path(output_path).parent)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
