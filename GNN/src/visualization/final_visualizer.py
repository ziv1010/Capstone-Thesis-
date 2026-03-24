from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.graph.schema import CleanedCase
from src.utils.io import ensure_dir
from src.visualization.graph_visualizer import (
    NODE_COLORS,
    _arrow,
    _bridge_stats_for_case,
    _display_name,
    _draw_story_box,
    _human_case_id,
    _set_axis_clean,
    _shorten,
    _summarize_case_graph,
    _summary_count,
    _summary_examples,
    _top_bridge_rows_for_case,
)


def save_final_node_storage_view(
    cleaned_case: CleanedCase,
    graph_cfg: dict[str, Any],
    cfg: dict[str, Any],
    graph_metadata: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    summary = _summarize_case_graph(cleaned_case, graph_cfg)
    embedding_dim = int(graph_metadata.get("embedding_dim", 0))
    scalar_dim = int(graph_metadata.get("scalar_dim", 0))
    feature_dim = int(graph_metadata.get("feature_dim", 0))
    case_scalar_names = list(cfg.get("features", {}).get("case_scalar_names", []))
    entity_scalar_names = list(cfg.get("features", {}).get("entity_scalar_names", []))

    court_example = _summary_examples(summary, "court")
    court_name = court_example[0] if court_example else "high court at bombay"

    fig, ax = plt.subplots(figsize=(19, 11))
    _set_axis_clean(ax)

    ax.text(
        9.5,
        10.45,
        "What Is Actually Stored Inside Vertices",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        9.5,
        9.95,
        "The text written inside the diagram nodes is only a human-readable label. The real model input is a numeric feature vector on each node.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    _draw_story_box(
        ax,
        0.7,
        4.9,
        5.5,
        4.0,
        "Case Vertex `case::A`",
        "Display label in the figure:\n"
        f"{_human_case_id(cleaned_case.case_id)}\n\n"
        f"Stored feature vector `x_case^0` ({feature_dim} dims total):\n"
        f"- {embedding_dim}-d text embedding of cleaned pre-judgment\n"
        "  preamble + facts + arguments\n"
        f"- {scalar_dim} case scalars\n\n"
        "Case scalars used now:\n"
        + "\n".join(f"- {name}" for name in case_scalar_names[:6])
        + ("\n- ..." if len(case_scalar_names) > 6 else "")
        + "\n\n"
        f"Example counts for this case:\n"
        f"- respondents={_summary_count(summary, 'respondent')}\n"
        f"- judges={_summary_count(summary, 'judge')}\n"
        f"- lawyers={_summary_count(summary, 'lawyer')}\n"
        f"- statutes={_summary_count(summary, 'statute')}\n"
        f"- provisions={_summary_count(summary, 'provision')}",
        NODE_COLORS["case"],
        fontsize=10.0,
    )

    _draw_story_box(
        ax,
        6.8,
        4.9,
        5.4,
        4.0,
        "Text Vertex Example `arguments`",
        "Display label in the figure:\n`arguments`\n\n"
        f"Stored feature vector `x_arguments^0` ({feature_dim} dims total):\n"
        f"- {embedding_dim}-d text embedding of the cleaned arguments text\n"
        f"- {scalar_dim}-slot scalar tail, with the first 4 used here\n\n"
        "Text-node scalars used now:\n"
        "- text_length\n"
        "- is_preamble\n"
        "- is_facts\n"
        "- is_arguments\n\n"
        f"Example for this case:\n- arguments length={len(cleaned_case.texts.get('arguments', ''))}\n"
        "- text is leakage-cleaned before embedding",
        NODE_COLORS["arguments"],
        fontsize=10.0,
    )

    _draw_story_box(
        ax,
        13.0,
        4.9,
        5.3,
        4.0,
        f"Entity Vertex Example `court::{court_name}`",
        "Display label in the figure:\n"
        f"{_shorten(court_name, width=34, max_lines=2)}\n\n"
        f"Stored feature vector `x_court^0` ({feature_dim} dims total):\n"
        f"- {embedding_dim}-d embedding of the normalized string\n"
        f"- {scalar_dim}-slot scalar tail, with entity fields filled\n\n"
        "Entity scalars used now:\n"
        + "\n".join(f"- {name}" for name in entity_scalar_names[:7])
        + ("\n- ..." if len(entity_scalar_names) > 7 else "")
        + "\n\n"
        "Important:\n"
        "- shared entities do not have labels\n"
        "- they only provide context and message paths",
        NODE_COLORS["court"],
        fontsize=10.0,
    )

    _draw_story_box(
        ax,
        0.9,
        1.45,
        17.1,
        2.3,
        "What Is NOT Stored In The Vertex Features",
        "- the outcome label is not part of `x_case`\n"
        "- `case_outcome_label` is used only later as supervision target `data['case'].y`\n"
        "- the case name string is mainly a node identifier, not the main semantic feature\n"
        "- decision text and other leakage-bearing text are excluded before features are built",
        "#991b1b",
        fontsize=10.2,
    )

    _draw_story_box(
        ax,
        5.0,
        0.15,
        9.0,
        0.95,
        "Mental Model",
        "A case vertex already has its own information at layer 0. The GNN layers do not create information from nothing; they add neighbor information on top of the case's own initial vector.",
        "#475569",
        header_height=0.24,
        fontsize=9.4,
    )

    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.8)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "embedding_dim": embedding_dim,
        "scalar_dim": scalar_dim,
        "feature_dim": feature_dim,
        "case_scalar_names": case_scalar_names,
        "entity_scalar_names": entity_scalar_names,
    }


def save_final_edge_storage_view(output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    _set_axis_clean(ax)

    ax.text(
        9.0,
        8.45,
        "What Is Actually Stored On Edges",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        9.0,
        7.95,
        "In the current pipeline, edges do not carry their own numeric feature vectors. They mainly store connectivity and relation type.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    _draw_story_box(
        ax,
        0.8,
        4.55,
        4.7,
        2.5,
        "Example Edge 1",
        "`case` --has_arguments--> `arguments`\n\n"
        "Meaning:\n"
        "- this arguments node belongs to this case\n"
        "- during message passing, relation type tells HGT\n"
        "  which relation-specific transform to use",
        NODE_COLORS["arguments"],
        fontsize=10.0,
    )
    _draw_story_box(
        ax,
        6.2,
        4.55,
        5.0,
        2.5,
        "Example Edge 2",
        "`arguments` --cites_statute--> `statute`\n\n"
        "Meaning:\n"
        "- this case's arguments cite this statute\n"
        "- this creates a legal path from case text to citation nodes",
        NODE_COLORS["statute"],
        fontsize=10.0,
    )
    _draw_story_box(
        ax,
        12.0,
        4.55,
        5.0,
        2.5,
        "Example Edge 3",
        "`case` --heard_in--> `court`\n\n"
        "Meaning:\n"
        "- this case is linked to that normalized court node\n"
        "- if other cases share the same court node, this becomes a cross-case bridge",
        NODE_COLORS["court"],
        fontsize=10.0,
    )

    _draw_story_box(
        ax,
        1.0,
        1.35,
        7.8,
        2.2,
        "Stored Now",
        "- source index and destination index in `edge_index`\n"
        "- heterogeneous edge type: `(src_type, relation, dst_type)`\n"
        "- reverse direction edges are added by `ToUndirected`, so messages can flow back\n"
        "  from neighbors to the case node",
        "#166534",
        fontsize=10.2,
    )
    _draw_story_box(
        ax,
        9.3,
        1.35,
        7.8,
        2.2,
        "Not Stored Now",
        "- no edge feature vector `edge_attr`\n"
        "- no edge weights for importance, time, or confidence\n"
        "- no label on edges\n"
        "- no custom citation-strength signal yet",
        "#991b1b",
        fontsize=10.2,
    )

    ax.text(
        9.0,
        0.45,
        "So when you ask what an edge stores in the current model, the honest answer is: relation type plus graph connectivity, not its own learned content vector.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#334155",
    )

    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.9)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_final_layer1_view(
    cleaned_case: CleanedCase,
    graph_cfg: dict[str, Any],
    cfg: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    summary = _summarize_case_graph(cleaned_case, graph_cfg)
    num_heads = int(cfg.get("model", {}).get("num_heads", 4))
    hidden_dim = int(cfg.get("model", {}).get("hidden_dim", 128))

    fig, ax = plt.subplots(figsize=(20, 10))
    _set_axis_clean(ax)

    ax.text(
        10.0,
        9.35,
        "How Layer 1 Works For Case A",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        10.0,
        8.9,
        "Layer 1 only aggregates direct neighbors of the case node.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    panel = FancyBboxPatch(
        (0.5, 0.8),
        8.8,
        7.4,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        linewidth=1.4,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
    )
    ax.add_patch(panel)
    ax.text(0.8, 7.95, "Direct neighborhood around Case A", ha="left", va="top", fontsize=13, fontweight="bold", color="#0f172a")

    _draw_story_box(
        ax,
        3.45,
        3.85,
        2.9,
        1.4,
        "Case A at Layer 0",
        "Own initial case vector:\ntext embedding + case scalars",
        NODE_COLORS["case"],
        fontsize=10.0,
    )

    direct_specs = [
        ("Text nodes", f"preamble { _summary_count(summary, 'preamble') }\nfacts { _summary_count(summary, 'facts') }\narguments { _summary_count(summary, 'arguments') }", NODE_COLORS["arguments"], (1.1, 6.0)),
        ("Party nodes", f"petitioner { _summary_count(summary, 'petitioner') }\nrespondent { _summary_count(summary, 'respondent') }", NODE_COLORS["petitioner"], (0.9, 2.0)),
        ("Bench nodes", f"court { _summary_count(summary, 'court') }\njudge { _summary_count(summary, 'judge') }\nlawyer { _summary_count(summary, 'lawyer') }", NODE_COLORS["court"], (6.8, 2.0)),
        ("Context nodes", f"gpe { _summary_count(summary, 'gpe') }\ndate { _summary_count(summary, 'date') }\ncase_number { _summary_count(summary, 'case_number') }\norg { _summary_count(summary, 'org') }", NODE_COLORS["org"], (6.7, 6.0)),
    ]
    for title, body, color, (x, y) in direct_specs:
        _draw_story_box(ax, x, y, 2.0, 1.35, title, body, color, header_height=0.28, fontsize=9.2)

    _arrow(ax, (3.1, 6.0), (4.1, 5.25))
    _arrow(ax, (2.9, 3.35), (4.0, 4.1))
    _arrow(ax, (6.8, 3.35), (6.15, 4.1))
    _arrow(ax, (6.7, 6.0), (5.95, 5.25))

    panel2 = FancyBboxPatch(
        (10.0, 0.8),
        9.5,
        7.4,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        linewidth=1.4,
        edgecolor="#cbd5e1",
        facecolor="#f8fafc",
    )
    ax.add_patch(panel2)
    ax.text(10.3, 7.95, "What HGT Layer 1 computes", ha="left", va="top", fontsize=13, fontweight="bold", color="#0f172a")

    _draw_story_box(
        ax,
        10.6,
        5.85,
        8.1,
        1.55,
        "Step 1: Project Every Node To Hidden Space",
        f"`z_v^0 = Linear(x_v) + type_embedding[node_type]`\nAll node types are projected to the same hidden size {hidden_dim}.",
        "#1d4ed8",
        fontsize=10.0,
    )
    _draw_story_box(
        ax,
        10.6,
        3.75,
        8.1,
        1.6,
        "Step 2: Send Relation-Specific Messages Into Case A",
        f"HGT uses different transforms for different edge types.\nExample: messages through `has_arguments`, `heard_in`, and `has_lawyer` are not treated identically.\nCurrent model: {num_heads} attention heads.",
        "#7c2d12",
        fontsize=10.0,
    )
    _draw_story_box(
        ax,
        10.6,
        1.5,
        8.1,
        1.8,
        "Step 3: Update Case A",
        "`h_case^1 = ReLU(LayerNorm(z_case^0 + aggregated_messages_from_1-hop_neighbors))`\n\n"
        "Important:\n"
        "- the case keeps its own initial information\n"
        "- direct-neighbor information is added on top of it\n"
        "- Layer 1 still does not see 2-hop nodes",
        "#166534",
        fontsize=10.0,
    )

    arrow = FancyArrowPatch((9.35, 4.6), (9.95, 4.6), arrowstyle="-|>", mutation_scale=22, linewidth=1.7, color="#475569")
    ax.add_patch(arrow)
    ax.text(9.65, 5.0, "HGT\nLayer 1", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#334155")

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9.8)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "direct_neighbor_counts": {
            "preamble": _summary_count(summary, "preamble"),
            "facts": _summary_count(summary, "facts"),
            "arguments": _summary_count(summary, "arguments"),
            "petitioner": _summary_count(summary, "petitioner"),
            "respondent": _summary_count(summary, "respondent"),
            "court": _summary_count(summary, "court"),
            "judge": _summary_count(summary, "judge"),
            "lawyer": _summary_count(summary, "lawyer"),
            "gpe": _summary_count(summary, "gpe"),
            "date": _summary_count(summary, "date"),
            "case_number": _summary_count(summary, "case_number"),
            "org": _summary_count(summary, "org"),
        }
    }


def save_final_layer2_view(
    cleaned_case: CleanedCase,
    graph_cfg: dict[str, Any],
    data: Any,
    case_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    summary = _summarize_case_graph(cleaned_case, graph_cfg)
    bridge_stats = _bridge_stats_for_case(data, case_id)
    bridge_rows = _top_bridge_rows_for_case(data, case_id)

    fig, ax = plt.subplots(figsize=(20, 11))
    _set_axis_clean(ax)

    ax.text(
        10.0,
        10.25,
        "How Layer 2 Works For Case A",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        10.0,
        9.8,
        "Layer 2 does not jump directly to all distant nodes. It only sees what can be reached in two graph hops.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    _draw_story_box(
        ax,
        0.8,
        6.1,
        8.6,
        2.8,
        "Path A: Local 2-hop legal information",
        "Case A does not connect directly to statutes or provisions.\n"
        "Instead, the path is:\n\n"
        "`case A -> arguments -> statute / provision / precedent`\n\n"
        "So after 2 layers, Case A can use its own citation nodes.\n"
        f"For this selected case:\n- statutes={_summary_count(summary, 'statute')}\n"
        f"- provisions={_summary_count(summary, 'provision')}\n"
        f"- precedents={_summary_count(summary, 'precedent')}",
        NODE_COLORS["statute"],
        fontsize=10.0,
    )

    _draw_story_box(
        ax,
        10.1,
        6.1,
        9.0,
        2.8,
        "Path B: Cross-case 2-hop information through shared direct nodes",
        "Because the graph is made undirected, Case A can receive information through paths like:\n\n"
        "`case A <- heard_in -> court <- heard_in -> case B`\n"
        "`case A <- decided_by_bench -> judge <- decided_by_bench -> case C`\n"
        "`case A <- has_lawyer -> lawyer <- has_lawyer -> case D`\n\n"
        "So after 2 layers, some other-case information is reachable.",
        NODE_COLORS["court"],
        fontsize=10.0,
    )

    _draw_story_box(
        ax,
        0.9,
        2.5,
        8.5,
        2.6,
        "What Layer 2 Updates",
        "`h_case^2 = ReLU(LayerNorm(h_case^1 + aggregated_messages_from_updated_1-hop_neighbors))`\n\n"
        "This means the case node now receives messages from neighbors whose own representations already include their neighbors.\n"
        "That is why 2-hop information appears at layer 2.",
        "#1d4ed8",
        fontsize=10.0,
    )

    bridge_body = []
    for row in bridge_rows[:4]:
        bridge_body.append(
            f"- via {row['node_type']} `{row['shared_node_label']}` -> {row['other_case_count']} other cases"
        )
    bridge_text = "\n".join(bridge_body) if bridge_body else "- no shared 2-hop case bridges found"
    _draw_story_box(
        ax,
        10.1,
        2.5,
        9.0,
        2.6,
        "Visible 2-hop cross-case bridges for this selected case",
        bridge_text
        + "\n\n"
        f"Bridge totals:\n- court: {bridge_stats.get('court', {}).get('other_cases', 0)}\n"
        f"- judge: {bridge_stats.get('judge', {}).get('other_cases', 0)}\n"
        f"- lawyer: {bridge_stats.get('lawyer', {}).get('other_cases', 0)}",
        "#166534",
        fontsize=9.8,
    )

    _draw_story_box(
        ax,
        2.2,
        0.35,
        15.5,
        1.45,
        "What Is Still NOT Reachable In Only 2 Layers",
        "Cross-case sharing through statutes is farther away. Example longer path:\n"
        "`case A -> arguments -> statute -> arguments(other case) -> case(other case)`\n"
        "That is effectively a 4-hop case-to-case route, so the current 2-layer model does not propagate that far.",
        "#991b1b",
        fontsize=10.0,
    )

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10.7)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "bridge_stats": bridge_stats,
        "bridge_rows": bridge_rows,
    }


def save_final_training_view(
    graph_metadata: dict[str, Any],
    label_names: list[str],
    output_path: str | Path,
) -> None:
    split_counts = graph_metadata.get("case_split_counts", {})
    train_count = int(split_counts.get("train", 0))
    val_count = int(split_counts.get("val", 0))
    test_count = int(split_counts.get("test", 0))

    fig, ax = plt.subplots(figsize=(19, 10))
    _set_axis_clean(ax)

    ax.text(
        9.5,
        9.3,
        "How Training Uses Labels",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        9.5,
        8.85,
        "Only case nodes are supervised. Other node types help build the case representation but are not assigned win/loss labels.",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    _draw_story_box(
        ax,
        0.9,
        4.9,
        5.0,
        2.6,
        "Full Graph During Training",
        "All cases and all shared nodes are present in one global graph.\n\n"
        f"Current split sizes:\n- train cases={train_count}\n- val cases={val_count}\n- test cases={test_count}",
        "#475569",
        fontsize=10.2,
    )

    _draw_story_box(
        ax,
        7.0,
        4.9,
        5.0,
        2.6,
        "Where Labels Live",
        "The supervision target is stored only on the `case` node type:\n\n"
        "`data['case'].y = [0, 1, ...]`\n\n"
        f"Current class names:\n- {', '.join(label_names)}",
        NODE_COLORS["case"],
        fontsize=10.2,
    )

    _draw_story_box(
        ax,
        13.1,
        4.9,
        5.0,
        2.6,
        "Where Labels Do NOT Live",
        "There is no `win` or `lose` label on:\n- court nodes\n- judge nodes\n- statute nodes\n- lawyer nodes\n- text nodes\n\n"
        "Those nodes may appear in both winning and losing cases.",
        "#991b1b",
        fontsize=10.2,
    )

    _draw_story_box(
        ax,
        1.2,
        1.25,
        16.5,
        2.5,
        "Loss Computation In The Current Setup",
        "The model runs on the whole graph each epoch, but cross-entropy loss is computed only on `data['case'].train_mask`.\n\n"
        "So:\n"
        "- train-case labels supervise the model\n"
        "- val/test case labels are not used in the loss\n"
        "- non-case nodes never receive outcome labels\n"
        "- this is transductive training, because all node features are present in one shared graph",
        "#166534",
        fontsize=10.2,
    )

    ax.set_xlim(0, 19)
    ax.set_ylim(0, 9.8)
    ensure_dir(Path(output_path).parent)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_final_visualisations_readme(
    output_path: str | Path,
    selected_case_id: str,
    node_storage_info: dict[str, Any],
    layer1_info: dict[str, Any],
    layer2_info: dict[str, Any],
    graph_metadata: dict[str, Any],
) -> None:
    bridge_stats = layer2_info.get("bridge_stats", {})
    lines = [
        "# Final Guide To The Current GNN",
        "",
        f"Selected example case: `{selected_case_id}`",
        "",
        "This folder is meant to answer five concrete questions:",
        "",
        "1. What is stored inside a vertex before message passing starts?",
        "2. What is stored on an edge?",
        "3. What does layer 1 add to the case node?",
        "4. What does layer 2 add to the case node?",
        "5. Where do the labels live during training?",
        "",
        "Open the files in this order:",
        "",
        "1. `01_what_is_stored_in_vertices.png`",
        "2. `02_what_is_stored_on_edges.png`",
        "3. `03_how_layer_1_updates_case_A.png`",
        "4. `04_how_layer_2_updates_case_A.png`",
        "5. `05_how_training_uses_labels.png`",
        "",
        "## 1. What is stored inside the case vertex",
        "",
        "The case vertex is not empty and it is not just the case name.",
        "",
        "It already has its own layer-0 feature vector.",
        "",
        f"- feature dimension: `{node_storage_info.get('feature_dim', 'unknown')}`",
        f"- text embedding part: `{node_storage_info.get('embedding_dim', 'unknown')}` dimensions",
        f"- scalar part: `{node_storage_info.get('scalar_dim', 'unknown')}` dimensions",
        "",
        "For the current case node, that initial vector is built from:",
        "",
        "- the cleaned pre-judgment preamble + facts + arguments text",
        "- case-level safe metadata such as counts, lengths, and year",
        "",
        "So before the GNN even looks at neighbors, the case node already contains a summary of its own case.",
        "",
        "## 2. What is stored inside text and entity vertices",
        "",
        "Text nodes store:",
        "",
        "- an embedding of that section's text",
        "- a few section-specific scalars such as length and section identity",
        "",
        "Entity nodes store:",
        "",
        "- an embedding of the normalized string",
        "- metadata such as mention count, first seen section, global frequency, degree, and shared-node flag",
        "",
        "## 3. What is stored on edges",
        "",
        "In the current pipeline, edges mainly store graph structure rather than rich edge features.",
        "",
        "An edge currently contributes:",
        "",
        "- source and destination connectivity in `edge_index`",
        "- the heterogeneous relation type, such as `has_arguments`, `heard_in`, or `cites_statute`",
        "- reverse-direction connectivity after `ToUndirected`",
        "",
        "An edge currently does not have:",
        "",
        "- a numeric edge feature vector",
        "- a confidence score",
        "- a learned citation weight",
        "",
        "## 4. How layer 1 works",
        "",
        "Layer 1 only uses direct neighbors of the case node.",
        "",
        "That means the case can directly absorb information from:",
        "",
    ]
    for node_type, count in layer1_info.get("direct_neighbor_counts", {}).items():
        if count:
            lines.append(f"- `{node_type}`: {count}")
    lines.extend(
        [
            "",
            "Conceptually, layer 1 does this:",
            "",
            "- start with the case's own layer-0 vector",
            "- collect messages from direct neighbors using relation-specific HGT transforms",
            "- add those messages to the case's own representation",
            "- apply residual connection, layer norm, and nonlinearity",
            "",
            "So after layer 1, the case node contains:",
            "",
            "- its own original case information",
            "- plus direct-neighbor information",
            "",
            "## 5. How layer 2 works",
            "",
            "Layer 2 does not directly jump to distant nodes. Instead, it updates the case again using neighbors whose own embeddings were already updated in layer 1.",
            "",
            "That is why 2-hop information becomes available.",
            "",
            "For this selected case, layer 2 can now use:",
            "",
            "- local citation information through `case -> arguments -> statute/provision/precedent`",
            "- some other-case information through shared direct nodes such as court, judge, and lawyer",
            "",
            "Current visible 2-hop bridge totals:",
            "",
            f"- `court`: {bridge_stats.get('court', {}).get('other_cases', 0)} other cases",
            f"- `judge`: {bridge_stats.get('judge', {}).get('other_cases', 0)} other cases",
            f"- `lawyer`: {bridge_stats.get('lawyer', {}).get('other_cases', 0)} other cases",
            "",
            "## 6. What 2 layers still cannot do",
            "",
            "The current 2-layer model cannot fully propagate cross-case information through statutes, because that path is longer.",
            "",
            "Example longer path:",
            "",
            "`case A -> arguments -> statute -> arguments(other case) -> case(other case)`",
            "",
            "That is why adding more layers could change which parts of the graph become reachable.",
            "",
            "## 7. Where labels live",
            "",
            "Only `case` nodes have outcome targets.",
            "",
            "That means:",
            "",
            "- the graph does not create a separate `win` node or `lose` node",
            "- non-case nodes are never labeled win/loss",
            "- the model predicts from the final `case` embedding only",
            "",
            "## 8. One-sentence mental model",
            "",
            "A case vertex starts with its own pre-judgment case summary, layer 1 adds direct-neighbor context, and layer 2 adds second-hop citation and shared-cross-case context.",
            "",
            "## 9. Current graph metadata",
            "",
            f"- feature dim: `{graph_metadata.get('feature_dim', 'unknown')}`",
            f"- embedding dim: `{graph_metadata.get('embedding_dim', 'unknown')}`",
            f"- scalar dim: `{graph_metadata.get('scalar_dim', 'unknown')}`",
            f"- encoder backend: `{graph_metadata.get('encoder_backend', 'unknown')}`",
            f"- split counts: `{graph_metadata.get('case_split_counts', {})}`",
        ]
    )
    ensure_dir(Path(output_path).parent)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
