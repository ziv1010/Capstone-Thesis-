#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
FIXED_GPU_OPENNYAI_ROOT = REPO_ROOT / "Fixed_GPU_OpenNyai"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.schema import ARGUMENT_ROLE_NODE_TYPES, BASE_TEXT_NODE_TYPES, ENTITY_NODE_TYPES, RELATION_DEFINITIONS
from src.utils.io import ensure_dir, load_yaml


plt.rcParams["font.family"] = "DejaVu Sans"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_snapshot() -> dict[str, Any]:
    base_cfg = load_yaml(PROJECT_ROOT / "configs" / "gnn_case_star.yaml")
    current_run_cfg_path = (
        PROJECT_ROOT
        / "outputs"
        / "augmented_jsons_3way"
        / "models"
        / "augmented_jsons_3way"
        / "run_config_snapshot.yaml"
    )
    current_run_cfg = load_yaml(current_run_cfg_path) if current_run_cfg_path.exists() else {}

    preprocess_summary = _load_json(
        PROJECT_ROOT / "data" / "augmented_jsons_3way" / "processed" / "preprocess_summary.json"
    )
    graph_metadata = _load_json(
        PROJECT_ROOT / "data" / "augmented_jsons_3way" / "graph_cache" / "graph_metadata.json"
    )

    model_cfg = current_run_cfg.get("model", {})
    features_cfg = current_run_cfg.get("features", {})
    paths_cfg = current_run_cfg.get("paths", {})

    return {
        "base_raw_json_dir": str(base_cfg.get("paths", {}).get("raw_json_dir", "")),
        "current_raw_json_dir": str(paths_cfg.get("raw_json_dir", "")),
        "current_processed_dir": str(paths_cfg.get("processed_dir", "")),
        "current_graph_cache_dir": str(paths_cfg.get("graph_cache_dir", "")),
        "current_outputs_dir": str(paths_cfg.get("outputs_dir", "")),
        "num_input_files": int(preprocess_summary.get("num_files", 0)),
        "num_cleaned_cases": int(len(preprocess_summary.get("cases", []))),
        "label_names": list(graph_metadata.get("label_names", [])),
        "feature_dim": int(graph_metadata.get("feature_dim", 0)),
        "embedding_dim": int(graph_metadata.get("embedding_dim", 0)),
        "scalar_dim": int(graph_metadata.get("scalar_dim", 0)),
        "node_type_count": int(len(graph_metadata.get("node_counts", {}))),
        "forward_relation_count": int(len(graph_metadata.get("global_relation_stats", {}))),
        "relation_count_with_reverse": int(len(graph_metadata.get("relation_mappings", []))),
        "split_counts": dict(graph_metadata.get("case_split_counts", {})),
        "raw_label_distribution_after_filtering": dict(graph_metadata.get("raw_label_distribution_after_filtering", {})),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "num_layers": int(model_cfg.get("num_layers", 2)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
        "mlp_hidden_dim": int(model_cfg.get("mlp_hidden_dim", 128)),
        "architecture": str(model_cfg.get("architecture", "hgt")).lower(),
        "text_encoder_backend": str(features_cfg.get("text_encoder", {}).get("backend", "")),
        "text_encoder_model_name": str(features_cfg.get("text_encoder", {}).get("model_name", "")),
    }


def _setup_canvas(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    body_face: str = "#f8fafc",
    title_size: float = 13.0,
    body_size: float = 9.2,
    title_height: float = 0.62,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=1.3,
            edgecolor=color,
            facecolor=body_face,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - title_height),
            w,
            title_height,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            linewidth=0.0,
            facecolor=color,
        )
    )
    ax.text(
        x + 0.18,
        y + h - title_height / 2.0,
        title,
        va="center",
        ha="left",
        fontsize=title_size,
        color="white",
        fontweight="bold",
    )
    ax.text(
        x + 0.18,
        y + h - title_height - 0.18,
        body,
        va="top",
        ha="left",
        fontsize=body_size,
        color="#0f172a",
        linespacing=1.18,
    )


def _note(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str = "#475569",
    face: str = "#f8fafc",
    size: float = 8.9,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.0,
            edgecolor=color,
            facecolor=face,
        )
    )
    ax.text(
        x + 0.16,
        y + h - 0.16,
        text,
        va="top",
        ha="left",
        fontsize=size,
        color="#0f172a",
        linespacing=1.16,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    style: str = "-|>",
    lw: float = 1.8,
    mutation_scale: float = 15.0,
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            linestyle=linestyle,
            connectionstyle=connectionstyle,
        )
    )


def _save(fig: plt.Figure, output_base: Path) -> None:
    ensure_dir(output_base.parent)
    fig.savefig(output_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_pipeline_flow_figure(output_base: Path) -> None:
    snapshot = _runtime_snapshot()
    fig, ax = _setup_canvas(32.0, 18.0)

    ax.text(
        0.5,
        17.65,
        "Cross-Repository Pipeline Flow: Fixed_GPU_OpenNyai preprocessing -> labelled augmented JSON -> section_GNN graph build -> training",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
        color="#0f172a",
    )
    ax.text(
        0.5,
        17.15,
        "This is the actual code path across both repositories. The left half is created in Fixed_GPU_OpenNyai; the right half is created in section_GNN.",
        fontsize=10.1,
        ha="left",
        va="top",
        color="#475569",
    )

    _box(
        ax,
        0.6,
        12.9,
        4.3,
        3.5,
        "0. Source documents",
        "Input corpus for the upstream repo:\n"
        "- PDF files can be converted by 01_extract_pdf_text.py\n"
        "- or raw .txt judgments can be read directly\n\n"
        "Fixed_GPU_OpenNyai ultimately feeds .txt judgments into OpenNyAI.",
        color="#7c3aed",
        body_face="#f5f3ff",
        body_size=8.5,
    )

    _box(
        ax,
        5.3,
        12.9,
        4.0,
        3.5,
        "1. 01_extract_pdf_text.py",
        "Optional upstream preprocessing:\n"
        "- read PDFs\n"
        "- extract text\n"
        "- write .txt files into an input folder\n\n"
        "This stage exists only when the corpus starts from PDFs.",
        color="#6d28d9",
        body_face="#faf5ff",
        body_size=8.4,
    )

    _box(
        ax,
        9.8,
        12.3,
        6.2,
        4.1,
        "2. run_pipeline.py -> src/pipeline_runner.py",
        "Main OpenNyAI orchestration:\n"
        "- discover_text_files()\n"
        "- prepare_documents()\n"
        "- import OpenNyAI Data + Pipeline\n"
        "- construct Pipeline(components=['NER','Rhetorical_Role','Summarizer'])\n"
        "- run batch inference, GPU-aware if requested\n"
        "- persist outputs per document",
        color="#ea580c",
        body_face="#fff7ed",
        body_size=8.3,
    )

    _box(
        ax,
        16.5,
        12.3,
        5.4,
        4.1,
        "3. Fixed_GPU_OpenNyai outputs/",
        "Written by output_formatter.py:\n"
        "- combined/<file_id>.json\n"
        "- annotations/<file_id>.json\n"
        "- ner/<file_id>.json\n"
        "- rhetorical_roles/<file_id>.json\n"
        "- summaries/<file_id>.json and .txt\n"
        "- logs/run_report.json",
        color="#ca8a04",
        body_face="#fefce8",
        body_size=8.4,
    )

    _box(
        ax,
        22.5,
        12.3,
        5.8,
        4.1,
        "4. add_case_outcome_labels_mistral.py",
        "Reads combined JSONs only.\n"
        "collect_input_texts() extracts:\n"
        "- raw_result.summary.decision\n"
        "- all annotation texts whose labels include RPC\n\n"
        "LLM classifier assigns one of:\n"
        "appellant_won / postponed_or_procedural / appellant_lost",
        color="#15803d",
        body_face="#f0fdf4",
        body_size=8.15,
    )

    _note(
        ax,
        22.8,
        9.5,
        5.2,
        2.3,
        "The labeller writes top-level fields such as case_outcome_label and case_outcome_score, plus llm_case_outcome with decision_text, rpc_texts, confidence, explanation, and raw_model_response.",
        color="#166534",
        face="#f0fdf4",
        size=8.0,
    )

    _box(
        ax,
        0.6,
        7.5,
        6.2,
        4.1,
        "5. section_GNN/scripts/preprocess_cases.py",
        "Reads labelled augmented_jsons/*.json.\n"
        "For each file:\n"
        "- load_case_json()\n"
        "- clean_case_for_gnn()\n"
        "- dump cleaned case JSON\n"
        "- dump normalized entities JSON\n"
        "- dump leakage audit JSON\n"
        "- append summary entry",
        color="#0f766e",
        body_face="#f0fdfa",
        body_size=8.25,
    )

    _note(
        ax,
        0.8,
        4.9,
        6.0,
        2.2,
        "clean_case_for_gnn() calls detect_leakage_spans(), extract_prejudgment_text(), extract_entities_from_annotations(), and _build_case_metadata(). It keeps only leakage-safe pre-decision text and role-bucketed argument text.",
        color="#0f766e",
        face="#ecfeff",
        size=8.05,
    )

    _box(
        ax,
        7.3,
        7.8,
        4.8,
        3.8,
        "6. Cleaned artifacts",
        "Under section_GNN/data/<exp>/:\n"
        "- processed/cleaned_cases/*.json\n"
        "- processed/normalized_entities/*.json\n"
        "- audits/*.json\n"
        "- processed/preprocess_summary.json",
        color="#0891b2",
        body_face="#ecfeff",
        body_size=8.45,
    )

    _box(
        ax,
        12.6,
        7.5,
        6.3,
        4.1,
        "7. section_GNN/scripts/build_graph.py",
        "Loads cleaned cases, then build_graph_bundle() performs:\n"
        "- assert_cleaned_case_integrity()\n"
        "- prepare_cases_for_task()\n"
        "- build_case_star_graph() per retained case\n"
        "- merge_case_graphs_into_global_graph()\n"
        "- build_split_assignments()\n"
        "- build_pyg_heterodata()",
        color="#2563eb",
        body_face="#eff6ff",
        body_size=8.15,
    )

    _note(
        ax,
        12.8,
        4.9,
        6.0,
        2.2,
        "Key graph semantics: label filtering happens before local graph build; shareable nodes merge globally by canonical node_key; ToUndirected() duplicates forward relations into reverse relations so case nodes can receive messages back from their context nodes.",
        color="#1d4ed8",
        face="#eff6ff",
        size=8.0,
    )

    _box(
        ax,
        19.4,
        7.8,
        4.7,
        3.8,
        "8. Graph bundle + caches",
        "Under graph_cache/ and embeddings_cache/:\n"
        "- <cache_name>.pt\n"
        "- graph_metadata.json\n"
        "- node_mappings.json\n"
        "- relation_mappings.json\n"
        "- split_assignments.json\n"
        "- graph_debug_samples.json\n"
        "- graph_config_snapshot.yaml\n"
        "- *.npz embedding caches",
        color="#1d4ed8",
        body_face="#eff6ff",
        body_size=8.0,
    )

    _box(
        ax,
        24.6,
        7.5,
        6.1,
        4.1,
        "9. section_GNN/scripts/train_gnn.py",
        "Loads the cached bundle and runs train_model():\n"
        "- instantiate HeteroLegalOutcomeGNN\n"
        "- full-graph forward pass on one device\n"
        "- loss only on logits[train_mask]\n"
        "- validation selection by macro F1\n"
        "- save best checkpoint and plots",
        color="#be185d",
        body_face="#fdf2f8",
        body_size=8.2,
    )

    _box(
        ax,
        19.4,
        1.1,
        11.3,
        3.0,
        "10. section_GNN outputs and side branches",
        "Main training outputs:\n"
        "- outputs/<exp>/logs/{preprocess_cases,build_graph,train_gnn}.log\n"
        "- outputs/<exp>/models/<run_name>/{model.pt,metrics.json,predictions.csv,history.csv,plots}\n\n"
        "Side branches:\n"
        "- visualize_graph.py / generate_final_visualisations.py\n"
        "- export_training_graph_visualiser.py / export_visualiser_catalog.py",
        color="#7c2d12",
        body_face="#fff7ed",
        body_size=8.15,
    )

    _note(
        ax,
        7.3,
        1.1,
        11.3,
        3.0,
        "Current checked augmented_jsons_3way snapshot:\n"
        f"- input_dir = {snapshot['current_raw_json_dir']}\n"
        f"- {snapshot['num_input_files']} raw JSONs -> {snapshot['num_cleaned_cases']} cleaned cases -> train/val/test = {snapshot['split_counts'].get('train', 0)}/{snapshot['split_counts'].get('val', 0)}/{snapshot['split_counts'].get('test', 0)}\n"
        f"- label counts after filtering = {snapshot['raw_label_distribution_after_filtering']}",
        color="#334155",
        face="#f8fafc",
        size=8.0,
    )

    _note(
        ax,
        29.0,
        12.3,
        2.3,
        4.1,
        "Canonical base-config handoff path:\n"
        f"{snapshot['base_raw_json_dir']}\n\n"
        "Current 3-way run uses the same augmented-json schema but from a section_GNN-local copy of that dataset.",
        color="#475569",
        face="#f8fafc",
        size=7.6,
    )

    _arrow(ax, (4.9, 14.65), (5.3, 14.65), color="#7c3aed")
    _arrow(ax, (9.3, 14.65), (9.8, 14.65), color="#6d28d9")
    _arrow(ax, (16.0, 14.35), (16.5, 14.35), color="#ea580c")
    _arrow(ax, (21.9, 14.35), (22.5, 14.35), color="#ca8a04")
    _arrow(ax, (25.4, 12.3), (25.4, 11.85), color="#15803d")
    _arrow(ax, (22.5, 10.2), (6.8, 10.2), color="#334155", connectionstyle="arc3,rad=0.0")
    _arrow(ax, (6.8, 10.2), (6.8, 9.55), color="#334155")
    _arrow(ax, (6.8, 9.55), (7.3, 9.55), color="#0f766e")
    _arrow(ax, (12.1, 9.55), (12.6, 9.55), color="#0891b2")
    _arrow(ax, (18.9, 9.55), (19.4, 9.55), color="#2563eb")
    _arrow(ax, (24.1, 9.55), (24.6, 9.55), color="#1d4ed8")
    _arrow(ax, (27.5, 7.5), (27.5, 4.15), color="#be185d")
    _arrow(ax, (15.8, 7.5), (12.8, 4.1), color="#64748b", connectionstyle="arc3,rad=0.15", linestyle="--")
    _arrow(ax, (21.6, 7.8), (18.6, 4.1), color="#64748b", connectionstyle="arc3,rad=-0.12", linestyle="--")

    _save(fig, output_base)


def save_gnn_architecture_figure(output_base: Path) -> None:
    snapshot = _runtime_snapshot()
    fig, ax = _setup_canvas(32.0, 18.0)

    base_text_nodes = ", ".join(BASE_TEXT_NODE_TYPES)
    role_text_nodes = ", ".join(ARGUMENT_ROLE_NODE_TYPES)
    entity_nodes = ", ".join(ENTITY_NODE_TYPES)

    ax.text(
        0.5,
        17.65,
        "Exact HeteroLegalOutcomeGNN Architecture Used in section_GNN",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
        color="#0f172a",
    )
    ax.text(
        0.5,
        17.15,
        "This figure follows src/models/hetero_gnn.py, src/models/mlp_head.py, src/graph/pyg_builder.py, and the installed PyG HGTConv implementation that the code calls.",
        fontsize=10.1,
        ha="left",
        va="top",
        color="#475569",
    )

    _box(
        ax,
        0.6,
        12.8,
        5.7,
        4.0,
        "A. Runtime snapshot of the current checked run",
        "Current augmented_jsons_3way run:\n"
        f"- architecture = {snapshot['architecture']}\n"
        f"- hidden_dim = {snapshot['hidden_dim']}\n"
        f"- num_layers = {snapshot['num_layers']}\n"
        f"- num_heads = {snapshot['num_heads']}\n"
        f"- dropout = {snapshot['dropout']}\n"
        f"- mlp_hidden_dim = {snapshot['mlp_hidden_dim']}\n"
        f"- labels = {snapshot['label_names']}",
        color="#0f766e",
        body_face="#f0fdfa",
        body_size=8.45,
    )

    _box(
        ax,
        0.6,
        8.2,
        5.7,
        4.0,
        "B. Active graph ontology reaching the model",
        "Current graph cache contains:\n"
        f"- {snapshot['node_type_count']} active node types\n"
        f"- {snapshot['forward_relation_count']} forward relation families\n"
        f"- {snapshot['relation_count_with_reverse']} edge types after ToUndirected\n\n"
        "These reverse relations are essential: the case node updates by receiving messages back from text/entity nodes over rev_* relations.",
        color="#9333ea",
        body_face="#faf5ff",
        body_size=8.35,
    )

    _note(
        ax,
        0.6,
        3.2,
        5.7,
        4.3,
        "Node families encoded into x_dict:\n"
        f"- root: case\n"
        f"- base text: {base_text_nodes}\n"
        f"- role text: {role_text_nodes}\n"
        f"- entities: {entity_nodes}\n\n"
        f"RELATION_DEFINITIONS declares {len(RELATION_DEFINITIONS)} forward relation types; the current run activates all of them.",
        color="#7e22ce",
        face="#faf5ff",
        size=8.0,
    )

    _note(
        ax,
        0.6,
        0.7,
        5.7,
        2.0,
        "Current node features:\n"
        f"- text encoder = {snapshot['text_encoder_backend']} ({snapshot['text_encoder_model_name']})\n"
        f"- embedding_dim = {snapshot['embedding_dim']}, scalar_dim = {snapshot['scalar_dim']}, total feature_dim = {snapshot['feature_dim']}",
        color="#7e22ce",
        face="#faf5ff",
        size=8.2,
    )

    _box(
        ax,
        6.9,
        12.2,
        5.6,
        4.6,
        "C. Node feature construction before the GNN",
        "build_pyg_heterodata() creates one dense feature vector per node:\n"
        "- case nodes = text embedding(case_text_sections) + case scalar vector\n"
        "- text nodes = text embedding(section text) + text scalar vector\n"
        "- entity nodes = text embedding(canonical_name) + entity scalar vector\n\n"
        "No edge_attr is created. Edges contribute only connectivity and typed relation identity.",
        color="#ea580c",
        body_face="#fff7ed",
        body_size=8.2,
    )

    _box(
        ax,
        6.9,
        6.9,
        5.6,
        4.7,
        "D. Typed input encoder in HeteroLegalOutcomeGNN",
        "For every node type t:\n"
        "- input_projections[t] = Linear(input_dim_t, hidden_dim)\n"
        "- type_embeddings[t] = learned Parameter(1, hidden_dim)\n\n"
        "Initial state:\n"
        "h_t^0 = input_projections[t](x_t) + type_embeddings[t]\n\n"
        "Interpretation in this codebase:\n"
        "- L0 is type-aware but not graph-aware yet.",
        color="#c2410c",
        body_face="#fff7ed",
        body_size=8.25,
    )

    _box(
        ax,
        13.1,
        10.2,
        9.0,
        6.6,
        "E. One HGT message-passing layer as actually invoked",
        "Configured in src/models/hetero_gnn.py as:\n"
        f"HGTConv(in_channels={snapshot['hidden_dim']}, out_channels={snapshot['hidden_dim']}, heads={snapshot['num_heads']}, metadata=data.metadata())\n\n"
        "Inside PyG HGTConv:\n"
        "1) HeteroDictLinear creates K, Q, V for every node type.\n"
        "2) k_rel and v_rel apply relation-specific transforms to source K/V using (edge_type, head).\n"
        "3) p_rel stores one learned prior per edge type per head.\n"
        "4) Attention uses alpha_ij = softmax(((q_i * k_j).sum * p_rel) / sqrt(D)).\n"
        "5) Weighted V messages are summed to each destination node type.\n"
        "6) out_lin maps outputs per target node type, then HGTConv applies its own skip mix when dimensions match.",
        color="#15803d",
        body_face="#f0fdf4",
        body_size=8.0,
    )

    _box(
        ax,
        13.1,
        5.3,
        9.0,
        4.2,
        "F. Extra wrapper logic added by this repository around every conv layer",
        "After conv(hidden, edge_index_dict), the repo applies per node type:\n"
        "1) residual = hidden[node_type]\n"
        "2) message = conv_out.get(node_type, residual)\n"
        "3) message = dropout(message, p=dropout)\n"
        "4) updated = LayerNorm(residual + message)\n"
        "5) updated = ReLU(updated)\n\n"
        "Important nuance: in HGT mode, there is an internal skip inside HGTConv and an external residual in this wrapper.",
        color="#166534",
        body_face="#f0fdf4",
        body_size=8.05,
    )

    _box(
        ax,
        13.1,
        0.8,
        9.0,
        3.8,
        "G. What layers 1, 2, and 3 mean in this graph",
        "Layer 1:\n"
        "- case reaches its direct reverse-edge neighbors: text sections, parties, court, bench, lawyers, org/gpe/date/case_number.\n\n"
        "Layer 2:\n"
        "- case can incorporate arguments-linked authorities and globally shared nodes reached through the first-hop context.\n\n"
        "Layer 3:\n"
        "- deeper mixed context propagates through the merged graph before readout. There is still no graph-level pooling; only node states are updated.",
        color="#166534",
        body_face="#f0fdf4",
        body_size=8.0,
    )

    _box(
        ax,
        23.0,
        12.1,
        8.2,
        4.7,
        "H. Readout and classifier",
        "Only hidden['case'] is classified.\n\n"
        "MLPHead architecture:\n"
        f"- Linear({snapshot['hidden_dim']} -> {snapshot['mlp_hidden_dim']})\n"
        "- ReLU\n"
        f"- Dropout({snapshot['dropout']})\n"
        f"- Linear({snapshot['mlp_hidden_dim']} -> C)\n\n"
        "C = number of labels from metadata.label_names",
        color="#2563eb",
        body_face="#eff6ff",
        body_size=8.35,
    )

    _box(
        ax,
        23.0,
        6.3,
        8.2,
        4.9,
        "I. Loss, splitting, and evaluation path",
        "train_model() keeps the whole graph on one device.\n"
        "Forward returns logits for all case nodes and hidden states for all node types.\n\n"
        "Loss:\n"
        "cross_entropy(logits[train_mask], y_case[train_mask], weight=class_weights)\n\n"
        "Best epoch:\n"
        "- selected by validation macro F1\n"
        "- repeated-run tie break in train_gnn.py = validation accuracy",
        color="#1d4ed8",
        body_face="#eff6ff",
        body_size=8.05,
    )

    _box(
        ax,
        23.0,
        0.9,
        8.2,
        4.4,
        "J. Alternative layer stack if the config changes",
        "If model.architecture is 'heteroconv' or 'rgcn', the repo does not use HGTConv.\n"
        "It instead builds HeteroConv({edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim)}, aggr='sum').\n\n"
        "That means:\n"
        "- one GraphSAGE-style relation module per edge type\n"
        "- neighborhood aggregation inside each SAGEConv\n"
        "- relation outputs summed by HeteroConv\n"
        "- the repo's external residual + LayerNorm + ReLU still stays the same",
        color="#7c2d12",
        body_face="#fff7ed",
        body_size=7.95,
    )

    _arrow(ax, (6.3, 14.8), (6.9, 14.8), color="#0f766e")
    _arrow(ax, (6.3, 9.2), (6.9, 9.2), color="#9333ea")
    _arrow(ax, (12.5, 14.5), (13.1, 14.5), color="#ea580c")
    _arrow(ax, (12.5, 9.2), (13.1, 7.6), color="#c2410c", connectionstyle="arc3,rad=0.08")
    _arrow(ax, (22.1, 13.5), (23.0, 14.0), color="#15803d")
    _arrow(ax, (22.1, 7.4), (23.0, 8.6), color="#166534")
    _arrow(ax, (27.1, 12.1), (27.1, 11.2), color="#2563eb")
    _arrow(ax, (27.1, 6.3), (27.1, 5.3), color="#1d4ed8")
    _arrow(ax, (17.6, 10.2), (17.6, 9.5), color="#15803d")
    _arrow(ax, (17.6, 5.3), (17.6, 4.6), color="#166534")
    _arrow(ax, (4.0, 3.2), (23.0, 2.9), color="#64748b", connectionstyle="arc3,rad=-0.08", linestyle="--")

    _save(fig, output_base)


def main() -> None:
    output_dir = ensure_dir(PROJECT_ROOT / "documentation" / "figures")
    save_pipeline_flow_figure(output_dir / "section_gnn_pipeline_flow")
    save_gnn_architecture_figure(output_dir / "section_gnn_gnn_architecture")
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
