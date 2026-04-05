#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TEXT_NODE_TYPES = ("preamble", "facts", "arguments")
ENTITY_NODE_TYPES = (
    "petitioner",
    "respondent",
    "court",
    "judge",
    "petitioner_lawyer",
    "defence_lawyer",
    "lawyer",
    "statute",
    "provision",
    "precedent",
    "org",
    "gpe",
    "date",
    "case_number",
)

PAGE_SIZE = (8.27, 11.69)
BODY_FONT_SIZE = 8.7
HEADER_FONT_SIZE = 11
TITLE_FONT_SIZE = 22
TITLE_SUB_FONT_SIZE = 12
TEXT_WIDTH = 104
LINES_PER_PAGE = 60


@dataclass
class RuntimeFacts:
    cfg: dict[str, Any]
    bundle: dict[str, Any]
    data: Any
    metadata: dict[str, Any]
    node_counts: dict[str, int]
    edge_counts: dict[tuple[str, str, str], int]
    forward_edge_counts: dict[str, int]
    total_nodes: int
    total_edges: int
    forward_edges: int
    feature_dim: int
    embedding_dim: int
    scalar_dim: int
    approx_node_feature_mb: float
    approx_edge_index_mb: float
    entity_nodes: int
    text_nodes: int
    party_feature_mb: float
    duplicate_entity_slot_mb: float
    dead_text_tail_mb: float
    case_embedding_duplication_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detailed architecture and scaling PDFs for the current GNN.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "gnn_case_star.yaml"),
    )
    parser.add_argument(
        "--graph-cache",
        default=str(PROJECT_ROOT / "data" / "graph_cache" / "case_star_global_graph.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "document"),
    )
    return parser.parse_args()


def load_runtime_facts(config_path: str | Path, graph_cache_path: str | Path) -> RuntimeFacts:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    bundle = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    data = bundle["data"]
    metadata = bundle["metadata"]
    node_counts = {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}
    edge_counts = {edge_type: int(data[edge_type].edge_index.shape[1]) for edge_type in data.edge_types}
    forward_edge_counts = {
        "|".join(edge_type): count for edge_type, count in edge_counts.items() if not edge_type[1].startswith("rev_")
    }

    total_nodes = sum(node_counts.values())
    total_edges = sum(edge_counts.values())
    forward_edges = sum(forward_edge_counts.values())
    feature_dim = int(metadata["feature_dim"])
    embedding_dim = int(metadata["embedding_dim"])
    scalar_dim = int(metadata["scalar_dim"])
    approx_node_feature_mb = total_nodes * feature_dim * 4.0 / 1024.0 / 1024.0
    approx_edge_index_mb = total_edges * 2.0 * 8.0 / 1024.0 / 1024.0
    entity_nodes = sum(node_counts.get(node_type, 0) for node_type in ENTITY_NODE_TYPES)
    text_nodes = sum(node_counts.get(node_type, 0) for node_type in TEXT_NODE_TYPES)
    party_feature_mb = sum(node_counts.get(node_type, 0) for node_type in ("petitioner", "respondent")) * feature_dim * 4.0 / 1024.0 / 1024.0
    duplicate_entity_slot_mb = entity_nodes * 4.0 / 1024.0 / 1024.0
    dead_text_tail_mb = text_nodes * 8.0 * 4.0 / 1024.0 / 1024.0
    case_embedding_duplication_mb = node_counts.get("case", 0) * embedding_dim * 4.0 / 1024.0 / 1024.0

    return RuntimeFacts(
        cfg=cfg,
        bundle=bundle,
        data=data,
        metadata=metadata,
        node_counts=node_counts,
        edge_counts=edge_counts,
        forward_edge_counts=forward_edge_counts,
        total_nodes=total_nodes,
        total_edges=total_edges,
        forward_edges=forward_edges,
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        scalar_dim=scalar_dim,
        approx_node_feature_mb=approx_node_feature_mb,
        approx_edge_index_mb=approx_edge_index_mb,
        entity_nodes=entity_nodes,
        text_nodes=text_nodes,
        party_feature_mb=party_feature_mb,
        duplicate_entity_slot_mb=duplicate_entity_slot_mb,
        dead_text_tail_mb=dead_text_tail_mb,
        case_embedding_duplication_mb=case_embedding_duplication_mb,
    )


def wrap_paragraph(text: str, width: int = TEXT_WIDTH) -> list[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=width) or [""]


def add_heading(lines: list[str], text: str, level: int = 1) -> None:
    if lines and lines[-1] != "":
        lines.append("")
    marker = "=" if level == 1 else "-"
    lines.append(text)
    lines.append(marker * len(text))
    lines.append("")


def add_paragraph(lines: list[str], text: str) -> None:
    lines.extend(wrap_paragraph(text))
    lines.append("")


def add_bullets(lines: list[str], items: list[str]) -> None:
    for item in items:
        wrapped = textwrap.wrap(item, width=TEXT_WIDTH, initial_indent="- ", subsequent_indent="  ")
        lines.extend(wrapped or ["-"])
    lines.append("")


def table_lines(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    fmt = " | ".join("{:<" + str(width) + "}" for width in widths)
    lines = [fmt.format(*headers), "-+-".join("-" * width for width in widths)]
    for row in rows:
        lines.append(fmt.format(*row))
    return lines


def build_node_scope_rows(facts: RuntimeFacts) -> list[list[str]]:
    shareable = set(facts.cfg.get("graph", {}).get("shareable_node_types", []))
    rows: list[list[str]] = []
    ordered = [
        "case",
        "preamble",
        "facts",
        "arguments",
        "petitioner",
        "respondent",
        "court",
        "judge",
        "petitioner_lawyer",
        "defence_lawyer",
        "lawyer",
        "statute",
        "provision",
        "precedent",
        "org",
        "gpe",
        "date",
        "case_number",
    ]
    for node_type in ordered:
        if node_type == "case":
            scope = "case-local"
            source = "cleaned preamble+facts+arguments concat"
        elif node_type in TEXT_NODE_TYPES:
            scope = "case-local"
            source = f"cleaned {node_type} text when present"
        elif node_type in {"petitioner", "respondent"}:
            scope = "case-local"
            source = "normalized party string"
        else:
            scope = "shared-global" if node_type in shareable else "case-local"
            source = "normalized canonical string"
        rows.append([node_type, scope, str(facts.node_counts.get(node_type, 0)), source])
    return rows


def build_forward_relation_rows(facts: RuntimeFacts) -> list[list[str]]:
    descriptions = {
        "case|has_preamble|preamble": "case to preamble section",
        "case|has_facts|facts": "case to facts section, only when facts text exists",
        "case|has_arguments|arguments": "case to arguments section, only when arguments text exists",
        "case|has_petitioner|petitioner": "case to case-local petitioner entity",
        "case|has_respondent|respondent": "case to case-local respondent entity",
        "case|heard_in|court": "case to normalized court node",
        "case|decided_by_bench|judge": "case to normalized judge node",
        "case|has_lawyer|lawyer": "case to generic lawyer node",
        "case|has_petitioner_lawyer|petitioner_lawyer": "case to petitioner-side lawyer node",
        "case|has_defence_lawyer|defence_lawyer": "case to defence-side lawyer node",
        "case|mentions_org|org": "case to organization mention node",
        "case|mentions_gpe|gpe": "case to location mention node",
        "case|has_date|date": "case to date mention node",
        "case|has_case_number|case_number": "case to case-number mention node",
        "arguments|cites_statute|statute": "arguments section to cited statute",
        "arguments|cites_provision|provision": "arguments section to cited provision",
        "arguments|cites_precedent|precedent": "arguments section to cited precedent",
        "provision|belongs_to_statute|statute": "legal hierarchy link",
        "petitioner_lawyer|citation|arguments": "petitioner-side lawyer connected to arguments node",
        "defence_lawyer|citation|arguments": "defence-side lawyer connected to arguments node",
        "provision|used_in_arguments|arguments": "shortcut bridge back into arguments",
        "statute|used_in_arguments|arguments": "shortcut bridge back into arguments",
        "petitioner|is_party_in_arguments|arguments": "party bridge into arguments",
        "respondent|is_party_in_arguments|arguments": "party bridge into arguments",
        "judge|presided_arguments|arguments": "judge bridge into arguments",
    }
    ordered = [
        "case|has_preamble|preamble",
        "case|has_facts|facts",
        "case|has_arguments|arguments",
        "case|has_petitioner|petitioner",
        "case|has_respondent|respondent",
        "case|heard_in|court",
        "case|decided_by_bench|judge",
        "case|has_lawyer|lawyer",
        "case|has_petitioner_lawyer|petitioner_lawyer",
        "case|has_defence_lawyer|defence_lawyer",
        "case|mentions_org|org",
        "case|mentions_gpe|gpe",
        "case|has_date|date",
        "case|has_case_number|case_number",
        "arguments|cites_statute|statute",
        "arguments|cites_provision|provision",
        "arguments|cites_precedent|precedent",
        "provision|belongs_to_statute|statute",
        "petitioner_lawyer|citation|arguments",
        "defence_lawyer|citation|arguments",
        "provision|used_in_arguments|arguments",
        "statute|used_in_arguments|arguments",
        "petitioner|is_party_in_arguments|arguments",
        "respondent|is_party_in_arguments|arguments",
        "judge|presided_arguments|arguments",
    ]
    rows: list[list[str]] = []
    for key in ordered:
        src, relation, dst = key.split("|")
        rows.append([src, relation, dst, str(facts.forward_edge_counts.get(key, 0)), descriptions[key]])
    return rows


def build_architecture_lines(facts: RuntimeFacts) -> list[str]:
    cfg = facts.cfg
    model_cfg = cfg.get("model", {})
    features_cfg = cfg.get("features", {})
    split_counts = facts.metadata.get("case_split_counts", {})
    case_scalar_names = list(features_cfg.get("case_scalar_names", []))
    entity_scalar_names = list(features_cfg.get("entity_scalar_names", []))
    text_scalar_names = [
        "text_length",
        "is_preamble",
        "is_facts",
        "is_arguments",
        "cited_statute_count",
        "cited_provision_count",
        "cited_precedent_count",
        "petitioner_lawyer_count",
        "defence_lawyer_count",
        "petitioner_count",
        "respondent_count",
        "judge_count",
    ]

    lines: list[str] = []
    add_heading(lines, "Exact GNN Architecture", level=1)
    add_paragraph(
        lines,
        "This document is generated from the current implementation and cached graph, not from the old phase-1 note. "
        "Source of truth files inspected: configs/gnn_case_star.yaml, data/graph_cache/case_star_global_graph.pt, "
        "src/preprocessing/extract.py, src/graph/case_star_builder.py, src/graph/global_graph_builder.py, "
        "src/graph/pyg_builder.py, src/models/hetero_gnn.py, and src/training/train.py."
    )
    add_bullets(
        lines,
        [
            f"Generated UTC timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}",
            f"Model in code/config: architecture={model_cfg.get('architecture')}, num_layers={model_cfg.get('num_layers')}, hidden_dim={model_cfg.get('hidden_dim')}, num_heads={model_cfg.get('num_heads')}, dropout={model_cfg.get('dropout')}, mlp_hidden_dim={model_cfg.get('mlp_hidden_dim')}",
            f"Graph snapshot in cache: {facts.node_counts.get('case', 0)} case nodes, {facts.total_nodes} total nodes, {facts.forward_edges} forward edges, {facts.total_edges} directed edges after reverse-edge expansion, {len(facts.data.node_types)} node types, {len(facts.data.edge_types)} edge types",
            f"Feature size: {facts.feature_dim} = {facts.embedding_dim} text-embedding dimensions + {facts.scalar_dim} scalar dimensions",
            f"Split counts on case nodes: train={split_counts.get('train', 0)}, val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}",
            "Important correction: the README still says 2 message-passing layers, but the current config and model code use 3 HGT layers."
        ],
    )

    add_heading(lines, "1. Build Stages", level=2)
    add_bullets(
        lines,
        [
            "Stage A - clean each raw case: leakage-bearing fields are removed, decision-like annotations are dropped, and pre-judgment text is assembled into retained sections preamble/facts/arguments.",
            "Stage B - extract normalized entities: petitioner/respondent/court/judge/lawyer/statute/provision/precedent/org/gpe/date/case_number are read from retained annotations; lawyer mentions are refined into petitioner_lawyer or defence_lawyer when the local text window suggests a side.",
            "Stage C - build one local case star graph: one case node is created first, then section nodes and entity nodes are attached by typed relations.",
            "Stage D - merge all local graphs into one global graph: shareable node types are merged by node_key across cases; local-only types keep case-specific keys.",
            "Stage E - tensorize into PyTorch Geometric HeteroData: every node gets a dense feature vector x, every relation gets an edge_index tensor, and only case nodes receive y/train_mask/val_mask/test_mask.",
            "Stage F - make the graph bidirectional: ToUndirected() adds a reverse relation for every forward relation so the case node can aggregate back from its neighbors.",
            "Stage G - run 3 HGT layers and an MLP head: logits are produced only for case nodes, and loss is applied only on training-mask case nodes."
        ],
    )

    add_heading(lines, "2. Node Types, Scope, and Live Counts", level=2)
    lines.extend(table_lines(["node_type", "scope", "count", "node.text stores"], build_node_scope_rows(facts)))
    lines.append("")
    add_paragraph(
        lines,
        "The count column comes from the live cached graph. Preamble exists for every case in the current cache, but facts and arguments are optional in practice and appear only when retained text exists. "
        "Petitioner and respondent are intentionally case-local because share_party_nodes=false in the current config."
    )

    add_heading(lines, "3. What A Node Stores Before And After Tensorization", level=2)
    add_bullets(
        lines,
        [
            "Before PyG tensorization, each node is a GraphNode with fields: node_type, node_key, text, metadata, share_across_cases.",
            "For section nodes, text is the cleaned section string. For entity nodes, text is the normalized canonical name string. For case nodes, text is the concatenation of retained preamble + facts + arguments.",
            "During global merge, merged node metadata gains global_case_frequency and then degree.",
            "After build_pyg_heterodata(), the model-facing HeteroData stores data[node_type].x and data[node_type].node_id for all node types. Only the case store additionally gets y, train_mask, val_mask, test_mask, case_id, file_name, and raw_label.",
            "The final HeteroData does not carry raw text strings for every node. The raw cleaned texts remain on disk in data/processed/cleaned_cases/*.json; the graph tensors contain numeric features only."
        ],
    )

    add_heading(lines, "4. Exact Case Node Feature Layout", level=2)
    case_rows = [[f"{idx}", name] for idx, name in enumerate(case_scalar_names, start=facts.embedding_dim)]
    lines.extend(table_lines(["slot", "meaning"], [["0-383", "SentenceTransformer embedding of cleaned case text (preamble + facts + arguments)"]] + case_rows))
    lines.append("")
    add_paragraph(
        lines,
        "The scalar tail on the case node is filled from preprocessing metadata. Concretely this gives the model both a dense semantic summary of the full retained text and a compact numeric summary of selected case-level counts and lengths."
    )

    add_heading(lines, "5. Exact Text Node Feature Layout", level=2)
    text_rows = [["0-383", "SentenceTransformer embedding of the section text"]]
    for idx, name in enumerate(text_scalar_names, start=facts.embedding_dim):
        text_rows.append([str(idx), name])
    lines.extend(table_lines(["slot", "meaning"], text_rows))
    lines.append("")
    add_bullets(
        lines,
        [
            "Observed non-zero scalar columns in the current cache: preamble uses only slots 384 and 385; facts uses only 384 and 386; arguments uses only 384 and 387.",
            "Slots 388 through 395 exist in the feature vector but are all zero right now because the builder initializes those metadata fields to 0 and never updates them later.",
            "Because the model is already heterogeneous, the one-hot type slots is_preamble/is_facts/is_arguments duplicate information that is also available from the node type itself."
        ],
    )

    add_heading(lines, "6. Exact Entity Node Feature Layout", level=2)
    entity_rows = [["0-383", "SentenceTransformer embedding of the canonical entity string"]]
    for idx, name in enumerate(entity_scalar_names, start=facts.embedding_dim):
        entity_rows.append([str(idx), name])
    entity_rows.extend([[str(facts.embedding_dim + 10), "padding zero"], [str(facts.embedding_dim + 11), "padding zero"]])
    lines.extend(table_lines(["slot", "meaning"], entity_rows))
    lines.append("")
    add_bullets(
        lines,
        [
            "global_case_frequency is written only after the cross-case merge, so it describes how many distinct cases use the merged node.",
            "degree is computed from the merged graph before PyG conversion and then scaled into the entity scalar tail.",
            "In the current implementation, slot 384 mention_count and slot 390 local_case_frequency are populated from the same metadata field, so they are exact duplicates.",
            "Case-local petitioner/respondent nodes keep is_shared_node=0. Shared node types such as court/statute/gpe can still have global_case_frequency=1 if they ended up unique in the current corpus."
        ],
    )

    add_heading(lines, "7. Exact Edge Storage", level=2)
    add_bullets(
        lines,
        [
            "Edges do not have edge_attr in the current pipeline. The graph stores only connectivity and edge type.",
            "Each relation becomes a separate edge_index tensor of shape [2, E_relation].",
            "After ToUndirected(), every forward relation gets a reverse relation named rev_<relation>.",
            f"The live cached graph therefore has {len(facts.forward_edge_counts)} forward relation types and {len(facts.data.edge_types)} total directed relation types."
        ],
    )

    add_heading(lines, "8. Forward Relations And Live Counts", level=2)
    lines.extend(
        table_lines(
            ["src", "relation", "dst", "count", "meaning"],
            build_forward_relation_rows(facts),
        )
    )
    lines.append("")
    add_paragraph(
        lines,
        "There are no dense case-to-case similarity edges in this implementation. Cross-case signal appears only because shared nodes such as court/judge/statute/provision/precedent/org/gpe/date/case_number are reused across cases."
    )

    add_heading(lines, "9. Exact Layer-By-Layer Computation", level=2)
    add_bullets(
        lines,
        [
            f"Input projection step: for every node type t, x_t with shape [N_t, {facts.feature_dim}] is mapped by a node-type-specific Linear({facts.feature_dim} -> {model_cfg.get('hidden_dim')}). Then a learned type embedding of shape [1, {model_cfg.get('hidden_dim')}] is added.",
            "Layer formula used for each of the 3 message-passing layers: hidden = HGTConv(hidden, edge_index_dict); then for each node type the block does residual + dropout + LayerNorm + ReLU.",
            "More explicitly, per node type: updated_t = ReLU(LayerNorm(residual_t + Dropout(message_t))).",
            "Case node at layer 1: it can aggregate only direct neighbors through reverse relations, so it sees preamble/facts/arguments and the attached entity nodes in their layer-0 state.",
            "Case node at layer 2: this is the first layer where true cross-case information reaches a case node, because shared nodes such as court/statute/provision/judge have already aggregated from all attached cases during layer 1.",
            "Case node at layer 3: it receives neighbor states that already contain two rounds of contextual mixing, so deeper patterns like case <- shared authority <- other case <- that case's local neighborhood can influence the representation.",
            "Examples of meaningful layer-2 and layer-3 routes: case_A <- court <- case_B, case_A <- arguments_A <- statute <- arguments_B, case_A <- judge <- case_B, case_A <- arguments_A <- provision <- arguments_B."
        ],
    )

    add_heading(lines, "10. HGT Block And Classification Head", level=2)
    add_bullets(
        lines,
        [
            f"HGT block configuration: HGTConv(in_channels={model_cfg.get('hidden_dim')}, out_channels={model_cfg.get('hidden_dim')}, heads={model_cfg.get('num_heads')}).",
            "The model is heterogeneous at every layer: relation type and node type both matter, so each relation gets relation-specific attention projections inside HGTConv.",
            f"Classifier head on case nodes only: Linear({model_cfg.get('hidden_dim')} -> {model_cfg.get('mlp_hidden_dim')}) -> ReLU -> Dropout({model_cfg.get('dropout')}) -> Linear({model_cfg.get('mlp_hidden_dim')} -> {len(facts.metadata.get('label_names', []))}).",
            "Output logits are produced for every case node, but loss is computed only on case nodes where train_mask is True."
        ],
    )

    add_heading(lines, "11. Training Procedure", level=2)
    add_bullets(
        lines,
        [
            "Device is cuda if available, otherwise cpu.",
            f"Optimizer: AdamW with lr={facts.cfg.get('training', {}).get('lr')} and weight_decay={facts.cfg.get('training', {}).get('weight_decay')}.",
            "Balanced class weights are computed from only the training case labels when class_weight=balanced.",
            "At each epoch, the model runs on the full graph; only the supervision mask changes which case nodes contribute to loss and which contribute to validation/test metrics.",
            f"Early stopping is enabled with patience={facts.cfg.get('training', {}).get('early_stopping_patience')} over validation macro F1.",
            f"Training budget in config: epochs={facts.cfg.get('training', {}).get('epochs')}, log_every={facts.cfg.get('training', {}).get('log_every')}.",
            "Best checkpoint selection uses validation macro F1; the saved best state is reloaded before final train/val/test evaluation."
        ],
    )

    add_heading(lines, "12. Important Implementation Notes", level=2)
    add_bullets(
        lines,
        [
            "Facts and arguments nodes are optional, not guaranteed. In the current cache there are 795 facts nodes and 709 arguments nodes for 1010 cases.",
            "If a provision mention references a statute but the statute node is otherwise absent, the builder can create a synthetic statute node so belongs_to_statute still has a target.",
            "The current SentenceTransformer backend ignores the max_length value from the config in this code path; truncation behavior is governed by the encoder implementation itself, not by an explicit max_length argument here.",
            "Approximate live tensor footprint from node features alone is about "
            f"{facts.approx_node_feature_mb:.2f} MB; edge_index tensors add about {facts.approx_edge_index_mb:.2f} MB more."
        ],
    )
    return lines


def build_audit_lines(facts: RuntimeFacts) -> list[str]:
    lines: list[str] = []
    add_heading(lines, "Redundancy And Scaling Audit", level=1)
    add_paragraph(
        lines,
        "This audit separates exact duplication from acceptable summarization. Some repeated information is intentional because it shortens message-passing paths; other parts are exact duplicates or currently-dead feature slots that can be removed with no loss of information."
    )
    add_bullets(
        lines,
        [
            f"Current node-feature tensor footprint: about {facts.approx_node_feature_mb:.2f} MB",
            f"Current directed edge_index footprint: about {facts.approx_edge_index_mb:.2f} MB",
            f"Party nodes alone (petitioner + respondent) use about {facts.party_feature_mb:.2f} MB of node-feature storage, which is about {(facts.party_feature_mb / facts.approx_node_feature_mb) * 100.0:.1f}% of all x memory",
            f"Exact duplicate entity scalar slot cost now: about {facts.duplicate_entity_slot_mb:.3f} MB",
            f"Dead text scalar tail cost now: about {facts.dead_text_tail_mb:.3f} MB",
            f"Case-level combined text embedding duplication cost now: about {facts.case_embedding_duplication_mb:.3f} MB"
        ],
    )

    add_heading(lines, "1. High-Impact Scaling Risk: Case-Local Party Nodes", level=2)
    add_bullets(
        lines,
        [
            f"Live counts: petitioner={facts.node_counts.get('petitioner', 0)} and respondent={facts.node_counts.get('respondent', 0)}.",
            "These nodes are not exact duplicates in the strict sense, but they are by far the main storage hotspot because they remain case-local and therefore do not merge across cases.",
            f"Average per case in the live cache: petitioner={facts.node_counts.get('petitioner', 0) / max(facts.node_counts.get('case', 1), 1):.2f}, respondent={facts.node_counts.get('respondent', 0) / max(facts.node_counts.get('case', 1), 1):.2f}.",
            "If your next scaling step is mostly about storage and training throughput, party handling is the first place to simplify."
        ],
    )

    add_heading(lines, "2. Exact Duplicate Feature: mention_count vs local_case_frequency", level=2)
    add_bullets(
        lines,
        [
            "Entity slot 384 stores mention_count / 100.",
            "Entity slot 390 stores local_case_frequency / 100.",
            "In the current implementation both are populated from metadata['mention_count'], so they are numerically identical for every entity node.",
            "This is a true no-information-gain duplicate. Removing one of the two saves one float per entity node with no semantic loss."
        ],
    )

    add_heading(lines, "3. Dead Feature Tail On Text Nodes", level=2)
    add_bullets(
        lines,
        [
            "Text-node slots 388 through 395 are allocated for citation and bridge counts.",
            "The builder initializes cited_statute_count, cited_provision_count, cited_precedent_count, petitioner_lawyer_count, defence_lawyer_count, petitioner_count, respondent_count, and judge_count to 0 on section nodes.",
            "No later step populates those fields, so the current cached graph has those slots equal to 0 for every preamble, facts, and arguments node.",
            "This is dead capacity rather than harmful duplication. It does not change model behavior, but it does make the feature tail wider than necessary."
        ],
    )

    add_heading(lines, "4. Semantic Duplication: Case Text Embedding And Section Text Embeddings", level=2)
    add_bullets(
        lines,
        [
            "Every case node stores one embedding of the concatenated retained text.",
            "The same case can also store up to three section embeddings on preamble/facts/arguments nodes.",
            "This is not an exact tensor duplicate because one vector is a whole-case summary and the others are section-level summaries, but it is still repeated representation of the same underlying text.",
            "Storage impact today is modest compared with party nodes, but if you want a cleaner architecture you should decide whether you want both whole-case and section-level text embeddings or only one of those abstractions."
        ],
    )

    add_heading(lines, "5. Small But Real Redundancy: Text Type Flags", level=2)
    add_bullets(
        lines,
        [
            "Text-node slots 385, 386, and 387 are one-hot flags for preamble/facts/arguments.",
            "The model already knows the node type because the graph is heterogeneous, the input projection is node-type-specific, and HGTConv is relation-aware.",
            "These flags are therefore redundant in a conceptual sense. They are cheap, so they are not your main scaling problem, but they can be removed for a cleaner feature design."
        ],
    )

    add_heading(lines, "6. Not Duplicate, But High-Fanout Context", level=2)
    fanout_rows = [
        ["case|mentions_gpe|gpe", str(facts.forward_edge_counts.get("case|mentions_gpe|gpe", 0))],
        ["case|has_case_number|case_number", str(facts.forward_edge_counts.get("case|has_case_number|case_number", 0))],
        ["case|has_date|date", str(facts.forward_edge_counts.get("case|has_date|date", 0))],
        ["case|mentions_org|org", str(facts.forward_edge_counts.get("case|mentions_org|org", 0))],
    ]
    lines.extend(table_lines(["forward relation", "live edge count"], fanout_rows))
    lines.append("")
    add_bullets(
        lines,
        [
            "These are not duplicates by themselves, but they increase connectivity quickly and can dominate message passing.",
            "If extraction quality on gpe/date/case_number/org is noisy, these relations can add cost without proportional signal.",
            "If you need a lighter graph, pruning or capping these relation families is a much higher-value change than removing a few duplicate scalar slots."
        ],
    )

    add_heading(lines, "7. Recommended Change Order For Scaling", level=2)
    add_bullets(
        lines,
        [
            "Priority 1: compress or prune petitioner/respondent nodes. Options include top-k parties per case, one pooled petitioner node plus one pooled respondent node, or temporarily dropping party nodes in the first large-scale build.",
            "Priority 2: choose between whole-case text embedding and section-level text nodes if you want to reduce semantic duplication. Keeping both is defensible, but it is not minimal.",
            "Priority 3: remove one of mention_count/local_case_frequency on entity nodes. This is a true duplicate and easy to clean up.",
            "Priority 4: either populate the text-node count fields from actual edges or delete those scalar slots from the text tail.",
            "Priority 5: drop the text-node one-hot type flags if you want the feature design to rely purely on heterogeneous node typing.",
            "Priority 6: review gpe/date/case_number/org relation families for extraction noise and cap them if they are mostly surface metadata rather than task signal."
        ],
    )

    add_heading(lines, "8. Practical Bottom Line", level=2)
    add_bullets(
        lines,
        [
            "The graph will not crash because of the exact duplicate scalar slots. Those are small.",
            "The real scaling pressure today comes from graph breadth: very many case-local party nodes and high-fanout context relations.",
            "If you want the cleanest non-redundant design, the first architectural question is not whether to remove one duplicate scalar. It is whether you really need case-local party expansion plus whole-case text embeddings plus section text nodes all at the same time."
        ],
    )
    return lines


def paginate_lines(lines: list[str], max_lines: int = LINES_PER_PAGE) -> list[list[str]]:
    pages: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if len(current) >= max_lines:
            pages.append(current)
            current = []
    if current:
        pages.append(current)
    return pages


def add_text_pages(pdf: PdfPages, document_title: str, subtitle: str, lines: list[str]) -> None:
    pages = paginate_lines(lines)
    total = len(pages)
    for index, page_lines in enumerate(pages, start=1):
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        fig.text(0.06, 0.972, document_title, fontsize=HEADER_FONT_SIZE, fontweight="bold", va="top")
        fig.text(0.06, 0.952, subtitle, fontsize=8.5, color="#475569", va="top")
        fig.text(0.94, 0.972, f"Page {index}/{total}", fontsize=8.5, color="#475569", ha="right", va="top")

        fig.text(
            0.06,
            0.93,
            "\n".join(page_lines),
            fontsize=BODY_FONT_SIZE,
            family="DejaVu Sans Mono",
            va="top",
            ha="left",
            color="#0f172a",
            linespacing=1.25,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_title_page(pdf: PdfPages, title: str, subtitle: str, bullets: list[str]) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(0.06, 0.92, title, fontsize=TITLE_FONT_SIZE, fontweight="bold", va="top", color="#0f172a")
    fig.text(0.06, 0.875, subtitle, fontsize=TITLE_SUB_FONT_SIZE, va="top", color="#334155")

    y = 0.80
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=92)
        if wrapped:
            fig.text(0.08, y, f"- {wrapped[0]}", fontsize=11, va="top", color="#0f172a")
            y -= 0.032
            for continuation in wrapped[1:]:
                fig.text(0.11, y, continuation, fontsize=11, va="top", color="#0f172a")
                y -= 0.032
        else:
            y -= 0.032
        y -= 0.018

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_memory_chart_page(pdf: PdfPages, facts: RuntimeFacts) -> None:
    node_counts = facts.node_counts
    labels = []
    values = []
    for node_type in sorted(node_counts, key=node_counts.get, reverse=True)[:10]:
        labels.append(node_type)
        values.append(node_counts[node_type] * facts.feature_dim * 4.0 / 1024.0 / 1024.0)

    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.barh(labels[::-1], values[::-1], color="#1d4ed8")
    ax.set_xlabel("Approximate MB used by node features x")
    ax.set_title("Largest Node-Feature Memory Consumers")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for idx, value in enumerate(values[::-1]):
        ax.text(value + max(values) * 0.01, idx, f"{value:.2f} MB", va="center", fontsize=9)

    fig.text(
        0.06,
        0.965,
        "Redundancy And Scaling Audit",
        fontsize=HEADER_FONT_SIZE,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.06,
        0.945,
        "Current live graph snapshot from data/graph_cache/case_star_global_graph.pt",
        fontsize=8.5,
        color="#475569",
        va="top",
    )
    plt.tight_layout(rect=[0.06, 0.06, 0.96, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_architecture_pdf(output_path: Path, facts: RuntimeFacts) -> None:
    title_bullets = [
        f"Current implemented model: {facts.cfg.get('model', {}).get('architecture')} with {facts.cfg.get('model', {}).get('num_layers')} HGT layers, hidden_dim={facts.cfg.get('model', {}).get('hidden_dim')}, heads={facts.cfg.get('model', {}).get('num_heads')}",
        f"Live graph snapshot: {facts.node_counts.get('case', 0)} cases, {facts.total_nodes} total nodes, {facts.forward_edges} forward edges, {facts.total_edges} directed edges after reverse relations",
        f"Feature construction: {facts.embedding_dim}-d sentence-transformer embedding plus {facts.scalar_dim} scalar slots per node, total {facts.feature_dim} dims",
        "Purpose of this PDF: explain exactly what every node stores, how the graph is connected, and what happens layer by layer in the current implementation",
    ]
    lines = build_architecture_lines(facts)
    with PdfPages(output_path) as pdf:
        add_title_page(
            pdf,
            title="Exact GNN Architecture",
            subtitle="Case star graph + global authority graph for legal outcome prediction",
            bullets=title_bullets,
        )
        add_text_pages(
            pdf,
            document_title="Exact GNN Architecture",
            subtitle="Generated from current code and cached graph",
            lines=lines,
        )


def generate_audit_pdf(output_path: Path, facts: RuntimeFacts) -> None:
    title_bullets = [
        f"Total node-feature storage in the live graph is about {facts.approx_node_feature_mb:.2f} MB; edge_index tensors add about {facts.approx_edge_index_mb:.2f} MB",
        f"Petitioner + respondent nodes alone account for about {facts.party_feature_mb:.2f} MB of node features",
        "This audit distinguishes exact duplicates from intentional summarization and identifies which changes matter most for scaling",
    ]
    lines = build_audit_lines(facts)
    with PdfPages(output_path) as pdf:
        add_title_page(
            pdf,
            title="Redundancy And Scaling Audit",
            subtitle="Current storage duplication, dead features, and high-impact simplifications",
            bullets=title_bullets,
        )
        add_memory_chart_page(pdf, facts)
        add_text_pages(
            pdf,
            document_title="Redundancy And Scaling Audit",
            subtitle="Generated from current code and cached graph",
            lines=lines,
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    facts = load_runtime_facts(args.config, args.graph_cache)
    architecture_path = output_dir / "01_gnn_exact_architecture.pdf"
    audit_path = output_dir / "02_gnn_redundancy_and_scaling_audit.pdf"

    generate_architecture_pdf(architecture_path, facts)
    generate_audit_pdf(audit_path, facts)

    print(str(architecture_path))
    print(str(audit_path))


if __name__ == "__main__":
    main()
