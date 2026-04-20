from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from src.graph.schema import ARGUMENT_ROLE_NODE_TYPES, BASE_TEXT_NODE_TYPES, ENTITY_NODE_TYPES, TEXT_NODE_TYPES
from src.preprocessing.leakage import compile_leakage_patterns, match_leakage_phrases
from src.utils.text_encoder import load_or_compute_embeddings

# Sections encoded independently for the case node feature.
# Order determines the layout of the concatenated vector: [preamble | facts | arguments].
CASE_SECTION_ORDER = BASE_TEXT_NODE_TYPES  # ("preamble", "facts", "arguments")


def _scaled_length(value: Any) -> float:
    return min(float(value or 0), 5000.0) / 5000.0


def _scaled_count(value: Any) -> float:
    return min(float(value or 0), 100.0) / 100.0


def _scaled_year(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return (float(value) - 1900.0) / 200.0


def _case_scalar_vector(node: dict[str, Any], scalar_dim: int, scalar_names: list[str]) -> np.ndarray:
    metadata = node["metadata"]
    values = {
        "respondent_count": _scaled_count(metadata.get("respondent_count")),
        "judge_count": _scaled_count(metadata.get("judge_count")),
        "lawyer_count": _scaled_count(metadata.get("lawyer_count")),
        "statute_count": _scaled_count(metadata.get("statute_count")),
        "provision_count": _scaled_count(metadata.get("provision_count")),
        "precedent_count": _scaled_count(metadata.get("precedent_count")),
        "preamble_length": _scaled_length(metadata.get("preamble_length")),
        "facts_length": _scaled_length(metadata.get("facts_length")),
        "arguments_length": _scaled_length(metadata.get("arguments_length")),
        "case_year": _scaled_year(metadata.get("case_year")),
        "petition_type_known": 1.0 if metadata.get("petition_type") else 0.0,
        "petition_type_hash": float(metadata.get("petition_type_hash", 0.0)),
    }
    vector = np.zeros((scalar_dim,), dtype=np.float32)
    for idx, name in enumerate(scalar_names):
        if idx >= scalar_dim:
            break
        vector[idx] = float(values.get(name, 0.0))
    return vector


def _entity_scalar_vector(node: dict[str, Any], scalar_dim: int, scalar_names: list[str]) -> np.ndarray:
    metadata = node["metadata"]
    first_seen = str(metadata.get("first_seen_section") or "")
    values = {
        "mention_count": _scaled_count(metadata.get("mention_count")),
        "first_seen_preamble": 1.0 if first_seen == "PREAMBLE" else 0.0,
        "first_seen_facts": 1.0 if first_seen == "facts" else 0.0,
        "first_seen_arguments": 1.0 if first_seen == "arguments" else 0.0,
        "seen_in_arguments": 1.0 if metadata.get("seen_in_arguments") else 0.0,
        "seen_in_preamble": 1.0 if metadata.get("seen_in_preamble") else 0.0,
        "local_case_frequency": _scaled_count(metadata.get("mention_count")),
        "global_case_frequency": _scaled_count(metadata.get("global_case_frequency")),
        "degree": _scaled_count(metadata.get("degree")),
        "is_shared_node": 1.0 if metadata.get("is_shared_node") else 0.0,
    }
    vector = np.zeros((scalar_dim,), dtype=np.float32)
    for idx, name in enumerate(scalar_names):
        if idx >= scalar_dim:
            break
        vector[idx] = float(values.get(name, 0.0))
    return vector


def _text_scalar_vector(node_type: str, node: dict[str, Any], scalar_dim: int) -> np.ndarray:
    meta = node["metadata"]
    is_argument_family = node_type == "arguments" or node_type in ARGUMENT_ROLE_NODE_TYPES
    text_scalar_values = [
        _scaled_length(meta.get("text_length")),
        1.0 if node_type == "preamble" else 0.0,
        1.0 if node_type == "facts" else 0.0,
        1.0 if is_argument_family else 0.0,
        _scaled_count(meta.get("cited_statute_count", 0)),
        _scaled_count(meta.get("cited_provision_count", 0)),
        _scaled_count(meta.get("cited_precedent_count", 0)),
        _scaled_count(meta.get("petitioner_lawyer_count", 0)),
        _scaled_count(meta.get("defence_lawyer_count", 0)),
        _scaled_count(meta.get("petitioner_count", 0)),
        _scaled_count(meta.get("respondent_count", 0)),
        _scaled_count(meta.get("judge_count", 0)),
    ]
    vector = np.zeros((scalar_dim,), dtype=np.float32)
    for idx, val in enumerate(text_scalar_values):
        if idx >= scalar_dim:
            break
        vector[idx] = float(val)
    return vector


def _build_embeddings_section_sep(
    nodes_by_type: dict[str, list[dict[str, Any]]],
    features_cfg: dict[str, Any],
    cache_dir: str,
    logger: Any | None = None,
) -> tuple[dict[tuple[str, str], np.ndarray], int, int, str]:
    """Encode nodes with section-separated case embeddings.

    Non-case nodes: encoded from their text field (as normal).
    Case nodes: preamble, facts, arguments encoded independently, then concatenated.

    Returns:
        embeddings: mapping from (node_type, node_key) -> embedding array
        non_case_dim: embedding dim for non-case node types
        case_text_dim: 3 * non_case_dim (the concatenated case section embedding dim)
        backend_used: encoder backend name
    """
    encoder_cfg = dict(features_cfg.get("text_encoder", {}))
    fallback_cfg = dict(features_cfg.get("fallback_text_encoder", {"backend": "hashing", "dim": 384}))

    # ── Encode non-case nodes normally ────────────────────────────────────────
    non_case_ids: list[str] = []
    non_case_texts: list[str] = []
    non_case_keys: list[tuple[str, str]] = []
    for node_type, nodes in sorted(nodes_by_type.items()):
        if node_type == "case":
            continue
        for node in nodes:
            non_case_keys.append((node_type, node["node_key"]))
            non_case_ids.append(f"{node_type}::{node['node_key']}")
            non_case_texts.append(str(node.get("text", "") or ""))

    backend_used = str(encoder_cfg.get("backend", "sentence_transformers"))
    embeddings: dict[tuple[str, str], np.ndarray] = {}
    non_case_dim = 1024  # default; overwritten below

    if non_case_ids:
        try:
            matrix, non_case_dim = load_or_compute_embeddings(
                ids=non_case_ids,
                texts=non_case_texts,
                encoder_cfg=encoder_cfg,
                cache_dir=cache_dir,
                namespace="section_sep_non_case_nodes",
                logger=logger,
            )
        except Exception as exc:
            if logger is not None:
                logger.warning("Primary encoder failed for non-case nodes (%s). Falling back.", exc)
            matrix, non_case_dim = load_or_compute_embeddings(
                ids=non_case_ids,
                texts=non_case_texts,
                encoder_cfg=fallback_cfg,
                cache_dir=cache_dir,
                namespace="section_sep_non_case_nodes_fallback",
                logger=logger,
            )
            backend_used = str(fallback_cfg.get("backend", "hashing"))
        for idx, key in enumerate(non_case_keys):
            embeddings[key] = matrix[idx]

    # ── Encode case nodes section-by-section ──────────────────────────────────
    case_nodes = nodes_by_type.get("case", [])
    if not case_nodes:
        return embeddings, non_case_dim, 3 * non_case_dim, backend_used

    section_matrices: dict[str, np.ndarray] = {}
    for section in CASE_SECTION_ORDER:
        section_ids = [f"case_sec_{section}::{node['node_key']}" for node in case_nodes]
        section_texts = [
            str(node["metadata"].get("text_sections", {}).get(section, "") or "")
            for node in case_nodes
        ]
        namespace = f"section_sep_case_{section}"
        try:
            sec_matrix, _ = load_or_compute_embeddings(
                ids=section_ids,
                texts=section_texts,
                encoder_cfg=encoder_cfg,
                cache_dir=cache_dir,
                namespace=namespace,
                logger=logger,
            )
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    "Primary encoder failed for case section %s (%s). Falling back.", section, exc
                )
            sec_matrix, _ = load_or_compute_embeddings(
                ids=section_ids,
                texts=section_texts,
                encoder_cfg=fallback_cfg,
                cache_dir=cache_dir,
                namespace=f"{namespace}_fallback",
                logger=logger,
            )
        section_matrices[section] = sec_matrix

    for idx, node in enumerate(case_nodes):
        section_embs = [section_matrices[section][idx] for section in CASE_SECTION_ORDER]
        embeddings[("case", node["node_key"])] = np.concatenate(section_embs, axis=0)

    case_text_dim = len(CASE_SECTION_ORDER) * non_case_dim
    return embeddings, non_case_dim, case_text_dim, backend_used


def run_graph_quality_checks(
    nodes_by_type: dict[str, list[dict[str, Any]]],
    preprocessing_cfg: dict[str, Any],
) -> None:
    patterns = compile_leakage_patterns(list(preprocessing_cfg.get("leakage_phrases", [])))
    forbidden_terms = {"decision", "llm_case_outcome", "case_outcome_label", "rpc"}
    for node_type, nodes in nodes_by_type.items():
        for node in nodes:
            if node_type == "case":
                metadata_keys = set(node["metadata"].keys())
                if metadata_keys.intersection(forbidden_terms):
                    raise AssertionError(
                        f"Forbidden metadata keys on case node {node['node_key']}: "
                        f"{sorted(metadata_keys.intersection(forbidden_terms))}"
                    )
            if node_type in TEXT_NODE_TYPES:
                matches = match_leakage_phrases(str(node.get("text", "")), patterns)
                if matches:
                    raise AssertionError(
                        f"Leakage phrase survived in {node_type} node {node['node_key']}: {matches[:3]}"
                    )


def build_pyg_heterodata_section_sep(
    global_graph: dict[str, Any],
    cleaned_cases: list[Any],
    labels: np.ndarray,
    label_names: list[str],
    split_assignments: dict[str, str],
    cfg: dict[str, Any],
    logger: Any | None = None,
) -> tuple[HeteroData, dict[str, Any]]:
    """Like build_pyg_heterodata but case nodes use section-separated embeddings.

    Case node features: [preamble_emb(D) | facts_emb(D) | args_emb(D) | scalars(S)] = 3D+S dims.
    All other node features: [text_emb(D) | scalars(S)] = D+S dims (unchanged).
    """
    run_graph_quality_checks(global_graph["nodes_by_type"], cfg.get("preprocessing", {}))

    embeddings, non_case_dim, case_text_dim, encoder_backend = _build_embeddings_section_sep(
        nodes_by_type=global_graph["nodes_by_type"],
        features_cfg=cfg.get("features", {}),
        cache_dir=str(cfg.get("paths", {}).get("embeddings_cache_dir")),
        logger=logger,
    )

    case_scalar_names = list(cfg.get("features", {}).get("case_scalar_names", []))
    entity_scalar_names = list(cfg.get("features", {}).get("entity_scalar_names", []))
    scalar_dim = max(len(case_scalar_names), len(entity_scalar_names), 4)

    # Case nodes have wider features than all other node types.
    case_feature_dim = case_text_dim + scalar_dim
    non_case_feature_dim = non_case_dim + scalar_dim

    if logger is not None:
        logger.info(
            "Section-sep encoding | section_order=%s | non_case_feature_dim=%d | case_feature_dim=%d",
            list(CASE_SECTION_ORDER),
            non_case_feature_dim,
            case_feature_dim,
        )

    data = HeteroData()
    node_mappings: dict[str, dict[str, int]] = {}
    case_id_to_label = {case.case_id: int(label) for case, label in zip(cleaned_cases, labels)}
    case_id_to_file = {case.case_id: case.file_name for case in cleaned_cases}
    case_id_to_raw_label = {case.case_id: case.raw_label for case in cleaned_cases}

    for node_type, nodes in global_graph["nodes_by_type"].items():
        is_case = node_type == "case"
        feat_dim = case_feature_dim if is_case else non_case_feature_dim

        node_features = np.zeros((len(nodes), feat_dim), dtype=np.float32)
        node_ids: list[str] = []

        for idx, node in enumerate(nodes):
            embedding = embeddings[(node_type, node["node_key"])]
            if is_case:
                scalars = _case_scalar_vector(node, scalar_dim=scalar_dim, scalar_names=case_scalar_names)
            elif node_type in TEXT_NODE_TYPES:
                scalars = _text_scalar_vector(node_type, node, scalar_dim=scalar_dim)
            elif node_type in ENTITY_NODE_TYPES:
                scalars = _entity_scalar_vector(node, scalar_dim=scalar_dim, scalar_names=entity_scalar_names)
            else:
                scalars = np.zeros((scalar_dim,), dtype=np.float32)
            node_features[idx] = np.concatenate([embedding, scalars], axis=0)
            node_ids.append(node["node_key"])

        data[node_type].x = torch.tensor(node_features, dtype=torch.float32)
        data[node_type].node_id = node_ids
        node_mappings[node_type] = {node["node_key"]: i for i, node in enumerate(nodes)}

        if is_case:
            case_ids = [node["metadata"]["case_id"] for node in nodes]
            case_labels = torch.tensor(
                [case_id_to_label[case_id] for case_id in case_ids], dtype=torch.long
            )
            train_mask = torch.tensor(
                [split_assignments[case_id] == "train" for case_id in case_ids], dtype=torch.bool
            )
            val_mask = torch.tensor(
                [split_assignments[case_id] == "val" for case_id in case_ids], dtype=torch.bool
            )
            test_mask = torch.tensor(
                [split_assignments[case_id] == "test" for case_id in case_ids], dtype=torch.bool
            )
            data["case"].y = case_labels
            data["case"].train_mask = train_mask
            data["case"].val_mask = val_mask
            data["case"].test_mask = test_mask
            data["case"].case_id = case_ids
            data["case"].file_name = [case_id_to_file[case_id] for case_id in case_ids]
            data["case"].raw_label = [case_id_to_raw_label[case_id] for case_id in case_ids]

    for edge_type, pairs in global_graph["edges_by_type"].items():
        if not pairs:
            continue
        data[edge_type].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()

    if bool(cfg.get("graph", {}).get("use_to_undirected", True)):
        data = ToUndirected()(data)

    metadata = {
        "label_names": label_names,
        "feature_dim": non_case_feature_dim,
        "case_feature_dim": case_feature_dim,
        "embedding_dim": non_case_dim,
        "case_text_dim": case_text_dim,
        "scalar_dim": scalar_dim,
        "encoder_backend": encoder_backend,
        "section_order": list(CASE_SECTION_ORDER),
        "node_mappings": node_mappings,
        "relation_mappings": ["|".join(edge_type) for edge_type in data.edge_types],
        "case_split_counts": {
            "train": int(data["case"].train_mask.sum().item()),
            "val": int(data["case"].val_mask.sum().item()),
            "test": int(data["case"].test_mask.sum().item()),
        },
        "node_counts": {node_type: int(data[node_type].num_nodes) for node_type in data.node_types},
    }
    return data, metadata
