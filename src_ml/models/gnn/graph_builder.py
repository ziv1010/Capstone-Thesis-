from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src_ml.common.text_utils import safe_list, safe_text

NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(value: Any) -> str:
    return NORM_RE.sub("_", safe_text(value).lower()).strip("_")


@dataclass
class GraphBuildResult:
    x: np.ndarray
    edge_index: np.ndarray
    case_node_indices: np.ndarray
    case_id_to_node: dict[str, int]
    node_type: list[str]


def build_case_entity_graph(
    records: list[dict[str, Any]],
    case_embeddings: np.ndarray,
    add_case_case_edges: bool = True,
    seed: int = 42,
) -> GraphBuildResult:
    if len(records) != case_embeddings.shape[0]:
        raise ValueError("records and case_embeddings size mismatch")

    dim = case_embeddings.shape[1]
    rng = np.random.default_rng(seed)

    case_id_to_node: dict[str, int] = {}
    node_type: list[str] = []
    for idx, rec in enumerate(records):
        case_id = str(rec["case_id"])
        case_id_to_node[case_id] = idx
        node_type.append("case")

    entity_to_node: dict[str, int] = {}
    edges: list[tuple[int, int]] = []

    def get_entity_node(entity_key: str) -> int:
        if entity_key in entity_to_node:
            return entity_to_node[entity_key]
        node_idx = len(node_type)
        entity_to_node[entity_key] = node_idx
        node_type.append("entity")
        return node_idx

    title_to_case_node: dict[str, int] = {}
    for rec in records:
        cid = str(rec["case_id"])
        node = case_id_to_node[cid]
        title_norm = _norm(rec.get("case_title"))
        if title_norm:
            title_to_case_node[title_norm] = node

    for rec in records:
        case_node = case_id_to_node[str(rec["case_id"])]
        for statute in safe_list(rec.get("statutes")):
            e_node = get_entity_node(f"statute::{_norm(statute)}")
            edges.append((case_node, e_node))
            edges.append((e_node, case_node))
        for provision in safe_list(rec.get("provisions")):
            e_node = get_entity_node(f"provision::{_norm(provision)}")
            edges.append((case_node, e_node))
            edges.append((e_node, case_node))
        for precedent in safe_list(rec.get("precedents")):
            norm_prec = _norm(precedent)
            e_node = get_entity_node(f"precedent::{norm_prec}")
            edges.append((case_node, e_node))
            edges.append((e_node, case_node))

            if add_case_case_edges and norm_prec:
                # direct case-id mention
                if norm_prec in case_id_to_node:
                    other = case_id_to_node[norm_prec]
                    if other != case_node:
                        edges.append((case_node, other))
                        edges.append((other, case_node))
                # direct title mention
                if norm_prec in title_to_case_node:
                    other = title_to_case_node[norm_prec]
                    if other != case_node:
                        edges.append((case_node, other))
                        edges.append((other, case_node))

    n_nodes = len(node_type)
    x = np.zeros((n_nodes, dim), dtype=np.float32)
    x[: len(records)] = case_embeddings

    if n_nodes > len(records):
        x[len(records) :] = rng.normal(0.0, 0.01, size=(n_nodes - len(records), dim)).astype(np.float32)

    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    return GraphBuildResult(
        x=x,
        edge_index=edge_index,
        case_node_indices=np.arange(len(records), dtype=np.int64),
        case_id_to_node=case_id_to_node,
        node_type=node_type,
    )
