#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv
from torch_geometric.utils import softmax


REPO_ROOT = Path(__file__).resolve().parents[1]
SECTION_GNN_ROOT = REPO_ROOT / "section_GNN"
if str(SECTION_GNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SECTION_GNN_ROOT))

from src.models.mlp_head import MLPHead  # noqa: E402


DEFAULT_MODEL_PATH = (
    SECTION_GNN_ROOT
    / "outputs/timed_bucket_runs/cross_bucket_total_dataset/models/"
    / "ablation_section_sep_enc_cross_bucket_kfold/kfold/fold_00/model.pt"
)
DEFAULT_CONFIG_PATH = (
    SECTION_GNN_ROOT
    / "ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml"
)
DEFAULT_GRAPH_CACHE = (
    SECTION_GNN_ROOT
    / "data/timed_bucket_runs/cross_bucket_total_dataset/graph_cache/"
    / "case_star_cross_bucket_section_sep_enc.reasoning_focused.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "FINAL_EXPLANATION/outputs/target_fold_00"

SECTION_NODE_TYPES = {
    "preamble",
    "facts",
    "arguments",
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
}
LEGAL_EVIDENCE_NODE_TYPES = {
    "statute",
    "provision",
    "precedent",
    "court",
    "judge",
    "petitioner_lawyer",
    "defence_lawyer",
    "lawyer",
    "petitioner",
    "respondent",
}
LEAKAGE_AUDIT_TYPES = {
    "judge",
    "petitioner",
    "respondent",
    "petitioner_lawyer",
    "defence_lawyer",
    "lawyer",
    "court",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def edge_type_to_str(edge_type: tuple[str, str, str]) -> str:
    return f"{edge_type[0]}__{edge_type[1]}__{edge_type[2]}"


def edge_type_from_str(value: str) -> tuple[str, str, str]:
    parts = value.split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type string: {value}")
    return (parts[0], parts[1], parts[2])


def display_node_id(node_type: str, node_id: str, max_len: int = 180) -> str:
    marker = f"::{node_type}::"
    if marker in node_id:
        label = node_id.split(marker, 1)[1]
    elif node_id.startswith(f"{node_type}::"):
        label = node_id.split("::", 1)[1]
    elif "::" in node_id:
        label = node_id.rsplit("::", 1)[-1]
    else:
        label = node_id
    label = " ".join(str(label).split())
    if len(label) > max_len:
        return label[: max_len - 3] + "..."
    return label


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except Exception:
        return None
    return float(value)


class RecordingHGTConv(HGTConv):
    """HGTConv variant that records the per-edge attention used internally."""

    last_alpha: Tensor | None

    def message(
        self,
        k_j: Tensor,
        q_i: Tensor,
        v_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Tensor | None,
        size_i: int | None,
    ) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        self.last_alpha = alpha.detach()
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)


class RecordingHeteroLegalOutcomeGNN(nn.Module):
    """Local copy of the project HGT model with recordable HGT attention."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        out_dim: int,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.architecture = str(cfg.get("architecture", "hgt")).lower()
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.2))

        self.input_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dims[node_type], self.hidden_dim)
                for node_type in metadata[0]
            }
        )
        self.type_embeddings = nn.ParameterDict(
            {
                node_type: nn.Parameter(torch.zeros(1, self.hidden_dim))
                for node_type in metadata[0]
            }
        )
        self.layer_norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(self.hidden_dim)
                        for node_type in metadata[0]
                    }
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.architecture == "hgt":
            self.convs = nn.ModuleList(
                [
                    RecordingHGTConv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        metadata=metadata,
                        heads=self.num_heads,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif self.architecture in {"heteroconv", "rgcn"}:
            self.convs = nn.ModuleList(
                [
                    HeteroConv(
                        {
                            edge_type: SAGEConv(
                                (self.hidden_dim, self.hidden_dim),
                                self.hidden_dim,
                            )
                            for edge_type in metadata[1]
                        },
                        aggr="sum",
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise ValueError(f"Unsupported GNN architecture: {self.architecture}")

        self.classifier = MLPHead(
            in_dim=self.hidden_dim,
            hidden_dim=int(cfg.get("mlp_hidden_dim", self.hidden_dim)),
            out_dim=out_dim,
            dropout=self.dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.input_projections.values():
            projection.reset_parameters()
        for parameter in self.type_embeddings.values():
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        for layer_norms in self.layer_norms:
            for layer_norm in layer_norms.values():
                layer_norm.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for module in self.classifier.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _encode_inputs(self, x_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            node_type: self.input_projections[node_type](x)
            + self.type_embeddings[node_type]
            for node_type, x in x_dict.items()
        }

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[Any, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        hidden = self._encode_inputs(x_dict)
        for layer_idx, conv in enumerate(self.convs):
            residual = hidden
            if hasattr(conv, "last_alpha"):
                conv.last_alpha = None
            conv_out = conv(hidden, edge_index_dict)
            updated: dict[str, Tensor] = {}
            for node_type in residual:
                message = conv_out.get(node_type, residual[node_type])
                if message is None:
                    message = residual[node_type]
                message = F.dropout(message, p=self.dropout, training=self.training)
                updated[node_type] = self.layer_norms[layer_idx][node_type](
                    residual[node_type] + message
                )
                updated[node_type] = F.relu(updated[node_type])
            hidden = updated
        logits = self.classifier(hidden["case"])
        return logits, hidden


class SortedAdjacency:
    def __init__(self, data: Any) -> None:
        self.data = data
        self.in_edge_types_by_dst: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.out_edge_types_by_src: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.by_dst: dict[tuple[str, str, str], tuple[Tensor, Tensor]] = {}
        self.by_src: dict[tuple[str, str, str], tuple[Tensor, Tensor]] = {}

        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index.cpu()
            src_values = edge_index[0].to(torch.long)
            dst_values = edge_index[1].to(torch.long)

            src_order = torch.argsort(src_values, stable=True)
            dst_order = torch.argsort(dst_values, stable=True)
            self.by_src[edge_type] = (
                src_values[src_order].contiguous(),
                src_order.to(torch.long).contiguous(),
            )
            self.by_dst[edge_type] = (
                dst_values[dst_order].contiguous(),
                dst_order.to(torch.long).contiguous(),
            )
            self.out_edge_types_by_src[edge_type[0]].append(edge_type)
            self.in_edge_types_by_dst[edge_type[2]].append(edge_type)

    @staticmethod
    def _positions_for_values(
        sorted_values: Tensor,
        sorted_positions: Tensor,
        values: set[int],
    ) -> Tensor:
        if not values:
            return torch.empty(0, dtype=torch.long)
        chunks: list[Tensor] = []
        for value in values:
            key = torch.tensor(value, dtype=sorted_values.dtype)
            left = int(torch.searchsorted(sorted_values, key, right=False))
            right = int(torch.searchsorted(sorted_values, key, right=True))
            if right > left:
                chunks.append(sorted_positions[left:right])
        if not chunks:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(chunks).unique(sorted=True)

    def positions_by_dst(
        self,
        edge_type: tuple[str, str, str],
        dst_indices: set[int],
    ) -> Tensor:
        sorted_values, sorted_positions = self.by_dst[edge_type]
        return self._positions_for_values(sorted_values, sorted_positions, dst_indices)

    def positions_by_src(
        self,
        edge_type: tuple[str, str, str],
        src_indices: set[int],
    ) -> Tensor:
        sorted_values, sorted_positions = self.by_src[edge_type]
        return self._positions_for_values(sorted_values, sorted_positions, src_indices)


@dataclass
class LocalRF:
    nodes: dict[str, Tensor]
    global_to_local: dict[str, dict[int, int]]
    x_dict: dict[str, Tensor]
    edge_index_dict: dict[tuple[str, str, str], Tensor]
    edge_global_pos: dict[tuple[str, str, str], Tensor]
    target_case_local: int


@dataclass
class MaskGroup:
    group_id: str
    group_kind: str
    evidence_type: str
    evidence_global_index: int | None
    evidence_id: str
    evidence_name: str
    path_family: str
    relation_types: str
    mask_positions: dict[tuple[str, str, str], Tensor]
    masked_edge_count: int
    attention_score: float | None = None


class SupportComputer:
    def __init__(
        self,
        data: Any,
        adjacency: SortedAdjacency,
        train_case_mask: np.ndarray,
        label_names: list[str],
        support_hops: int,
    ) -> None:
        self.data = data
        self.adjacency = adjacency
        self.train_case_mask = train_case_mask.astype(bool)
        self.label_names = label_names
        self.support_hops = support_hops
        self.case_labels = data["case"].y.detach().cpu().numpy()
        self.cache: dict[tuple[str, int], dict[str, Any]] = {}

    def compute(self, node_type: str, node_index: int) -> dict[str, Any]:
        key = (node_type, int(node_index))
        if key in self.cache:
            return self.cache[key]

        seen: set[tuple[str, int]] = {key}
        frontier: set[tuple[str, int]] = {key}
        connected_train_cases: set[int] = set()

        if node_type == "case" and self.train_case_mask[node_index]:
            connected_train_cases.add(node_index)

        for _ in range(self.support_hops):
            next_frontier: set[tuple[str, int]] = set()
            by_type: dict[str, set[int]] = defaultdict(set)
            for current_type, current_idx in frontier:
                by_type[current_type].add(current_idx)

            for current_type, indices in by_type.items():
                for edge_type in self.adjacency.out_edge_types_by_src.get(current_type, []):
                    positions = self.adjacency.positions_by_src(edge_type, indices)
                    if positions.numel() == 0:
                        continue
                    dst_type = edge_type[2]
                    dst_nodes = self.data[edge_type].edge_index[1, positions].tolist()
                    for dst_idx in dst_nodes:
                        item = (dst_type, int(dst_idx))
                        if item not in seen:
                            seen.add(item)
                            next_frontier.add(item)
                        if dst_type == "case" and self.train_case_mask[int(dst_idx)]:
                            connected_train_cases.add(int(dst_idx))

                for edge_type in self.adjacency.in_edge_types_by_dst.get(current_type, []):
                    positions = self.adjacency.positions_by_dst(edge_type, indices)
                    if positions.numel() == 0:
                        continue
                    src_type = edge_type[0]
                    src_nodes = self.data[edge_type].edge_index[0, positions].tolist()
                    for src_idx in src_nodes:
                        item = (src_type, int(src_idx))
                        if item not in seen:
                            seen.add(item)
                            next_frontier.add(item)
                        if src_type == "case" and self.train_case_mask[int(src_idx)]:
                            connected_train_cases.add(int(src_idx))

            frontier = next_frontier
            if not frontier:
                break

        label_counts = {label: 0 for label in self.label_names}
        for case_idx in connected_train_cases:
            label = self.label_names[int(self.case_labels[case_idx])]
            label_counts[label] = label_counts.get(label, 0) + 1

        support_n = int(sum(label_counts.values()))
        positive_label = "1" if "1" in label_counts else self.label_names[-1]
        negative_label = "-1" if "-1" in label_counts else self.label_names[0]
        result = {
            "support_train_n": support_n,
            "support_positive_label": positive_label,
            "support_negative_label": negative_label,
            "support_positive_n": int(label_counts.get(positive_label, 0)),
            "support_negative_n": int(label_counts.get(negative_label, 0)),
            "support_positive_rate": (
                float(label_counts.get(positive_label, 0) / support_n)
                if support_n
                else None
            ),
            "support_negative_rate": (
                float(label_counts.get(negative_label, 0) / support_n)
                if support_n
                else None
            ),
        }
        for label in self.label_names:
            result[f"support_label_{label}"] = int(label_counts.get(label, 0))
        self.cache[key] = result
        return result


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(
    data: Any,
    label_names: list[str],
    cfg: dict[str, Any],
    model_path: Path,
    device: torch.device,
    record_attention: bool,
) -> nn.Module:
    input_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}
    model = RecordingHeteroLegalOutcomeGNN(
        metadata=data.metadata(),
        input_dims=input_dims,
        out_dim=len(label_names),
        cfg=cfg.get("model", {}),
    )
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    if not record_attention:
        for conv in getattr(model, "convs", []):
            if hasattr(conv, "last_alpha"):
                conv.last_alpha = None
    return model


def extract_l_hop_induced_receptive_field(
    data: Any,
    adjacency: SortedAdjacency,
    case_index: int,
    num_hops: int,
) -> tuple[dict[str, set[int]], dict[tuple[str, str, str], set[int]]]:
    """Return the L-hop undirected node neighbourhood and its induced edges.

    The trained graph stores both original and reverse typed relations. HGT uses
    the directed edge tensors at every layer, so an exact local receptive field
    needs the undirected L-hop node set plus all directed edges among that node
    set, not only the backward computation-tree edges.
    """
    selected_nodes: dict[str, set[int]] = defaultdict(set)
    selected_nodes["case"].add(int(case_index))
    frontier: set[tuple[str, int]] = {("case", int(case_index))}

    for _ in range(num_hops):
        next_frontier: set[tuple[str, int]] = set()
        frontier_by_type: dict[str, set[int]] = defaultdict(set)
        for node_type, node_index in frontier:
            frontier_by_type[node_type].add(node_index)

        for node_type, node_indices in frontier_by_type.items():
            for edge_type in adjacency.out_edge_types_by_src.get(node_type, []):
                positions = adjacency.positions_by_src(edge_type, node_indices)
                if positions.numel() == 0:
                    continue
                dst_type = edge_type[2]
                dst_nodes = data[edge_type].edge_index[1, positions].tolist()
                for dst_idx in dst_nodes:
                    dst_idx = int(dst_idx)
                    selected_nodes[dst_type].add(dst_idx)
                    next_frontier.add((dst_type, dst_idx))

            for edge_type in adjacency.in_edge_types_by_dst.get(node_type, []):
                positions = adjacency.positions_by_dst(edge_type, node_indices)
                if positions.numel() == 0:
                    continue
                src_type = edge_type[0]
                src_nodes = data[edge_type].edge_index[0, positions].tolist()
                for src_idx in src_nodes:
                    src_idx = int(src_idx)
                    selected_nodes[src_type].add(src_idx)
                    next_frontier.add((src_type, src_idx))
        frontier = next_frontier
        if not frontier:
            break

    selected_edges: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        src_set = selected_nodes.get(src_type, set())
        dst_set = selected_nodes.get(dst_type, set())
        if not src_set or not dst_set:
            continue

        src_positions = adjacency.positions_by_src(edge_type, src_set)
        dst_positions = adjacency.positions_by_dst(edge_type, dst_set)
        if src_positions.numel() == 0 or dst_positions.numel() == 0:
            continue
        candidate_positions = (
            src_positions if src_positions.numel() <= dst_positions.numel() else dst_positions
        )
        edge_index = data[edge_type].edge_index[:, candidate_positions]
        src_mask = torch.tensor(
            [int(idx) in src_set for idx in edge_index[0].tolist()],
            dtype=torch.bool,
        )
        dst_mask = torch.tensor(
            [int(idx) in dst_set for idx in edge_index[1].tolist()],
            dtype=torch.bool,
        )
        keep = src_mask & dst_mask
        if bool(keep.any()):
            selected_edges[edge_type].update(int(pos) for pos in candidate_positions[keep].tolist())

    return dict(selected_nodes), dict(selected_edges)


def build_local_rf(
    data: Any,
    selected_nodes: dict[str, set[int]],
    selected_edges: dict[tuple[str, str, str], set[int]],
    case_index: int,
) -> LocalRF:
    nodes: dict[str, Tensor] = {}
    global_to_local: dict[str, dict[int, int]] = {}
    x_dict: dict[str, Tensor] = {}
    for node_type, node_indices in selected_nodes.items():
        ordered = sorted(node_indices)
        tensor = torch.tensor(ordered, dtype=torch.long)
        nodes[node_type] = tensor
        global_to_local[node_type] = {global_idx: local_idx for local_idx, global_idx in enumerate(ordered)}
        x_dict[node_type] = data[node_type].x[tensor].detach().cpu()

    edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
    edge_global_pos: dict[tuple[str, str, str], Tensor] = {}
    for edge_type, edge_positions in selected_edges.items():
        if not edge_positions:
            continue
        src_type, _, dst_type = edge_type
        ordered_positions = sorted(edge_positions)
        global_pos = torch.tensor(ordered_positions, dtype=torch.long)
        global_edge_index = data[edge_type].edge_index[:, global_pos].detach().cpu()
        src_map = global_to_local[src_type]
        dst_map = global_to_local[dst_type]
        src_local = [src_map[int(idx)] for idx in global_edge_index[0].tolist()]
        dst_local = [dst_map[int(idx)] for idx in global_edge_index[1].tolist()]
        local_edge_index = torch.tensor([src_local, dst_local], dtype=torch.long)
        edge_index_dict[edge_type] = local_edge_index
        edge_global_pos[edge_type] = global_pos

    return LocalRF(
        nodes=nodes,
        global_to_local=global_to_local,
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        edge_global_pos=edge_global_pos,
        target_case_local=global_to_local["case"][int(case_index)],
    )


def shortest_path_family(
    local_rf: LocalRF,
    target_node_type: str,
    target_local_index: int,
    max_hops: int,
) -> str:
    start = ("case", local_rf.target_case_local)
    target = (target_node_type, int(target_local_index))
    if start == target:
        return "case"

    outgoing: dict[tuple[str, int], list[tuple[str, int, tuple[str, str, str]]]] = defaultdict(list)
    for edge_type, edge_index in local_rf.edge_index_dict.items():
        if edge_type[1].startswith("rev_"):
            continue
        src_type, _, dst_type = edge_type
        for pos in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, pos])
            dst_idx = int(edge_index[1, pos])
            outgoing[(src_type, src_idx)].append((dst_type, dst_idx, edge_type))

    queue: deque[tuple[tuple[str, int], list[str]]] = deque()
    queue.append((start, ["case"]))
    seen = {start}
    while queue:
        node, path = queue.popleft()
        depth = (len(path) - 1) // 2
        if depth >= max_hops:
            continue
        for dst_type, dst_idx, edge_type in outgoing.get(node, []):
            next_node = (dst_type, dst_idx)
            next_path = path + [edge_type[1], dst_type]
            if next_node == target:
                return "->".join(next_path)
            if next_node not in seen:
                seen.add(next_node)
                queue.append((next_node, next_path))
    return target_node_type


def relation_path_family(edge_type: tuple[str, str, str]) -> str:
    relation = edge_type[1]
    if relation.startswith("rev_"):
        return f"{edge_type[0]}->{relation}->{edge_type[2]}"
    return f"{edge_type[0]}->{relation}->{edge_type[2]}"


def build_mask_groups(
    local_rf: LocalRF,
    node_ids: dict[str, list[str]],
    num_hops: int,
    include_relation_groups: bool,
    include_node_groups: bool,
) -> list[MaskGroup]:
    groups: list[MaskGroup] = []

    if include_relation_groups:
        for edge_type, edge_index in local_rf.edge_index_dict.items():
            if edge_index.shape[1] == 0:
                continue
            edge_positions = torch.arange(edge_index.shape[1], dtype=torch.long)
            relation_label = edge_type_to_str(edge_type)
            groups.append(
                MaskGroup(
                    group_id=f"relation::{relation_label}",
                    group_kind="relation_type",
                    evidence_type="relation_type",
                    evidence_global_index=None,
                    evidence_id=relation_label,
                    evidence_name=edge_type[1],
                    path_family=relation_path_family(edge_type),
                    relation_types=relation_label,
                    mask_positions={edge_type: edge_positions},
                    masked_edge_count=int(edge_positions.numel()),
                )
            )

    if not include_node_groups:
        return groups

    incident: dict[tuple[str, int], dict[tuple[str, str, str], list[int]]] = defaultdict(lambda: defaultdict(list))
    for edge_type, edge_index in local_rf.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        for pos in range(edge_index.shape[1]):
            src_local = int(edge_index[0, pos])
            dst_local = int(edge_index[1, pos])
            incident[(src_type, src_local)][edge_type].append(pos)
            incident[(dst_type, dst_local)][edge_type].append(pos)

    candidate_node_types = SECTION_NODE_TYPES | LEGAL_EVIDENCE_NODE_TYPES
    for (node_type, local_idx), per_edge_type in sorted(incident.items()):
        if node_type == "case" or node_type not in candidate_node_types:
            continue
        global_idx = int(local_rf.nodes[node_type][local_idx])
        evidence_id = node_ids[node_type][global_idx]
        mask_positions = {
            edge_type: torch.tensor(sorted(set(positions)), dtype=torch.long)
            for edge_type, positions in per_edge_type.items()
        }
        masked_count = int(sum(t.numel() for t in mask_positions.values()))
        if masked_count == 0:
            continue
        relation_types = "|".join(sorted(edge_type_to_str(edge_type) for edge_type in mask_positions))
        if node_type in SECTION_NODE_TYPES:
            group_kind = "argument_or_section_node"
        elif node_type in {"statute", "provision", "precedent"}:
            group_kind = "legal_authority_node"
        elif node_type in LEAKAGE_AUDIT_TYPES:
            group_kind = "names_court_judge_node"
        else:
            group_kind = "evidence_node"
        groups.append(
            MaskGroup(
                group_id=f"node::{node_type}::{global_idx}",
                group_kind=group_kind,
                evidence_type=node_type,
                evidence_global_index=global_idx,
                evidence_id=evidence_id,
                evidence_name=display_node_id(node_type, evidence_id),
                path_family=shortest_path_family(local_rf, node_type, local_idx, num_hops),
                relation_types=relation_types,
                mask_positions=mask_positions,
                masked_edge_count=masked_count,
            )
        )

    return groups


def move_x_dict(x_dict: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {node_type: x.to(device, non_blocking=True) for node_type, x in x_dict.items()}


def move_edge_dict(
    edge_index_dict: dict[tuple[str, str, str], Tensor],
    device: torch.device,
) -> dict[tuple[str, str, str], Tensor]:
    return {
        edge_type: edge_index.to(device, non_blocking=True)
        for edge_type, edge_index in edge_index_dict.items()
    }


@torch.no_grad()
def forward_local_cases(
    model: nn.Module,
    x_dict: dict[str, Tensor],
    edge_index_dict: dict[tuple[str, str, str], Tensor],
    case_indices: list[int],
    device: torch.device,
) -> Tensor:
    logits, _ = model(move_x_dict(x_dict, device), move_edge_dict(edge_index_dict, device))
    probs = torch.softmax(logits, dim=-1)
    return probs[torch.tensor(case_indices, dtype=torch.long, device=probs.device)].detach().cpu()


def make_masked_batch(
    local_rf: LocalRF,
    groups: list[MaskGroup],
) -> tuple[dict[str, Tensor], dict[tuple[str, str, str], Tensor], list[int]]:
    n_variants = len(groups)
    x_dict: dict[str, Tensor] = {}
    node_counts = {node_type: int(x.shape[0]) for node_type, x in local_rf.x_dict.items()}
    for node_type, x in local_rf.x_dict.items():
        x_dict[node_type] = x.repeat((n_variants, 1))

    edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
    for edge_type, base_edge_index in local_rf.edge_index_dict.items():
        src_type, _, dst_type = edge_type
        edge_chunks: list[Tensor] = []
        base_edge_count = base_edge_index.shape[1]
        if base_edge_count == 0:
            continue
        for variant_idx, group in enumerate(groups):
            keep = torch.ones(base_edge_count, dtype=torch.bool)
            masked_positions = group.mask_positions.get(edge_type)
            if masked_positions is not None and masked_positions.numel() > 0:
                keep[masked_positions] = False
            if not bool(keep.any()):
                continue
            shifted = base_edge_index[:, keep].clone()
            shifted[0] += variant_idx * node_counts[src_type]
            shifted[1] += variant_idx * node_counts[dst_type]
            edge_chunks.append(shifted)
        if edge_chunks:
            edge_index_dict[edge_type] = torch.cat(edge_chunks, dim=1)
        else:
            edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long)

    target_indices = [
        local_rf.target_case_local + variant_idx * node_counts["case"]
        for variant_idx in range(n_variants)
    ]
    return x_dict, edge_index_dict, target_indices


def edge_attention_by_type(
    model: nn.Module,
    edge_index_dict: dict[tuple[str, str, str], Tensor],
) -> dict[tuple[str, str, str], Tensor]:
    edge_counts = {edge_type: edge_index.shape[1] for edge_type, edge_index in edge_index_dict.items()}
    attention_sum: dict[tuple[str, str, str], Tensor] = {}
    attention_count: dict[tuple[str, str, str], int] = defaultdict(int)

    for conv in getattr(model, "convs", []):
        alpha = getattr(conv, "last_alpha", None)
        if alpha is None:
            continue
        alpha_cpu = alpha.detach().cpu().mean(dim=1)
        offset = 0
        for edge_type, count in edge_counts.items():
            if count == 0:
                continue
            current = alpha_cpu[offset : offset + count]
            if edge_type not in attention_sum:
                attention_sum[edge_type] = current.clone()
            else:
                attention_sum[edge_type] = attention_sum[edge_type] + current
            attention_count[edge_type] += 1
            offset += count

    result: dict[tuple[str, str, str], Tensor] = {}
    for edge_type, values in attention_sum.items():
        count = max(1, attention_count[edge_type])
        result[edge_type] = values / count
    return result


def attach_attention_scores(
    groups: list[MaskGroup],
    attention_by_edge_type: dict[tuple[str, str, str], Tensor],
) -> None:
    for group in groups:
        chunks: list[Tensor] = []
        for edge_type, positions in group.mask_positions.items():
            values = attention_by_edge_type.get(edge_type)
            if values is not None and positions.numel() > 0:
                chunks.append(values[positions])
        if chunks:
            group.attention_score = float(torch.cat(chunks).mean().item())
        else:
            group.attention_score = None


def update_aggregate(
    aggregate: dict[str, dict[str, Any]],
    key: str,
    row: dict[str, Any],
) -> None:
    item = aggregate.setdefault(
        key,
        {
            "n_groups": 0,
            "case_indices": set(),
            "sum_delta_pred_proba": 0.0,
            "sum_abs_delta_pred_proba": 0.0,
            "sum_attention_score": 0.0,
            "n_attention": 0,
            "n_flips": 0,
            "sum_masked_edge_count": 0,
        },
    )
    item["n_groups"] += 1
    item["case_indices"].add(row["case_index"])
    item["sum_delta_pred_proba"] += float(row["delta_pred_proba"])
    item["sum_abs_delta_pred_proba"] += float(row["abs_delta_pred_proba"])
    att = safe_float(row.get("attention_score"))
    if att is not None:
        item["sum_attention_score"] += att
        item["n_attention"] += 1
    if row["prediction_flipped"]:
        item["n_flips"] += 1
    item["sum_masked_edge_count"] += int(row["masked_edge_count"])


def aggregate_to_rows(
    aggregate: dict[str, dict[str, Any]],
    key_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in aggregate.items():
        n_groups = int(item["n_groups"])
        rows.append(
            {
                key_name: key,
                "n_groups": n_groups,
                "n_cases": len(item["case_indices"]),
                "mean_delta_pred_proba": item["sum_delta_pred_proba"] / n_groups,
                "mean_abs_delta_pred_proba": item["sum_abs_delta_pred_proba"] / n_groups,
                "sum_abs_delta_pred_proba": item["sum_abs_delta_pred_proba"],
                "flip_rate": item["n_flips"] / n_groups,
                "mean_masked_edge_count": item["sum_masked_edge_count"] / n_groups,
                "mean_attention_score": (
                    item["sum_attention_score"] / item["n_attention"]
                    if item["n_attention"]
                    else None
                ),
            }
        )
    rows.sort(key=lambda row: row["sum_abs_delta_pred_proba"], reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for filename in (
        "case_counterfactual_groups.csv",
        "case_top_explanations.csv",
        "case_summary.csv",
        "connected_case_label_distribution.csv",
        "typed_path_importance.csv",
        "relation_type_importance.csv",
        "evidence_type_importance.csv",
        "leakage_sensitivity_summary.csv",
        "attention_counterfactual_overlap.csv",
        "run_summary.json",
        "manifest.json",
    ):
        target = path / filename
        if target.exists():
            target.unlink()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Typed counterfactual masking for the frozen HGT legal graph model."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--graph-cache", type=Path, default=DEFAULT_GRAPH_CACHE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Defaults to predictions.csv next to --model-path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--top-k-per-case", type=int, default=10)
    parser.add_argument("--num-hops", type=int, default=None)
    parser.add_argument("--mask-batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--support-hops", type=int, default=2)
    parser.add_argument("--support-mode", choices=["all", "top", "none"], default="all")
    parser.add_argument("--attention-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-relation-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-node-groups", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    start_time = time.time()
    predictions_csv = args.predictions_csv or (args.model_path.parent / "predictions.csv")
    prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    cfg = load_yaml(args.config)
    num_hops = args.num_hops or int(cfg.get("model", {}).get("num_layers", 2))
    device = choose_device(args.device)

    print(f"[load] graph={args.graph_cache}", flush=True)
    bundle = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    data = bundle["data"]
    metadata = bundle["metadata"]
    label_names = [str(label) for label in metadata["label_names"]]
    node_ids = {node_type: list(data[node_type].node_id) for node_type in data.node_types}

    print(f"[load] predictions={predictions_csv}", flush=True)
    predictions = pd.read_csv(predictions_csv)
    if len(predictions) != data["case"].num_nodes:
        raise ValueError(
            f"Predictions rows ({len(predictions)}) do not match graph cases "
            f"({data['case'].num_nodes}). Pass the matching --predictions-csv."
        )
    split_values = predictions["split"].astype(str).to_numpy()
    if args.split == "all":
        selected_case_indices = np.arange(len(predictions), dtype=np.int64)
    else:
        selected_case_indices = np.where(split_values == args.split)[0]
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    selected_case_indices = selected_case_indices[args.shard_index :: args.num_shards]
    if args.case_start:
        selected_case_indices = selected_case_indices[args.case_start :]
    if args.case_limit is not None:
        selected_case_indices = selected_case_indices[: args.case_limit]

    print(
        f"[load] cases={len(selected_case_indices)} split={args.split} "
        f"shard={args.shard_index}/{args.num_shards} hops={num_hops} device={device}",
        flush=True,
    )
    adjacency = SortedAdjacency(data)
    model = load_model(
        data=data,
        label_names=label_names,
        cfg=cfg,
        model_path=args.model_path,
        device=device,
        record_attention=args.attention_audit,
    )

    train_case_mask = split_values == "train"
    support_computer = SupportComputer(
        data=data,
        adjacency=adjacency,
        train_case_mask=train_case_mask,
        label_names=label_names,
        support_hops=args.support_hops,
    )

    long_fieldnames = [
        "case_index",
        "case_id",
        "file_name",
        "split",
        "target_label",
        "saved_pred_label",
        "baseline_pred_label",
        "baseline_pred_proba",
        "baseline_true_proba",
        "baseline_positive_proba",
        "rf_node_count",
        "rf_edge_count",
        "group_rank_abs",
        "group_id",
        "group_kind",
        "evidence_type",
        "evidence_global_index",
        "evidence_id",
        "evidence_name",
        "path_family",
        "relation_types",
        "masked_edge_count",
        "masked_pred_label",
        "masked_pred_proba",
        "masked_true_proba",
        "masked_positive_proba",
        "delta_pred_proba",
        "abs_delta_pred_proba",
        "delta_true_proba",
        "delta_positive_proba",
        "prediction_flipped",
        "attention_score",
        "support_train_n",
        "support_negative_n",
        "support_positive_n",
        "support_negative_rate",
        "support_positive_rate",
    ] + [f"support_label_{label}" for label in label_names]

    counterfactual_path = args.output_dir / "case_counterfactual_groups.csv"
    top_path = args.output_dir / "case_top_explanations.csv"
    case_summary_path = args.output_dir / "case_summary.csv"
    overlap_path = args.output_dir / "attention_counterfactual_overlap.csv"

    path_aggregate: dict[str, dict[str, Any]] = {}
    relation_aggregate: dict[str, dict[str, Any]] = {}
    evidence_type_aggregate: dict[str, dict[str, Any]] = {}
    leakage_values: dict[str, list[float]] = defaultdict(list)
    leakage_flips: dict[str, int] = defaultdict(int)
    leakage_counts: dict[str, int] = defaultdict(int)
    support_rows: dict[tuple[str, int], dict[str, Any]] = {}
    processed_cases = 0
    processed_groups = 0
    failed_cases: list[dict[str, Any]] = []

    with counterfactual_path.open("w", newline="", encoding="utf-8") as long_handle, \
        top_path.open("w", newline="", encoding="utf-8") as top_handle, \
        case_summary_path.open("w", newline="", encoding="utf-8") as summary_handle, \
        overlap_path.open("w", newline="", encoding="utf-8") as overlap_handle:

        long_writer = csv.DictWriter(long_handle, fieldnames=long_fieldnames, extrasaction="ignore")
        top_writer = csv.DictWriter(top_handle, fieldnames=long_fieldnames, extrasaction="ignore")
        summary_writer = csv.DictWriter(
            summary_handle,
            fieldnames=[
                "case_index",
                "case_id",
                "split",
                "target_label",
                "saved_pred_label",
                "baseline_pred_label",
                "baseline_pred_proba",
                "baseline_positive_proba",
                "rf_node_count",
                "rf_edge_count",
                "n_groups",
                "max_abs_delta_pred_proba",
                "top_group_id",
                "top_evidence_type",
                "top_evidence_name",
                "top_path_family",
                "n_prediction_flips",
            ],
        )
        overlap_writer = csv.DictWriter(
            overlap_handle,
            fieldnames=[
                "case_index",
                "case_id",
                "n_groups",
                "top_k",
                "counterfactual_attention_overlap",
                "counterfactual_attention_jaccard",
                "attention_available",
            ],
        )
        long_writer.writeheader()
        top_writer.writeheader()
        summary_writer.writeheader()
        overlap_writer.writeheader()

        positive_index = label_names.index("1") if "1" in label_names else len(label_names) - 1

        for ordinal, case_index_np in enumerate(selected_case_indices, start=1):
            case_index = int(case_index_np)
            try:
                selected_nodes, selected_edges = extract_l_hop_induced_receptive_field(
                    data=data,
                    adjacency=adjacency,
                    case_index=case_index,
                    num_hops=num_hops,
                )
                local_rf = build_local_rf(data, selected_nodes, selected_edges, case_index)
                rf_node_count = int(sum(t.numel() for t in local_rf.nodes.values()))
                rf_edge_count = int(sum(t.shape[1] for t in local_rf.edge_index_dict.values()))
                if rf_edge_count == 0:
                    continue

                groups = build_mask_groups(
                    local_rf=local_rf,
                    node_ids=node_ids,
                    num_hops=num_hops,
                    include_relation_groups=args.include_relation_groups,
                    include_node_groups=args.include_node_groups,
                )
                if not groups:
                    continue

                baseline_probs = forward_local_cases(
                    model=model,
                    x_dict=local_rf.x_dict,
                    edge_index_dict=local_rf.edge_index_dict,
                    case_indices=[local_rf.target_case_local],
                    device=device,
                )[0]
                attention_by_edge_type = (
                    edge_attention_by_type(model, local_rf.edge_index_dict)
                    if args.attention_audit
                    else {}
                )
                if args.attention_audit:
                    attach_attention_scores(groups, attention_by_edge_type)

                baseline_pred_idx = int(torch.argmax(baseline_probs).item())
                target_idx = int(data["case"].y[case_index].item())
                baseline_pred_proba = float(baseline_probs[baseline_pred_idx].item())
                baseline_true_proba = float(baseline_probs[target_idx].item())
                baseline_positive_proba = float(baseline_probs[positive_index].item())

                group_rows: list[dict[str, Any]] = []
                for chunk_start in range(0, len(groups), args.mask_batch_size):
                    chunk = groups[chunk_start : chunk_start + args.mask_batch_size]
                    batch_x, batch_edges, target_indices = make_masked_batch(local_rf, chunk)
                    masked_probs = forward_local_cases(
                        model=model,
                        x_dict=batch_x,
                        edge_index_dict=batch_edges,
                        case_indices=target_indices,
                        device=device,
                    )
                    for group, probs in zip(chunk, masked_probs):
                        masked_pred_idx = int(torch.argmax(probs).item())
                        masked_pred_proba = float(probs[baseline_pred_idx].item())
                        masked_true_proba = float(probs[target_idx].item())
                        masked_positive_proba = float(probs[positive_index].item())
                        delta_pred = baseline_pred_proba - masked_pred_proba
                        delta_true = baseline_true_proba - masked_true_proba
                        delta_positive = baseline_positive_proba - masked_positive_proba

                        support: dict[str, Any] = {}
                        if (
                            args.support_mode == "all"
                            and group.evidence_global_index is not None
                        ):
                            support = support_computer.compute(
                                group.evidence_type,
                                group.evidence_global_index,
                            )
                            support_key = (group.evidence_type, group.evidence_global_index)
                            support_rows.setdefault(
                                support_key,
                                {
                                    "evidence_type": group.evidence_type,
                                    "evidence_global_index": group.evidence_global_index,
                                    "evidence_id": group.evidence_id,
                                    "evidence_name": group.evidence_name,
                                    **support,
                                },
                            )

                        row = {
                            "case_index": case_index,
                            "case_id": data["case"].case_id[case_index],
                            "file_name": data["case"].file_name[case_index],
                            "split": split_values[case_index],
                            "target_label": label_names[target_idx],
                            "saved_pred_label": str(predictions.iloc[case_index].get("pred_label", "")),
                            "baseline_pred_label": label_names[baseline_pred_idx],
                            "baseline_pred_proba": baseline_pred_proba,
                            "baseline_true_proba": baseline_true_proba,
                            "baseline_positive_proba": baseline_positive_proba,
                            "rf_node_count": rf_node_count,
                            "rf_edge_count": rf_edge_count,
                            "group_rank_abs": None,
                            "group_id": group.group_id,
                            "group_kind": group.group_kind,
                            "evidence_type": group.evidence_type,
                            "evidence_global_index": group.evidence_global_index,
                            "evidence_id": group.evidence_id,
                            "evidence_name": group.evidence_name,
                            "path_family": group.path_family,
                            "relation_types": group.relation_types,
                            "masked_edge_count": group.masked_edge_count,
                            "masked_pred_label": label_names[masked_pred_idx],
                            "masked_pred_proba": masked_pred_proba,
                            "masked_true_proba": masked_true_proba,
                            "masked_positive_proba": masked_positive_proba,
                            "delta_pred_proba": delta_pred,
                            "abs_delta_pred_proba": abs(delta_pred),
                            "delta_true_proba": delta_true,
                            "delta_positive_proba": delta_positive,
                            "prediction_flipped": masked_pred_idx != baseline_pred_idx,
                            "attention_score": group.attention_score,
                            **support,
                        }
                        group_rows.append(row)

                group_rows.sort(key=lambda item: item["abs_delta_pred_proba"], reverse=True)
                for rank, row in enumerate(group_rows, start=1):
                    row["group_rank_abs"] = rank
                    if args.support_mode == "top" and rank <= args.top_k_per_case:
                        matching_group = next(
                            group for group in groups if group.group_id == row["group_id"]
                        )
                        if matching_group.evidence_global_index is not None:
                            support = support_computer.compute(
                                matching_group.evidence_type,
                                matching_group.evidence_global_index,
                            )
                            row.update(support)
                            support_key = (
                                matching_group.evidence_type,
                                matching_group.evidence_global_index,
                            )
                            support_rows.setdefault(
                                support_key,
                                {
                                    "evidence_type": matching_group.evidence_type,
                                    "evidence_global_index": matching_group.evidence_global_index,
                                    "evidence_id": matching_group.evidence_id,
                                    "evidence_name": matching_group.evidence_name,
                                    **support,
                                },
                            )
                    long_writer.writerow(row)
                    update_aggregate(path_aggregate, row["path_family"], row)
                    update_aggregate(evidence_type_aggregate, row["evidence_type"], row)
                    for relation_type in str(row["relation_types"]).split("|"):
                        if relation_type:
                            update_aggregate(relation_aggregate, relation_type, row)

                    if row["evidence_type"] in LEAKAGE_AUDIT_TYPES:
                        leakage_values[row["evidence_type"]].append(row["abs_delta_pred_proba"])
                        leakage_counts[row["evidence_type"]] += 1
                        if row["prediction_flipped"]:
                            leakage_flips[row["evidence_type"]] += 1

                    if rank <= args.top_k_per_case:
                        top_writer.writerow(row)

                top_row = group_rows[0] if group_rows else None
                summary_writer.writerow(
                    {
                        "case_index": case_index,
                        "case_id": data["case"].case_id[case_index],
                        "split": split_values[case_index],
                        "target_label": label_names[target_idx],
                        "saved_pred_label": str(predictions.iloc[case_index].get("pred_label", "")),
                        "baseline_pred_label": label_names[baseline_pred_idx],
                        "baseline_pred_proba": baseline_pred_proba,
                        "baseline_positive_proba": baseline_positive_proba,
                        "rf_node_count": rf_node_count,
                        "rf_edge_count": rf_edge_count,
                        "n_groups": len(group_rows),
                        "max_abs_delta_pred_proba": (
                            top_row["abs_delta_pred_proba"] if top_row else None
                        ),
                        "top_group_id": top_row["group_id"] if top_row else None,
                        "top_evidence_type": top_row["evidence_type"] if top_row else None,
                        "top_evidence_name": top_row["evidence_name"] if top_row else None,
                        "top_path_family": top_row["path_family"] if top_row else None,
                        "n_prediction_flips": sum(
                            1 for row in group_rows if row["prediction_flipped"]
                        ),
                    }
                )

                if args.attention_audit:
                    att_rows = [
                        row for row in group_rows if safe_float(row.get("attention_score")) is not None
                    ]
                    top_k = min(args.top_k_per_case, len(group_rows), len(att_rows))
                    if top_k > 0:
                        top_cf = {row["group_id"] for row in group_rows[:top_k]}
                        top_att = {
                            row["group_id"]
                            for row in sorted(
                                att_rows,
                                key=lambda item: float(item["attention_score"]),
                                reverse=True,
                            )[:top_k]
                        }
                        overlap = len(top_cf & top_att)
                        union = len(top_cf | top_att)
                        overlap_writer.writerow(
                            {
                                "case_index": case_index,
                                "case_id": data["case"].case_id[case_index],
                                "n_groups": len(group_rows),
                                "top_k": top_k,
                                "counterfactual_attention_overlap": overlap / top_k,
                                "counterfactual_attention_jaccard": overlap / union if union else None,
                                "attention_available": True,
                            }
                        )
                    else:
                        overlap_writer.writerow(
                            {
                                "case_index": case_index,
                                "case_id": data["case"].case_id[case_index],
                                "n_groups": len(group_rows),
                                "top_k": args.top_k_per_case,
                                "counterfactual_attention_overlap": None,
                                "counterfactual_attention_jaccard": None,
                                "attention_available": False,
                            }
                        )

                processed_cases += 1
                processed_groups += len(group_rows)
                if processed_cases % args.progress_every == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[progress] cases={processed_cases}/{len(selected_case_indices)} "
                        f"groups={processed_groups} elapsed_min={elapsed / 60:.1f}",
                        flush=True,
                    )

            except Exception as exc:
                failed_cases.append(
                    {
                        "case_index": case_index,
                        "case_id": data["case"].case_id[case_index],
                        "error": repr(exc),
                    }
                )
                print(f"[warn] case_index={case_index} failed: {exc!r}", flush=True)

    write_csv(
        args.output_dir / "typed_path_importance.csv",
        aggregate_to_rows(path_aggregate, "path_family"),
    )
    relation_rows = aggregate_to_rows(relation_aggregate, "relation_type")
    for row in relation_rows:
        try:
            edge_type = edge_type_from_str(row["relation_type"])
            row["src_type"] = edge_type[0]
            row["relation"] = edge_type[1]
            row["dst_type"] = edge_type[2]
        except Exception:
            row["src_type"] = None
            row["relation"] = None
            row["dst_type"] = None
    write_csv(args.output_dir / "relation_type_importance.csv", relation_rows)
    write_csv(
        args.output_dir / "evidence_type_importance.csv",
        aggregate_to_rows(evidence_type_aggregate, "evidence_type"),
    )

    leakage_rows: list[dict[str, Any]] = []
    for evidence_type, values in sorted(leakage_values.items()):
        arr = np.asarray(values, dtype=float)
        leakage_rows.append(
            {
                "evidence_type": evidence_type,
                "n_groups": int(leakage_counts[evidence_type]),
                "mean_abs_delta_pred_proba": float(np.mean(arr)) if arr.size else None,
                "median_abs_delta_pred_proba": float(np.median(arr)) if arr.size else None,
                "p95_abs_delta_pred_proba": float(np.quantile(arr, 0.95)) if arr.size else None,
                "max_abs_delta_pred_proba": float(np.max(arr)) if arr.size else None,
                "flip_rate": (
                    leakage_flips[evidence_type] / leakage_counts[evidence_type]
                    if leakage_counts[evidence_type]
                    else None
                ),
            }
        )
    leakage_rows.sort(key=lambda row: row["mean_abs_delta_pred_proba"] or 0.0, reverse=True)
    write_csv(args.output_dir / "leakage_sensitivity_summary.csv", leakage_rows)

    support_output_rows = list(support_rows.values())
    support_output_rows.sort(
        key=lambda row: (
            int(row.get("support_train_n") or 0),
            str(row.get("evidence_type") or ""),
            str(row.get("evidence_name") or ""),
        ),
        reverse=True,
    )
    write_csv(args.output_dir / "connected_case_label_distribution.csv", support_output_rows)

    manifest = {
        "counterfactual_groups": str(counterfactual_path),
        "top_explanations": str(top_path),
        "case_summary": str(case_summary_path),
        "connected_case_label_distribution": str(args.output_dir / "connected_case_label_distribution.csv"),
        "typed_path_importance": str(args.output_dir / "typed_path_importance.csv"),
        "relation_type_importance": str(args.output_dir / "relation_type_importance.csv"),
        "evidence_type_importance": str(args.output_dir / "evidence_type_importance.csv"),
        "leakage_sensitivity_summary": str(args.output_dir / "leakage_sensitivity_summary.csv"),
        "attention_counterfactual_overlap": str(overlap_path),
        "run_summary": str(args.output_dir / "run_summary.json"),
    }
    dump_json(args.output_dir / "manifest.json", manifest)
    dump_json(
        args.output_dir / "run_summary.json",
        {
            "model_path": str(args.model_path),
            "graph_cache": str(args.graph_cache),
            "config": str(args.config),
            "predictions_csv": str(predictions_csv),
            "output_dir": str(args.output_dir),
            "split": args.split,
            "case_limit": args.case_limit,
            "case_start": args.case_start,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "num_hops": num_hops,
            "receptive_field": "l_hop_undirected_node_neighbourhood_with_induced_directed_edges",
            "support_hops": args.support_hops,
            "support_mode": args.support_mode,
            "attention_audit": args.attention_audit,
            "mask_batch_size": args.mask_batch_size,
            "label_names": label_names,
            "processed_cases": processed_cases,
            "processed_groups": processed_groups,
            "failed_cases": failed_cases,
            "elapsed_seconds": time.time() - start_time,
        },
    )
    print(
        f"[done] cases={processed_cases} groups={processed_groups} "
        f"out={args.output_dir} elapsed_min={(time.time() - start_time) / 60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
