"""Architecture-ablation GNN for Reviewer 3 comment R3-04.

Answers: *"The paper uses HGT but does not compare against simpler GNNs (GCN,
GraphSAGE, GAT) on the same graph. It is unclear whether the complexity of
heterogeneous attention is necessary."*

`ArchLegalOutcomeGNN` is a drop-in replacement for
``src/models/hetero_gnn.HeteroLegalOutcomeGNN`` that keeps every part of the
paper model fixed -- per-node-type input projection to ``hidden_dim``, learned
node-type embedding, residual + per-type LayerNorm + ReLU, and the
``MLPHead`` classifier on the ``case`` node -- and swaps *only* the
message-passing operator:

    architecture   message passing                      graph  rel-aware  attention
    ------------   ----------------------------------   -----  ---------  ---------
    mlp            per-type Linear (no message passing)   no       no         no
    gcn            GCNConv   on the type-collapsed graph  yes      no         no
    sage           SAGEConv  on the type-collapsed graph  yes      no         no
    gat            GATConv   on the type-collapsed graph  yes      no      yes (untyped)
    rgcn           HeteroConv{relation: SAGEConv}         yes      yes        no
    hgat           HeteroConv{relation: GATConv}          yes      yes     yes (per-relation)
    hgt            HGTConv  -- delegates to the original  yes      yes     yes (typed)

The "type-collapsed" view is built from the *same* ``HeteroData``: after the
per-type input projection, node embeddings are concatenated with offsets and
all relation ``edge_index`` tensors are merged into a single untyped
``edge_index``, then split back per node type before the residual/LayerNorm.
That is literally "the same graph with the type information removed", which is
what the reviewer is asking about. It is also the only way to run a true
``GCNConv`` here, since ``GCNConv`` does not accept PyG's bipartite
``(src, dst)`` channel form and therefore cannot be placed inside
``HeteroConv``.

For ``architecture == "hgt"`` this class delegates to the unmodified
``HeteroLegalOutcomeGNN``, so it is provably a superset of the paper model
rather than a re-implementation of it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, HeteroConv, SAGEConv

_SECTION_GNN = Path(__file__).resolve().parents[3] / "section_GNN"
if str(_SECTION_GNN) not in sys.path:
    sys.path.insert(0, str(_SECTION_GNN))

from src.models.hetero_gnn import HeteroLegalOutcomeGNN  # noqa: E402
from src.models.mlp_head import MLPHead  # noqa: E402

# Message-passing operators applied to the type-collapsed (homogeneous) view.
COLLAPSED_ARCHITECTURES = {"gcn", "sage", "gat"}
# Message-passing operators that keep one set of weights per relation type.
RELATION_ARCHITECTURES = {"rgcn", "hgat"}
# No message passing at all.
NO_GRAPH_ARCHITECTURES = {"mlp"}
SUPPORTED_ARCHITECTURES = (
    COLLAPSED_ARCHITECTURES | RELATION_ARCHITECTURES | NO_GRAPH_ARCHITECTURES | {"hgt"}
)

# Descriptors used by the results table so the three axes stay in one place.
ARCHITECTURE_PROPERTIES: dict[str, dict[str, Any]] = {
    "mlp": {"label": "MLP (no graph)", "graph": False, "relation_aware": False, "attention": "none"},
    "gcn": {"label": "GCN", "graph": True, "relation_aware": False, "attention": "none"},
    "sage": {"label": "GraphSAGE", "graph": True, "relation_aware": False, "attention": "none"},
    "gat": {"label": "GAT", "graph": True, "relation_aware": False, "attention": "untyped"},
    "rgcn": {"label": "R-GCN (relational SAGE)", "graph": True, "relation_aware": True, "attention": "none"},
    "hgat": {"label": "Relational GAT", "graph": True, "relation_aware": True, "attention": "per-relation"},
    "hgt": {"label": "HGT (paper model)", "graph": True, "relation_aware": True, "attention": "typed"},
}


def _build_collapsed_conv(architecture: str, hidden_dim: int, num_heads: int) -> nn.Module:
    """One homogeneous message-passing layer over the type-collapsed graph."""
    if architecture == "gcn":
        # cached=True is safe: each fold builds a fresh model and sees one static graph.
        return GCNConv(hidden_dim, hidden_dim, cached=True, add_self_loops=True)
    if architecture == "sage":
        return SAGEConv(hidden_dim, hidden_dim)
    if architecture == "gat":
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        return GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, add_self_loops=True)
    raise ValueError(f"Not a collapsed architecture: {architecture}")


def _build_relation_conv(
    architecture: str,
    hidden_dim: int,
    num_heads: int,
    edge_types: list[tuple[str, str, str]],
) -> HeteroConv:
    """One relation-wise message-passing layer (one operator per edge type)."""
    if architecture == "rgcn":
        convs = {
            edge_type: SAGEConv((hidden_dim, hidden_dim), hidden_dim) for edge_type in edge_types
        }
    elif architecture == "hgat":
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")
        convs = {
            # add_self_loops must be False: PyG cannot add self-loops to a bipartite relation.
            edge_type: GATConv(
                (hidden_dim, hidden_dim),
                hidden_dim // num_heads,
                heads=num_heads,
                add_self_loops=False,
            )
            for edge_type in edge_types
        }
    else:
        raise ValueError(f"Not a relation-wise architecture: {architecture}")
    return HeteroConv(convs, aggr="sum")


class ArchLegalOutcomeGNN(nn.Module):
    """Same scaffold as the paper model; only the conv operator varies."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        out_dim: int,
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        architecture = str(cfg.get("architecture", "hgt")).lower()
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported GNN architecture: {architecture}. "
                f"Expected one of {sorted(SUPPORTED_ARCHITECTURES)}"
            )
        self.architecture = architecture

        # The paper model is used verbatim, not re-implemented.
        self._delegate: HeteroLegalOutcomeGNN | None = None
        if architecture == "hgt":
            self._delegate = HeteroLegalOutcomeGNN(
                metadata=metadata, input_dims=input_dims, out_dim=out_dim, cfg=cfg
            )
            self.metadata = metadata
            return

        self.metadata = metadata
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.num_heads = int(cfg.get("num_heads", 4))
        self.dropout = float(cfg.get("dropout", 0.2))

        self.input_projections = nn.ModuleDict(
            {node_type: nn.Linear(input_dims[node_type], self.hidden_dim) for node_type in self.node_types}
        )
        self.type_embeddings = nn.ParameterDict(
            {node_type: nn.Parameter(torch.zeros(1, self.hidden_dim)) for node_type in self.node_types}
        )
        self.layer_norms = nn.ModuleList(
            [
                nn.ModuleDict({node_type: nn.LayerNorm(self.hidden_dim) for node_type in self.node_types})
                for _ in range(self.num_layers)
            ]
        )

        if architecture in COLLAPSED_ARCHITECTURES:
            self.convs = nn.ModuleList(
                [
                    _build_collapsed_conv(architecture, self.hidden_dim, self.num_heads)
                    for _ in range(self.num_layers)
                ]
            )
        elif architecture in RELATION_ARCHITECTURES:
            self.convs = nn.ModuleList(
                [
                    _build_relation_conv(architecture, self.hidden_dim, self.num_heads, self.edge_types)
                    for _ in range(self.num_layers)
                ]
            )
        else:  # mlp -- a per-type linear stands in for the conv, so depth matches
            self.convs = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {node_type: nn.Linear(self.hidden_dim, self.hidden_dim) for node_type in self.node_types}
                    )
                    for _ in range(self.num_layers)
                ]
            )

        self.classifier = MLPHead(
            in_dim=self.hidden_dim,
            hidden_dim=int(cfg.get("mlp_hidden_dim", self.hidden_dim)),
            out_dim=out_dim,
            dropout=self.dropout,
        )
        # Cached type-collapsed edge index; rebuilt on the first forward pass.
        self._collapsed_edge_index: torch.Tensor | None = None
        self._collapsed_sizes: list[int] | None = None
        self.reset_parameters()

    # ------------------------------------------------------------------ utils

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_parameters(self) -> None:
        if self._delegate is not None:
            self._delegate.reset_parameters()
            return
        for projection in self.input_projections.values():
            projection.reset_parameters()
        for parameter in self.type_embeddings.values():
            nn.init.normal_(parameter, mean=0.0, std=0.02)
        for layer_norms in self.layer_norms:
            for layer_norm in layer_norms.values():
                layer_norm.reset_parameters()
        for conv in self.convs:
            if isinstance(conv, nn.ModuleDict):
                for linear in conv.values():
                    linear.reset_parameters()
            elif hasattr(conv, "reset_parameters"):
                conv.reset_parameters()
        for module in self.classifier.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _encode_inputs(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            node_type: self.input_projections[node_type](x) + self.type_embeddings[node_type]
            for node_type, x in x_dict.items()
        }

    def _collapse(
        self,
        hidden: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Merge the typed graph into one untyped graph (built once, then cached).

        The graph already went through ``ToUndirected``, so the merged edge index
        is symmetric and valid for ``GCNConv``'s normalisation.
        """
        sizes = [hidden[node_type].size(0) for node_type in self.node_types]
        if self._collapsed_edge_index is None or self._collapsed_sizes != sizes:
            offsets: dict[str, int] = {}
            running = 0
            for node_type, size in zip(self.node_types, sizes):
                offsets[node_type] = running
                running += size
            parts = []
            for (src_type, _, dst_type), edge_index in edge_index_dict.items():
                if edge_index.numel() == 0:
                    continue
                parts.append(
                    torch.stack(
                        (edge_index[0] + offsets[src_type], edge_index[1] + offsets[dst_type]),
                        dim=0,
                    )
                )
            merged = torch.cat(parts, dim=1) if parts else torch.zeros(2, 0, dtype=torch.long)
            self._collapsed_edge_index = merged.to(hidden[self.node_types[0]].device)
            self._collapsed_sizes = sizes
        flat = torch.cat([hidden[node_type] for node_type in self.node_types], dim=0)
        return flat, self._collapsed_edge_index, sizes

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._delegate is not None:
            return self._delegate(x_dict, edge_index_dict)

        hidden = self._encode_inputs(x_dict)
        for layer_idx, conv in enumerate(self.convs):
            residual = hidden

            if self.architecture in COLLAPSED_ARCHITECTURES:
                flat, collapsed_edge_index, sizes = self._collapse(residual, edge_index_dict)
                flat_out = conv(flat, collapsed_edge_index)
                chunks = torch.split(flat_out, sizes, dim=0)
                conv_out = dict(zip(self.node_types, chunks))
            elif self.architecture in RELATION_ARCHITECTURES:
                conv_out = conv(residual, edge_index_dict)
            else:  # mlp
                conv_out = {node_type: conv[node_type](value) for node_type, value in residual.items()}

            updated: dict[str, torch.Tensor] = {}
            for node_type in residual:
                message = conv_out.get(node_type, residual[node_type])
                message = F.dropout(message, p=self.dropout, training=self.training)
                updated[node_type] = self.layer_norms[layer_idx][node_type](residual[node_type] + message)
                updated[node_type] = F.relu(updated[node_type])
            hidden = updated

        logits = self.classifier(hidden["case"])
        return logits, hidden
