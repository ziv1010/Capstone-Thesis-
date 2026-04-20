from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HGTConv, HeteroConv, SAGEConv

from src.models.mlp_head import MLPHead


class HeteroLegalOutcomeGNN(nn.Module):
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
            {node_type: nn.Linear(input_dims[node_type], self.hidden_dim) for node_type in metadata[0]}
        )
        self.type_embeddings = nn.ParameterDict(
            {node_type: nn.Parameter(torch.zeros(1, self.hidden_dim)) for node_type in metadata[0]}
        )
        self.layer_norms = nn.ModuleList(
            [nn.ModuleDict({node_type: nn.LayerNorm(self.hidden_dim) for node_type in metadata[0]}) for _ in range(self.num_layers)]
        )

        if self.architecture == "hgt":
            self.convs = nn.ModuleList(
                [
                    HGTConv(
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
                            edge_type: SAGEConv((self.hidden_dim, self.hidden_dim), self.hidden_dim)
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

    def _encode_inputs(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded: dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            encoded[node_type] = self.input_projections[node_type](x) + self.type_embeddings[node_type]
        return encoded

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict[Any, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self._encode_inputs(x_dict)
        for layer_idx, conv in enumerate(self.convs):
            residual = hidden
            conv_out = conv(hidden, edge_index_dict)
            updated: dict[str, torch.Tensor] = {}
            for node_type in residual:
                message = conv_out.get(node_type, residual[node_type])
                message = F.dropout(message, p=self.dropout, training=self.training)
                updated[node_type] = self.layer_norms[layer_idx][node_type](residual[node_type] + message)
                updated[node_type] = F.relu(updated[node_type])
            hidden = updated
        logits = self.classifier(hidden["case"])
        return logits, hidden
