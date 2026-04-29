"""A PGExplainer-style explainer for a frozen heterogeneous HGT.

PyG's stock `PGExplainer` targets homogeneous graphs and expects a model that
accepts `edge_weight` as an argument. Our HGT does not. This module implements
the same core idea (a shared MLP that takes edge-endpoint embeddings and
produces a single logit per edge) but evaluates the perturbed prediction by
*selecting* edges via a Gumbel-sampled mask, so it works natively with the
existing HGTConv stack.

Sketch:
  * For each edge (u -> v) of each relation r, take:
      [h_u || h_v || relation_embedding(r)]
    where h_u, h_v are the frozen, layer-0-encoded node features.
  * A shared MLP maps that to a logit. Add per-edge Gumbel noise and pass
    through a sigmoid to get a soft mask.
  * Hard-sample the edge subset via straight-through Gumbel, drop masked-out
    edges from edge_index_dict, run the frozen HGT, and minimise a
    cross-entropy loss against the model's *original* predicted label, plus
    sparsity / entropy regularisers.

At inference time, the MLP's raw sigmoid outputs are used directly as per-edge
importance scores (no sampling noise).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class HeteroEdgeMaskerConfig:
    hidden_dim: int = 64
    temperature_start: float = 5.0
    temperature_end: float = 1.0
    coeff_size: float = 0.01
    coeff_ent: float = 1.0e-4


class HeteroEdgeMLP(nn.Module):
    """Shared MLP that scores every edge regardless of relation.

    Input: concat(h_src, h_dst, relation_embedding). The per-edge embeddings
    `h_*` come from the *frozen* HGT's input-projection layer (i.e. the hidden
    representation before any message passing), which keeps the explainer
    cheap and type-agnostic.
    """

    def __init__(
        self,
        node_hidden_dim: int,
        num_relations: int,
        mlp_hidden_dim: int = 64,
        relation_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(num_relations, relation_embedding_dim)
        in_dim = 2 * node_hidden_dim + relation_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
        relation_idx: int,
    ) -> torch.Tensor:
        if src_emb.numel() == 0:
            return src_emb.new_zeros(0)
        rel_id = torch.full(
            (src_emb.shape[0],),
            relation_idx,
            dtype=torch.long,
            device=src_emb.device,
        )
        rel_emb = self.relation_embedding(rel_id)
        edge_repr = torch.cat([src_emb, dst_emb, rel_emb], dim=-1)
        return self.net(edge_repr).squeeze(-1)


def gumbel_sigmoid_hard(
    logits: torch.Tensor,
    temperature: float,
    training: bool,
) -> torch.Tensor:
    """Straight-through Gumbel sigmoid.

    Returns a {0,1}-valued tensor in the forward pass but lets gradients flow
    through the soft sigmoid in the backward pass.
    """
    if not training:
        return (torch.sigmoid(logits) > 0.5).float()
    eps = 1.0e-7
    u1 = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    u2 = torch.rand_like(logits).clamp(eps, 1.0 - eps)
    gumbel_noise = torch.log(u1) - torch.log(u2)
    soft = torch.sigmoid((logits + gumbel_noise) / max(temperature, 1.0e-6))
    hard = (soft > 0.5).float()
    return hard + (soft - soft.detach())


class HeteroPGExplainer:
    """PGExplainer for a frozen heterogeneous HGT.

    The explainer is *trained on the training-split case nodes* (i.e. it learns
    what edges matter for the HGT's *predicted* label for those cases), and
    then used at test time to score edges for unseen cases.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Any,
        masker_cfg: HeteroEdgeMaskerConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.data = data
        self.device = torch.device(device)
        self.cfg = masker_cfg or HeteroEdgeMaskerConfig()
        self.edge_types = list(data.edge_types)
        self.relation_to_idx = {rel: i for i, rel in enumerate(self.edge_types)}
        node_hidden_dim = next(self.model.input_projections.parameters()).shape[0]
        self.masker = HeteroEdgeMLP(
            node_hidden_dim=node_hidden_dim,
            num_relations=len(self.edge_types),
            mlp_hidden_dim=self.cfg.hidden_dim,
        ).to(self.device)
        # Cache the frozen layer-0 encodings that the masker MLP consumes.
        with torch.no_grad():
            self._node_base_emb = {
                nt: self.model._encode_inputs(data.x_dict)[nt].detach()
                for nt in data.node_types
            }

    # ---------- internals ----------

    def _edge_logits(self, edge_type: tuple[str, str, str]) -> torch.Tensor:
        src_type, _relation, dst_type = edge_type
        edge_index = self.data[edge_type].edge_index
        if edge_index.numel() == 0:
            return edge_index.new_zeros(0, dtype=torch.float32)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        src_emb = self._node_base_emb[src_type][src_idx]
        dst_emb = self._node_base_emb[dst_type][dst_idx]
        return self.masker(src_emb, dst_emb, self.relation_to_idx[edge_type])

    def _masked_edge_index_dict(
        self, edge_masks: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        masked: dict[tuple[str, str, str], torch.Tensor] = {}
        for edge_type, edge_index in self.data.edge_index_dict.items():
            m = edge_masks.get(edge_type)
            if m is None or edge_index.numel() == 0:
                masked[edge_type] = edge_index
                continue
            keep = m > 0.5
            if keep.dtype != torch.bool:
                keep = keep.bool()
            masked[edge_type] = edge_index[:, keep]
        return masked

    # ---------- public API ----------

    def train(
        self,
        target_case_indices: list[int] | torch.Tensor,
        epochs: int = 30,
        lr: float = 3.0e-3,
        cases_per_epoch: int | None = None,
        log_every: int = 1,
    ) -> dict[str, Any]:
        device = self.device
        self.masker.train()
        optimizer = torch.optim.Adam(self.masker.parameters(), lr=lr)

        if isinstance(target_case_indices, list):
            target_case_indices = torch.tensor(target_case_indices, dtype=torch.long, device=device)
        else:
            target_case_indices = target_case_indices.to(device=device, dtype=torch.long)

        with torch.no_grad():
            logits_orig, _ = self.model(self.data.x_dict, self.data.edge_index_dict)
            target_labels = logits_orig.argmax(dim=-1).detach()

        history: list[dict[str, Any]] = []
        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            temperature = (
                self.cfg.temperature_start
                + (self.cfg.temperature_end - self.cfg.temperature_start) * progress
            )

            if cases_per_epoch is not None and cases_per_epoch < target_case_indices.numel():
                perm = torch.randperm(target_case_indices.numel(), device=device)[:cases_per_epoch]
                batch_idx = target_case_indices[perm]
            else:
                batch_idx = target_case_indices

            optimizer.zero_grad()

            edge_masks: dict[tuple[str, str, str], torch.Tensor] = {}
            size_terms: list[torch.Tensor] = []
            ent_terms: list[torch.Tensor] = []
            for edge_type in self.edge_types:
                logit = self._edge_logits(edge_type)
                if logit.numel() == 0:
                    edge_masks[edge_type] = logit
                    continue
                mask = gumbel_sigmoid_hard(logit, temperature=temperature, training=True)
                edge_masks[edge_type] = mask
                probs = torch.sigmoid(logit)
                size_terms.append(probs.mean())
                ent = -(probs * torch.log(probs + 1.0e-9) + (1 - probs) * torch.log(1 - probs + 1.0e-9))
                ent_terms.append(ent.mean())

            masked_edge_index_dict = self._masked_edge_index_dict(edge_masks)
            logits_pert, _ = self.model(self.data.x_dict, masked_edge_index_dict)
            target_slice = target_labels[batch_idx]
            pred_loss = F.cross_entropy(logits_pert[batch_idx], target_slice)
            size_loss = torch.stack(size_terms).mean() if size_terms else torch.tensor(0.0, device=device)
            ent_loss = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=device)
            loss = pred_loss + self.cfg.coeff_size * size_loss + self.cfg.coeff_ent * ent_loss
            loss.backward()
            optimizer.step()

            record = {
                "epoch": epoch,
                "temperature": float(temperature),
                "pred_loss": float(pred_loss.item()),
                "size_loss": float(size_loss.item()),
                "ent_loss": float(ent_loss.item()),
                "total_loss": float(loss.item()),
            }
            history.append(record)
            if log_every and epoch % log_every == 0:
                print(
                    f"[pg_explainer] epoch={epoch} loss={record['total_loss']:.4f} "
                    f"pred={record['pred_loss']:.4f} size={record['size_loss']:.4f} "
                    f"ent={record['ent_loss']:.4f} T={temperature:.2f}",
                    flush=True,
                )
        return {"history": history}

    @torch.no_grad()
    def edge_importance(self) -> dict[tuple[str, str, str], torch.Tensor]:
        """Return sigmoid(edge_logit) for every edge in every relation."""
        self.masker.eval()
        out: dict[tuple[str, str, str], torch.Tensor] = {}
        for edge_type in self.edge_types:
            logit = self._edge_logits(edge_type)
            out[edge_type] = torch.sigmoid(logit).detach().cpu()
        return out

    def save(self, path: str) -> None:
        state = {
            "masker_state_dict": self.masker.state_dict(),
            "cfg": self.cfg.__dict__,
            "edge_types": self.edge_types,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.masker.load_state_dict(state["masker_state_dict"])
        self.masker.to(self.device)
