from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def is_pyg_available() -> bool:
    try:
        import torch_geometric  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class PyGTrainResult:
    model_state: dict[str, Any]
    y_pred_case: np.ndarray
    y_proba_case: np.ndarray


def train_gcn_with_pyg(
    x: np.ndarray,
    edge_index: np.ndarray,
    y_case: np.ndarray,
    case_node_indices: np.ndarray,
    split_name_by_case: list[str],
    cfg: dict[str, Any],
    seed: int,
    logger: Any | None = None,
) -> PyGTrainResult:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv

    torch.manual_seed(seed)

    class GCN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
            self.dropout = dropout

        def forward(self, x_t: torch.Tensor, edge_index_t: torch.Tensor) -> torch.Tensor:
            h = self.conv1(x_t, edge_index_t)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.conv2(h, edge_index_t)
            return out

    n_nodes = x.shape[0]
    n_classes = int(np.max(y_case)) + 1

    y_all = np.full((n_nodes,), -1, dtype=np.int64)
    y_all[case_node_indices] = y_case

    train_mask = np.zeros((n_nodes,), dtype=bool)
    val_mask = np.zeros((n_nodes,), dtype=bool)
    test_mask = np.zeros((n_nodes,), dtype=bool)

    for case_idx, node_idx in enumerate(case_node_indices):
        split = split_name_by_case[case_idx]
        if split == "train":
            train_mask[node_idx] = True
        elif split == "val":
            val_mask[node_idx] = True
        elif split == "test":
            test_mask[node_idx] = True

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y_all, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
    )

    device = "cuda" if torch.cuda.is_available() and bool(cfg.get("use_cuda_if_available", True)) else "cpu"
    data = data.to(device)

    model = GCN(
        in_dim=x.shape[1],
        hidden_dim=int(cfg.get("hidden_dim", 128)),
        out_dim=n_classes,
        dropout=float(cfg.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(cfg.get("epochs", 100))
    best_state = None
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(data.x, data.edge_index)
            pred = logits_eval.argmax(dim=-1)
            if int(data.val_mask.sum()) > 0:
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            else:
                val_acc = 0.0

        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if logger is not None and epoch % max(1, epochs // 10) == 0:
            logger.info("[PyG] epoch=%d/%d loss=%.4f val_acc=%.4f", epoch, epochs, float(loss.item()), val_acc)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_final = model(data.x, data.edge_index)
        probs_final = torch.softmax(logits_final, dim=-1).cpu().numpy()
        pred_final = logits_final.argmax(dim=-1).cpu().numpy()

    case_pred = pred_final[case_node_indices]
    case_proba = probs_final[case_node_indices]

    return PyGTrainResult(
        model_state={k: v.cpu() for k, v in model.state_dict().items()},
        y_pred_case=case_pred,
        y_proba_case=case_proba,
    )
