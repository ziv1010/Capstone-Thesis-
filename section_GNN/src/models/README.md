# 🤖 src/models — Model Definitions

> Part of [`section_GNN/src/`](../README.md).

Neural model layers used by the training scripts.

| File | Role |
|------|------|
| `hetero_gnn.py` | Heterogeneous GNN classifier — the default architecture uses **HGT-style message passing** over PyG `HeteroData` metadata (relation-aware attention across all node/edge types). |
| `mlp_head.py` | Small MLP classifier head applied to the final `case`-node representation. |

## 🔁 Contract

**Inputs:** `x_dict` (node features by node type) and `edge_index_dict` (edge indices by
relation type).
**Output:** logits for `case` nodes. Training/evaluation scripts map case-node logits back to
labels and case IDs through the graph metadata, so predictions stay traceable to source files.

---

⬆️ Back to [`src/`](../README.md)
