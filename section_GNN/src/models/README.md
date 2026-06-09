# src/models

This package defines the neural model layers used by the training scripts.

## Files

- `hetero_gnn.py`: heterogeneous GNN classifier. The default architecture uses
  HGT-style message passing over PyG `HeteroData` metadata.
- `mlp_head.py`: small MLP classifier head used on the final case-node
  representation.

## Inputs and Outputs

The model receives:

- `x_dict`: node features by node type
- `edge_index_dict`: edge indices by relation type

It predicts logits for `case` nodes. Training/evaluation scripts map those case
node logits back to labels and case IDs using graph metadata.
