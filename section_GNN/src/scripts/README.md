# ⌨️ src/scripts — Command-Line Entry Points

> Part of [`section_GNN/src/`](../README.md) · run all commands from `section_GNN/`.

## 🚀 Main Commands

```bash
# Build a graph cache
micromamba run -n thesis_work python src/scripts/build_graph.py \
  --config runs/cross_bucket_total_dataset/config.yaml

# Train a single run on an existing graph cache
micromamba run -n thesis_work python src/scripts/train_gnn.py \
  --config runs/cross_bucket_total_dataset/config.yaml --run-name cross_bucket_run

# K-fold cross-validation
micromamba run -n thesis_work python src/scripts/kfold_cv.py \
  --config runs/cross_bucket_total_dataset/config.yaml --run-name cross_bucket_kfold

# Evaluate a saved checkpoint on a graph
micromamba run -n thesis_work python src/scripts/evaluate_saved_model.py \
  --config runs/cross_bucket_total_dataset/config.yaml \
  --checkpoint outputs/timed_bucket_runs/cross_bucket_total_dataset/models/<run>/kfold/fold_00/model.pt \
  --output-dir outputs/manual_eval
```

## 🔍 Audit & Analysis Helpers

| Script | Purpose |
|--------|---------|
| `audit_precedent_case_resolution.py` | Checks precedent → case resolution quality. |
| `analyze_shared_node_connectivity.py` | Summarizes how shared nodes connect to labels. |
| `export_authority_cases_all_buckets.py` | Exports cases neighbouring a named authority node. |
| `export_authority_node_degrees.py` | Exports authority-node degree tables. |

Run from `section_GNN/` so relative config and output paths resolve cleanly.

---

⬆️ Back to [`src/`](../README.md)
