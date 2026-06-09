# src/scripts

Command-line entry points for the reusable `src` pipeline live here.

## Main Commands

Build a graph:

```bash
micromamba run -n thesis_work python src/scripts/build_graph.py \
  --config runs/cross_bucket_total_dataset/config.yaml
```

Train one run on an existing graph cache:

```bash
micromamba run -n thesis_work python src/scripts/train_gnn.py \
  --config runs/cross_bucket_total_dataset/config.yaml \
  --run-name cross_bucket_run
```

Run K-fold CV:

```bash
micromamba run -n thesis_work python src/scripts/kfold_cv.py \
  --config runs/cross_bucket_total_dataset/config.yaml \
  --run-name cross_bucket_kfold
```

Evaluate a saved model on a graph:

```bash
micromamba run -n thesis_work python src/scripts/evaluate_saved_model.py \
  --config runs/cross_bucket_total_dataset/config.yaml \
  --checkpoint outputs/timed_bucket_runs/cross_bucket_total_dataset/models/<run>/kfold/fold_00/model.pt \
  --output-dir outputs/manual_eval
```

## Helper Scripts

- `audit_precedent_case_resolution.py`: checks precedent/case resolution quality.
- `analyze_shared_node_connectivity.py`: summarizes how shared nodes connect to labels.
- `export_authority_cases_all_buckets.py`: exports cases near a named authority node.
- `export_authority_node_degrees.py`: exports authority-node degree tables.

Run these from `section_GNN` so relative config paths and output paths resolve
cleanly.
