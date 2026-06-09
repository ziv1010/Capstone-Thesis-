# party_args_lr_decay

Shared v2 run family for party-argument case-node text with LR-decay training.

## Files

- `graph/build_graph_v2.py`: graph builder for the v2 case-node text policy.
- `scripts/kfold_cv_v2.py`: K-fold trainer with LR-decay settings.
- `03_kfold_v2.sh`: shell wrapper for parallel K-fold runs.
- `run_all_buckets.sh`: runs the v2 pipeline across all buckets.
- `<bucket>/config.yaml`: per-bucket configs.

## Run All Buckets

```bash
bash runs_v2/party_args_lr_decay/run_all_buckets.sh
```

## Run One K-Fold Job

```bash
bash runs_v2/party_args_lr_decay/03_kfold_v2.sh \
  --config runs_v2/party_args_lr_decay/cross_bucket_total_dataset/config.yaml \
  --run-name cross_bucket_party_args_lr_decay_kfold
```
