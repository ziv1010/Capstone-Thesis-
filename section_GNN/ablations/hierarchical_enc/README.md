# hierarchical_enc

The hierarchical-encoding ablation changes how case text is represented before
GNN training.

## Purpose

Rather than treating all text as one flat representation, this variant tests
whether a more structured text encoding improves downstream graph performance.

## Run Example

```bash
bash ablations/hierarchical_enc/cross_bucket_total_dataset/run.sh
```

There is also a group launcher:

```bash
bash runs/run_hierarchical_enc_all_buckets.sh
```
