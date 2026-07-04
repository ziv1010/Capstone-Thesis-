# ⚙️ party_args_lr_decay — Shared V2 Run Family

> Part of [`runs_v2/`](../README.md) · party-argument case-node text + LR-decay training.
> Also hosts the **shared v2 builder/trainer** used by the sibling v2 folders.

## 📄 Files

| File | Role |
|------|------|
| `graph/build_graph_v2.py` | Graph builder implementing the v2 case-node text policy. |
| `scripts/kfold_cv_v2.py` | K-fold trainer with LR-decay settings. |
| `03_kfold_v2.sh` | Shell wrapper for parallel K-fold runs. |
| `run_all_buckets.sh` | Runs the v2 pipeline across all buckets. |
| `<bucket>/config.yaml` | Per-bucket configurations. |

## ▶️ Run

```bash
# All buckets:
bash runs_v2/party_args_lr_decay/run_all_buckets.sh

# One K-fold job:
bash runs_v2/party_args_lr_decay/03_kfold_v2.sh \
  --config runs_v2/party_args_lr_decay/cross_bucket_total_dataset/config.yaml \
  --run-name cross_bucket_party_args_lr_decay_kfold
```

---

⬆️ Back to [`runs_v2/`](../README.md)
