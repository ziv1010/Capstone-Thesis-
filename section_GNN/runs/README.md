# 🏁 runs — Baseline BGE-M3 Timed-Bucket Experiments

> Part of [`section_GNN/`](../README.md) · the **baseline** experiment matrix.

Each bucket folder has the same three-step shape:

```text
runs/<bucket>/
  config.yaml           # full pipeline configuration
  01_preprocess.sh      # raw JSONs → cleaned cases
  02_build_graph.sh     # cleaned cases → graph cache + embeddings
  03_kfold_8gpu.sh      # 8-GPU K-fold training/evaluation
  run_all.sh            # all three steps in sequence
```

## 🪣 Buckets

`family_matrimonial_timed_mistral` · `fin_fraud_timed_mistral` · `land_property_timed_mistral`
· `motor_accidents_timed_mistral` · `sexual_offences_timed_mistral` ·
`cross_bucket_total_dataset`

## ▶️ Run One Bucket

From `section_GNN/`:

```bash
bash runs/fin_fraud_timed_mistral/run_all.sh
# or step by step:
bash runs/fin_fraud_timed_mistral/01_preprocess.sh
bash runs/fin_fraud_timed_mistral/02_build_graph.sh
bash runs/fin_fraud_timed_mistral/03_kfold_8gpu.sh
```

## 📤 Outputs

```text
data/timed_bucket_runs/<bucket>/                                   # cleaned cases, caches, audits
outputs/timed_bucket_runs/<bucket>/models/<run_name>/kfold/        # per-fold artifacts
outputs/timed_bucket_runs/<bucket>/models/<run_name>/kfold/kfold_summary.json   # headline metrics
```

## 🤝 Cross-Bucket Group Helpers

| Script | Effect |
|--------|--------|
| `run_hierarchical_enc_all_buckets.sh` | Runs the hierarchical-encoding ablation over all buckets. |
| `run_section_sep_enc_all_buckets.sh` | Runs the section-separated encoding ablation over all buckets. |

---

⬆️ Back to [`section_GNN/`](../README.md)
