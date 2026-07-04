# 🪜 hierarchical_enc — Hierarchical Text Encoding Ablation

> Part of [`ablations/`](../README.md).

Changes how case text is represented before GNN training: instead of one flat text
representation, sections are encoded with a **more structured, hierarchical strategy** —
testing whether richer text encoding improves downstream graph performance.

## ▶️ Run

```bash
# One bucket:
bash ablations/hierarchical_enc/cross_bucket_total_dataset/run.sh

# All buckets:
bash runs/run_hierarchical_enc_all_buckets.sh
```

---

⬆️ Back to [`ablations/`](../README.md)
