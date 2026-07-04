# 📑 section_sep_enc — Section-Separated Encoding Ablation

> Part of [`ablations/`](../README.md).

Keeps **separate embeddings for each major case section** (preamble, facts, arguments,
party-specific arguments) instead of collapsing them into a single case text representation —
testing whether preserving section distinctions helps the model.

## 🏗️ Builders

This graph path uses `final_graph/build_graph_section_sep.py` and
`src/graph/pyg_builder_section_sep.py`.

## ▶️ Run

```bash
# One bucket:
bash ablations/section_sep_enc/cross_bucket_total_dataset/run.sh

# All buckets:
bash runs/run_section_sep_enc_all_buckets.sh
```

The LR-decay counterpart (used in the final thesis tables and by the entity-resolved
explanation model) is [`section_sep_enc_lr_decay/`](../section_sep_enc_lr_decay/README.md).

---

⬆️ Back to [`ablations/`](../README.md)
