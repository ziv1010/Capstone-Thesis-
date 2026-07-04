# 🏛️ runs_inlegalbert — InLegalBERT Comparison Matrix

> Part of [`section_GNN/`](../README.md) · mirrors the main BGE-M3 matrix with
> **InLegalBERT** text features, keeping graph variants aligned for a fair encoder comparison.

## 🗂️ Variant Folders

`baseline/` · `baseline_lr_decay/` · `party_args_no_lr/` · `party_args_lr_decay/` ·
`text_only/` · `no_names/` · `no_cross_case/` · `hierarchical_enc/` · `section_sep_enc/` ·
`case_node_minimised/`

Each variant folder contains per-bucket `config.yaml` files; the matrix is orchestrated by
the top-level launcher:

```bash
bash run_scripts/run_inlegalbert_experiments.sh
```

## 📤 Outputs

```text
data/inlegalbert_runs/
outputs/inlegalbert_runs/
```

The headline BGE-M3 vs InLegalBERT table is assembled by
`summarize_bge_vs_inlegalbert.py` into `outputs/inlegalbert_vs_bge_comparison.csv`.

---

⬆️ Back to [`section_GNN/`](../README.md)
