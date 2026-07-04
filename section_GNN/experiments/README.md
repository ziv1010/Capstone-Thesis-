# 🧫 experiments — Standalone Workflows

> Part of [`section_GNN/`](../README.md) · workflows that sit **beside** the main bucket
> matrix, reusing the core `src` package to answer narrower experimental questions.

## 🗂️ Subfolders

| Folder | Purpose |
|--------|---------|
| [`fixed_open_pipeline/`](fixed_open_pipeline/README.md) | Converts sentence-level fixed-open JSONs into the cleaned-case schema — the **standard preprocessing entry point**. |
| [`dataset_size_sweep/`](dataset_size_sweep/README.md) | Measures how performance scales with training-set size. |

## 🤔 When to Use This Folder

Use `experiments/` when a workflow is *not* simply "run a bucket config through
preprocess → build → train". For ordinary baseline and ablation runs, prefer
[`runs/`](../runs/README.md), [`runs_v2/`](../runs_v2/README.md), or
[`ablations/`](../ablations/README.md).

---

⬆️ Back to [`section_GNN/`](../README.md)
