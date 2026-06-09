# experiments

`experiments` contains standalone workflows that sit beside the main bucket
matrix. They reuse the core `src` package but answer narrower experimental
questions.

## Subfolders

| Folder | Purpose |
| --- | --- |
| `fixed_open_pipeline/` | Converts sentence-level fixed-open JSONs into cleaned cases for the reasoning graph. |
| `dataset_size_sweep/` | Measures performance as training set size changes. |
| `multi_embed_test/` | Compares text encoder backends and evaluates checkpoint transfer. |

## When to Use This Folder

Use `experiments/` when a workflow is not simply "run a bucket config through
preprocess/build/train." For ordinary baseline and ablation runs, prefer
`runs/`, `runs_v2/`, or `ablations/`.
