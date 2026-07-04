# 🔬 ablations — Controlled Pipeline Variants

> Part of [`section_GNN/`](../README.md) · isolates **which information sources and graph
> design choices actually drive performance**.

Every ablation is a controlled variant of the baseline graph/model pipeline: one assumption
changes, everything else stays fixed.

## 🗂️ Common Layout

```text
ablations/<variant>/<bucket>/
  config.yaml
  run.sh
```

Config-only variants omit `run.sh` and are launched through top-level orchestration scripts.

## 🧪 Variant Matrix

| Variant | Question it isolates |
|---------|----------------------|
| [`text_only/`](text_only/README.md) | Is text encoding alone enough, without entity/authority structure? |
| [`no_names/`](no_names/README.md) | How much do identity/name-bearing nodes contribute? |
| [`no_cross_case/`](no_cross_case/README.md) | Does sharing authority nodes across cases help? |
| [`hierarchical_enc/`](hierarchical_enc/README.md) | Does hierarchical text encoding improve graph features? |
| [`section_sep_enc/`](section_sep_enc/README.md) | Do separate per-section embeddings help? |
| [`section_sep_enc_lr_decay/`](section_sep_enc_lr_decay/README.md) | Section-separated graph under the LR-decay schedule (thesis-table cells). |
| [`case_node_minimised/`](case_node_minimised/README.md) | How much can the case node's own features be reduced? |
| [`depth/`](depth/README.md) | Sensitivity to GNN depth / layer count. |
| [`entity_resolved_data/`](entity_resolved_data/README.md) | Effect of externally resolved (canonicalized) entities. |
| [`remove_central_authorities/`](remove_central_authorities/README.md) | Dependence on high-degree hub authorities. |

## ▶️ Running

```bash
# Single bucket:
bash ablations/text_only/cross_bucket_total_dataset/run.sh

# Larger groups:
bash run_scripts/run_complete_ablation_matrix.sh
bash run_scripts/run_remaining_non_cross_bucket_ablations.sh
```

## 📤 Outputs

Most ablation outputs land under `outputs/timed_bucket_runs/<bucket>/` or
`outputs/ablations/<variant>/` — the exact path is set by each `config.yaml`. Aggregate
results are collected in `outputs/master_ablation_results.csv`.

## ➕ Adding a New Ablation

1. Copy the closest existing variant config.
2. Keep `paths.*` relative to `section_GNN`.
3. Change **only** the fields needed for the ablation.
4. Add a short README documenting the new assumption.
5. Smoke-test one small bucket (or `--limit`) before launching the full matrix.

---

⬆️ Back to [`section_GNN/`](../README.md)
