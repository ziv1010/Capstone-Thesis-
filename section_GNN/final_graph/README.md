# 🏗️ final_graph — Reasoning-Focused Graph Builders

> Part of [`section_GNN/`](../README.md) · the graph builders behind the later thesis
> experiments — trimming noisy context from the older case-star graph while keeping
> everything needed for legal outcome prediction.

## 📄 Files

| File | Role |
|------|------|
| `build_graph.py` | Reasoning-focused graph build entry point. |
| `build_graph_section_sep.py` | Section-separated graph build entry point. |
| `graph_config_template.yaml` | Graph-section template for dataset configs. |
| `visualize_graph_structure.py` | Graph-structure visualisations. |
| [`updated_graph/`](updated_graph/README.md) | Implementation package for the reasoning-focused builder. |

## ⚖️ Reasoning-Focused Policy

**Keeps:** case→section edges · case→party/court/judge/lawyer edges ·
argument→statute/provision/precedent citation edges · provision→statute membership edges ·
party/lawyer→party-argument edges.

**Removes:** noisy context nodes (`org`, `gpe`, `date`, `case_number`) and shortcut edges
that over-connect argument nodes.

This same policy is mirrored by the [Graph Visualiser](../../GRAPH_VISUALISER/README.md)
so the interactive views match what the GNN actually sees.

## ▶️ Build Examples

From `section_GNN/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 micromamba run -n thesis_work python final_graph/build_graph.py \
  --config runs/cross_bucket_total_dataset/config.yaml

# Section-separated features:
CUDA_VISIBLE_DEVICES=0,1 micromamba run -n thesis_work python final_graph/build_graph_section_sep.py \
  --config ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml
```

Graph caches + metadata land under each config's `paths.graph_cache_dir`.

---

⬆️ Back to [`section_GNN/`](../README.md)
