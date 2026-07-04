# 🧩 updated_graph — Reasoning-Graph Implementation

> Part of [`final_graph/`](../README.md) · implementation package, not an entry point.

| File | Role |
|------|------|
| `reasoning_graph_policy.py` | Defines which node/edge types the reasoning-focused graph keeps or removes. |
| `case_star_builder.py` | Builds local case-star records under that policy. |
| `pipeline.py` | Assembles cleaned cases, labels, splits, and PyG conversion into a graph bundle. |

## 🚪 Entry Points

Always use the scripts one level up so config loading, logging, and cache snapshots stay
consistent:

```bash
python final_graph/build_graph.py --config runs/cross_bucket_total_dataset/config.yaml
python final_graph/build_graph_section_sep.py --config ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml
```

Do not import this package from shell scripts directly.

---

⬆️ Back to [`final_graph/`](../README.md)
