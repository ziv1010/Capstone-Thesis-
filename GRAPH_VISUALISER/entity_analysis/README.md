# 🔗 entity_analysis — Entity Co-Occurrence Explorer (Port 8052)

> Part of [`GRAPH_VISUALISER/`](../README.md) · within-bucket and cross-bucket entity
> co-occurrence analysis over the final resolved Timeline Maker dataset.

## 📥 Input

`analyse.py` reads the final resolved JSONs:

```text
../../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/
```

Active buckets: `family_matrimonial`, `land_property`, `motor_accidents`,
`sexual_offences`, `fin_fraud`. Cross-bucket analysis uses
`output_merged_v3_resolved/combined_dataset_without_food_safety/`.

## 📤 Outputs

Written under `GRAPH_VISUALISER/entity_analysis/outputs/`:

| Subfolder | Contents |
|-----------|----------|
| `within_bucket/` | Per-domain entity graph summaries. |
| `cross_bucket/` | Shared cross-domain entity graph summaries. |
| `figures/` | Exported Plotly figures. |
| `figures_readable/` | Reduced, thesis-readable figure exports. |

## ▶️ Run

From the repository root:

```bash
# Analysis only (refresh JSON outputs):
bash GRAPH_VISUALISER/entity_analysis/run.sh analyse-only

# Analysis + Dash app on port 8052:
bash GRAPH_VISUALISER/entity_analysis/run.sh both 8052

# Static figure exports:
micromamba run -n graph_vis python GRAPH_VISUALISER/entity_analysis/export_plots.py
micromamba run -n graph_vis python GRAPH_VISUALISER/entity_analysis/export_plots_readable.py
```

---

⬆️ Back to [`GRAPH_VISUALISER/`](../README.md)
