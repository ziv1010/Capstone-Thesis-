# Entity Analysis Visualiser

Within-bucket and cross-bucket entity co-occurrence analysis for the final resolved Timeline Maker dataset.

## Input

`analyse.py` reads final JSONs from:

```text
../../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/
```

The active buckets are:

- `family_matrimonial`
- `land_property`
- `motor_accidents`
- `sexual_offences`
- `fin_fraud`

Cross-bucket analysis uses:

```text
../../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/combined_dataset_without_food_safety/
```

## Outputs

Generated JSONs and figures are written under:

```text
GRAPH_VISUALISER/entity_analysis/outputs/
```

Important subfolders:

- `within_bucket/`: per-domain entity graph summaries.
- `cross_bucket/`: shared cross-domain entity graph summaries.
- `figures/`: exported Plotly figures.
- `figures_readable/`: reduced, thesis-readable figure exports.

## Run

From the repo root:

```bash
bash GRAPH_VISUALISER/entity_analysis/run.sh analyse-only
```

Run analysis and launch the Dash app:

```bash
bash GRAPH_VISUALISER/entity_analysis/run.sh both 8052
```

Export static figures after analysis:

```bash
micromamba run -n graph_vis python GRAPH_VISUALISER/entity_analysis/export_plots.py
micromamba run -n graph_vis python GRAPH_VISUALISER/entity_analysis/export_plots_readable.py
```
