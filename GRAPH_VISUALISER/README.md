# Graph Visualiser

Dash-based visual exploration tools for the final legal case graph and entity-connectivity analyses.

## Data Source

The visualiser now points to the final entity-resolved Timeline Maker output:

```text
../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/
```

`config.yaml` uses repo-relative paths for these five buckets:

- `fin_fraud`
- `family_matrimonial`
- `land_property`
- `motor_accidents`
- `sexual_offences`

The path resolver in `path_utils.py` interprets relative paths from the location of `config.yaml`, so scripts work whether they are launched from the repo root or from `GRAPH_VISUALISER/`.

## Main Files

- `config.yaml`: bucket list, final data paths, graph policy, sampling limits, colours, and Dash settings.
- `path_utils.py`: shared path resolver for config, data, and output paths.
- `build_graph.py`: builds the NetworkX graph artifacts from the final resolved JSONs.
- `app.py`: launches the main interactive Dash graph visualiser.
- `generate_plots.py`: exports static figures from the sampled graph artifacts.
- `generate_plots_full.py`: exports full-dataset descriptive figures directly from the final JSON buckets.
- `graph_stats.py`: computes full-graph statistics and graph-stat figures.
- `generate_rhetorical_before_after.py`: compares rhetorical-role distributions before and after leakage filtering.
- `entity_analysis/`: separate within-bucket and cross-bucket entity co-occurrence analysis and Dash app.

## Running

Set up the environment once:

```bash
bash GRAPH_VISUALISER/setup_env.sh
```

Build the graph artifacts:

```bash
bash GRAPH_VISUALISER/run_build.sh
```

For a quick smoke test:

```bash
bash GRAPH_VISUALISER/run_build.sh --limit 500 --skip-layout
```

Launch the main Dash app:

```bash
bash GRAPH_VISUALISER/run_app.sh 8050
```

If the environment name is different, override it without editing scripts:

```bash
MAMBA_ENV=my_env bash GRAPH_VISUALISER/run_app.sh 8050
```

## Static Outputs

Generated graph pickles, layouts, plots, logs, and thesis figures are written under:

```text
GRAPH_VISUALISER/outputs/
```

Common commands:

```bash
micromamba run -n graph_vis python GRAPH_VISUALISER/generate_plots.py --config GRAPH_VISUALISER/config.yaml
micromamba run -n graph_vis python GRAPH_VISUALISER/generate_plots_full.py --config GRAPH_VISUALISER/config.yaml
micromamba run -n graph_vis python GRAPH_VISUALISER/graph_stats.py --config GRAPH_VISUALISER/config.yaml
```

## Entity Analysis App

The entity-analysis sub-app also reads from the final resolved Timeline Maker output and writes to:

```text
GRAPH_VISUALISER/entity_analysis/outputs/
```

Run both within-bucket and cross-bucket analysis, then launch the app:

```bash
bash GRAPH_VISUALISER/entity_analysis/run.sh both 8052
```

Use analysis-only mode when you only need refreshed JSON outputs:

```bash
bash GRAPH_VISUALISER/entity_analysis/run.sh analyse-only
```
