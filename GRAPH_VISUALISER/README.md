# Graph Visualiser

Dash-based visual exploration tools for the legal case graph and entity-connectivity analyses.

## Main Scripts

- `build_graph.py`: builds graph artifacts from the configured case dataset.
- `app.py`: launches the interactive graph visualizer.
- `generate_plots.py` and `generate_plots_full.py`: export graph summary figures.
- `graph_stats.py`: computes graph-level statistics.
- `entity_analysis/`: entity-level analysis and plot export helpers.

## Running

```bash
bash GRAPH_VISUALISER/setup_env.sh
bash GRAPH_VISUALISER/run_build.sh
bash GRAPH_VISUALISER/run_app.sh
```

Generated graph pickles, layouts, plots, and cached dashboards are written under `outputs/` and are ignored by Git.
