# outputs

Generated graph visualiser artifacts.

This folder is populated by `GRAPH_VISUALISER/build_graph.py`, plotting scripts,
and graph-stat analysis scripts. Typical contents include graph pickles, GEXF
exports, layouts, static plots, graph statistics, thesis figure collections, and
logs.

Regenerate outputs from:

```bash
bash GRAPH_VISUALISER/run_build.sh
micromamba run -n graph_vis python GRAPH_VISUALISER/generate_plots.py --config GRAPH_VISUALISER/config.yaml
micromamba run -n graph_vis python GRAPH_VISUALISER/graph_stats.py --config GRAPH_VISUALISER/config.yaml
```

Large generated artifacts are ignored by Git.
