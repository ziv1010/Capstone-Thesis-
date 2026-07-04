# 📤 outputs — Generated Visualiser Artifacts

> Part of [`GRAPH_VISUALISER/`](../README.md) · 📤 **generated data**, large artifacts
> ignored by Git.

Populated by `build_graph.py`, the plotting scripts, and `graph_stats.py`. Typical contents:

| Artifact | Produced by |
|----------|-------------|
| `graph_full.pkl` / `graph_full.gexf` | Full NetworkX graph (+ GEXF export) — `build_graph.py` |
| `graph_sample.pkl` · `layout.pkl` · `case_layout.pkl` · `case_connections.pkl` | Connectivity-ranked display sample + layouts — `build_graph.py` |
| `stats.json` · `graph_stats/` | Full-graph statistics — `graph_stats.py` |
| `plots/` · `plots_full/` | Static figures — `generate_plots.py` / `generate_plots_full.py` |
| `leakage_role_comparison/` | Rhetorical-role before/after leakage filtering — `generate_rhetorical_before_after.py` |
| `thesis_figures/` · `thesis_extras/` | Curated thesis figure collections |
| `logs/` | Build/plot logs |

## ♻️ Regenerate

```bash
bash GRAPH_VISUALISER/run_build.sh
micromamba run -n graph_vis python GRAPH_VISUALISER/generate_plots.py --config GRAPH_VISUALISER/config.yaml
micromamba run -n graph_vis python GRAPH_VISUALISER/graph_stats.py --config GRAPH_VISUALISER/config.yaml
```

---

⬆️ Back to [`GRAPH_VISUALISER/`](../README.md)
