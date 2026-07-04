# 🌐 GRAPH_VISUALISER — Extra Interactive Graph Explorer (Port 8050)

> ➕ The repository's **extra** visualiser — a Dash app for exploring the final legal case
> graph, plus static plotting/statistics tools for thesis figures.
> *(The two main visualisers are the [Multi-Hearing Stage Test Visualiser](../section_GNN/multi_hearing_stage_test/visualiser/README.md) on port 8050 and the [Final Explanation Visualizer](../FINAL_EXPLANATION/README.md) on port 8899.)*

An interactive, **GNN-aligned** view of the case-entity graph: the graph policy in
`config.yaml` mirrors `section_GNN`'s `reasoning_graph_policy`, so what you explore here is
structurally what the model was trained on.

---

## 🗄️ Data Source

The visualiser reads the final entity-resolved Timeline Maker corpus (repo-relative paths in
`config.yaml`, resolved by `path_utils.py` regardless of the launch directory):

```text
../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/<bucket>/
```

for the five buckets `fin_fraud`, `family_matrimonial`, `land_property`, `motor_accidents`,
`sexual_offences` — each with its own colour throughout the app.

## 🧬 Graph Policy (mirrors the GNN)

- **Shared nodes** (one node per canonical entity across cases): `STATUTE`, `PROVISION`,
  `PRECEDENT`, `COURT`, `JUDGE`, `LAWYER`
- **Local nodes** (case-specific): `PETITIONER`, `RESPONDENT`
- **Skipped** (too noisy): `DATE`, `GPE`, `OTHER_PERSON`, `CASE_NUMBER`, `ORG`, `WITNESS`
- **Connectivity-ranked sampling** — cases are ranked by the sum of their shared-entity
  degrees; the app shows the most structurally central cases per bucket (slider from top-10
  up to `cases_per_bucket`, default 300; display cap 6 000 nodes; hubs with degree ≥ 8 always kept).

## 🪟 App Layout & Tabs

Three-column layout — sidebar | graph + tabs | persistent details panel. Tabs:
**Overview** · **Top Hubs** · **Bridges** · **Top Cases** · **Case Connections**.
Clicking a case shows every connected case with the specific shared entities
(statute/court/judge/…) grouped by type.

---

## 📄 Main Files

| File | Role |
|------|------|
| `config.yaml` | Buckets, data paths, graph policy, sampling limits, colours, Dash settings. |
| `path_utils.py` | Shared path resolver (config-relative). |
| `build_graph.py` | Builds the NetworkX graph artifacts from the resolved JSONs. |
| `app.py` | The interactive Dash app. |
| `generate_plots.py` | Static figures from the sampled graph artifacts. |
| `generate_plots_full.py` | Full-dataset descriptive figures straight from the JSON buckets. |
| `graph_stats.py` | Full-graph statistics + figures. |
| `generate_rhetorical_before_after.py` | Rhetorical-role distributions before/after leakage filtering. |
| `generate_thesis_extras.py` / `organise_thesis_figures.py` | Extra thesis figures and figure organisation. |
| [`entity_analysis/`](entity_analysis/README.md) | Separate entity co-occurrence analysis + Dash app (port 8052). |
| [`outputs/`](outputs/README.md) | 📤 Generated pickles, layouts, plots, stats, thesis figures. |
| [`dump_old_data_visualisations/`](dump_old_data_visualisations/README.md) | 🗄️ Archived older visualisation outputs. |

---

## ▶️ Running

```bash
# One-time environment setup (creates 'graph_vis'):
bash GRAPH_VISUALISER/setup_env.sh

# Build the graph artifacts:
bash GRAPH_VISUALISER/run_build.sh
bash GRAPH_VISUALISER/run_build.sh --limit 500 --skip-layout   # quick smoke test

# Launch the Dash app (default port 8050):
bash GRAPH_VISUALISER/run_app.sh
bash GRAPH_VISUALISER/run_app.sh 8051                          # custom port
MAMBA_ENV=my_env bash GRAPH_VISUALISER/run_app.sh 8050         # custom env
```

> ⚠️ The ⭐ Multi-Hearing Stage Test Visualiser also defaults to port **8050** — run one at a
> time or pass different ports. Remote server? Tunnel first:
> `ssh -L 8050:localhost:8050 <user>@<server>`.

## 🖼️ Static Outputs

```bash
micromamba run -n graph_vis python GRAPH_VISUALISER/generate_plots.py --config GRAPH_VISUALISER/config.yaml
micromamba run -n graph_vis python GRAPH_VISUALISER/generate_plots_full.py --config GRAPH_VISUALISER/config.yaml
micromamba run -n graph_vis python GRAPH_VISUALISER/graph_stats.py --config GRAPH_VISUALISER/config.yaml
```

All artifacts land in [`outputs/`](outputs/README.md).

## 🔗 Entity Analysis Sub-App

```bash
bash GRAPH_VISUALISER/entity_analysis/run.sh both 8052        # analyse + launch app
bash GRAPH_VISUALISER/entity_analysis/run.sh analyse-only     # refresh JSONs only
```

---

⬆️ Back to the [repository root](../README.md)
