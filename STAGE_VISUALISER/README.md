# 🔬 STAGE_VISUALISER — Pipeline Stage Inspector (Port 8053)

> 🔧 Auxiliary Dash app for inspecting how a single case changes across the
> **Fixed_GPU_OpenNyai pipeline stages** — useful for debugging Stage ② and demonstrating the
> extraction chain.
> *(The two main visualisers are the [Multi-Hearing Stage Test Visualiser](../section_GNN/multi_hearing_stage_test/visualiser/README.md), port 8050, and the [Final Explanation Visualizer](../FINAL_EXPLANATION/README.md), port 8899; the extra [Graph Visualiser](../GRAPH_VISUALISER/README.md) runs on port 8050.)*

## 🎞️ Stages Shown

| Stage | Transformation |
|:-----:|----------------|
| 1 | OpenNyAI NER + rhetorical-role extraction |
| 2 | OpenNyAI summary enrichment |
| 3 | Mistral outcome labelling |
| 4 | Cross-validated outcome augmentation |

## ▶️ Running

```bash
bash STAGE_VISUALISER/run_app.sh          # default port 8053, env: graph_vis
bash STAGE_VISUALISER/run_app.sh 8060     # custom port
```

Remote server? Tunnel first: `ssh -L 8053:localhost:8053 <user>@<server>`, then open
`http://localhost:8053`.

## 📥 Requirements

The app expects the local generated outputs of Stage ②:

```text
Fixed_GPU_OpenNyai/final_outputs/
Fixed_GPU_OpenNyai/cross_validated_outputs/
```

(Both are Git-ignored — regenerate them via
[`Fixed_GPU_OpenNyai/run_scripts/`](../Fixed_GPU_OpenNyai/run_scripts/README.md).)
The `graph_vis` environment is created by `GRAPH_VISUALISER/setup_env.sh`.

## 📄 Files

| Path | Role |
|------|------|
| `app.py` | The Dash application. |
| `run_app.sh` | Launcher (port handling + env). |
| [`assets/`](assets/README.md) | Static frontend assets. |

---

⬆️ Back to the [repository root](../README.md)
