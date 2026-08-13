# ⭐ Multi-Hearing Stage Test Visualiser — Port 8050

> Part of [`multi_hearing_stage_test/`](../README.md) · one of the repository's **two main
> visualisers** (the other is the [Final Explanation Visualizer](../../../FINAL_EXPLANATION/README.md#-visualizer--port-8899), port 8899).

An interactive Dash app for browsing the multi-hearing and early-detection results — how
soon the model matches the final outcome and how its prediction changes from hearing to
hearing.

## 🪟 Views

| Tab | Contents |
|-----|----------|
| **Overview** | Aggregate transition statistics and dataset summary. |
| **Early detection** | First-hearing correctness, high-confidence early detection, later correction rate, and ranked present-vs-absent signals. |
| **Transition explorer** | Interactive exploration of prediction transitions across stages. |
| **Case drill-down** | Per-case timeline of stage predictions, factors, and differences. |

## 📄 Files

| File | Role |
|------|------|
| `app.py` | The Dash application (reads only from `../outputs/` — no GPU needed). |
| `run_app.sh` | Launcher (port + env handling). |
| `assets/` | Static styling assets. |

## 📥 Inputs

Produced by [`../scripts/`](../scripts/README.md):

- `../outputs/stage_manifest.csv`
- `../outputs/inference/predictions.csv`
- `../outputs/analysis/stage_transitions.csv`
- `../outputs/analysis/early_signal_test/early_signal_summary.json`
- early- and later-signal CSVs under `../outputs/analysis/early_signal_test/`
- per-case factor reports under `../outputs/analysis/`

## ▶️ Run

From `section_GNN/`:

```bash
bash multi_hearing_stage_test/visualiser/run_app.sh        # default port 8050, env: graph_vis
bash multi_hearing_stage_test/visualiser/run_app.sh 8060   # custom port
```

On a remote server, tunnel first: `ssh -L 8050:localhost:8050 <user>@<server>`, then open
`http://localhost:8050`.

> ⚠️ The extra [Graph Visualiser](../../../GRAPH_VISUALISER/README.md) also defaults to
> port **8050** — run one at a time or pass different ports.

---

⬆️ Back to [`multi_hearing_stage_test/`](../README.md)
