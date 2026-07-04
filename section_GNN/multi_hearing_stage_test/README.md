# 🎬 multi_hearing_stage_test — Early-Signal & Stage-Transition Experiment

> Part of [`section_GNN/`](../README.md) · home of the ⭐ **Multi-Hearing Stage Test
> Visualiser (port 8050)**, one of the repository's two main visualisers.

Tests how the model's predictions **evolve across the multiple hearings of the same case**:
can the eventual outcome be detected early, and which factors flip a prediction between
hearings?

## 🔄 Workflow

The pipeline builds stage-tagged JSONs for multi-hearing cases, preprocesses them into the
normal graph format, runs a trained GNN checkpoint on the stage graph, and analyzes
prediction transitions from earlier to later hearings.

## 📄 Main Files

| File | Role |
|------|------|
| `config.yaml` | Graph/inference configuration. |
| [`scripts/`](scripts/README.md) | Numbered step scripts `01`–`06` + `run_all.sh` (details in the scripts README). |
| [`visualiser/`](visualiser/README.md) | ⭐ Multi-Hearing Stage Test Visualiser — Dash app, port **8050**. |
| `data/` · `outputs/` | 📤 Generated stage inputs and results. |
| `dump/` | 🗄️ Superseded material. |

## ▶️ Run

From `section_GNN/`:

```bash
bash multi_hearing_stage_test/scripts/run_all.sh

# then browse the results:
bash multi_hearing_stage_test/visualiser/run_app.sh        # http://localhost:8050
```

## 📤 Key Outputs

```text
outputs/stage_manifest.csv                  # stage-tagged case inventory
outputs/inference/predictions.csv           # per-stage model predictions
outputs/analysis/stage_transitions.csv      # prediction transitions per base case
outputs/analysis/transition_counts.json     # aggregate transition statistics
outputs/analysis/per_case_diffs/            # per-case difference reports
```

These outputs also feed `posthoc_case_reports/timeline_merger/` and the early-detection
figure in `Latex_Documentation/PAPER_DATA/`.

---

⬆️ Back to [`section_GNN/`](../README.md)
