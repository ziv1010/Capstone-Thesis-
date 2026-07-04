# 🚀 run_scripts — Stage ⑤ Shell Launchers

> Part of [`FINAL_EXPLANATION/`](../README.md) · run from the repository root **or** from
> `FINAL_EXPLANATION/` — every script resolves `SCRIPT_DIR`, `APP_ROOT`
> (`FINAL_EXPLANATION/`), `REPO_ROOT`, and `SECTION_GNN` (`REPO_ROOT/section_GNN`) from its
> own location.

## ⭐ Primary Launchers

| Script | Purpose |
|--------|---------|
| `run_entity_resolved_section_sep_lr_decay_cross_bucket_all.sh` | **Full current end-to-end workflow** — explanations, validation, audits, pattern analyses, full-graph analyses, optional visualizer startup. |
| `run_default.sh` | Single-process fold-00 explanation run — ideal for smoke tests. |
| `run_multi_gpu.sh` | Shards explanation cases over multiple GPUs; writes shard logs/outputs and merges shard CSVs with `merge_outputs.py` (skip with `--no-merge`). |
| `run_validation_multi_gpu.sh` | Shards faithfulness + prediction-bucket validation over GPUs; merges with `merge_validation_outputs.py` (skip with `--no-merge`). |
| `run_visualizer.sh` | Starts the ⭐ **Final Explanation Visualizer** (`visualizer.py`) on port **8899**. |

## 🕵️ Audit & Analysis Launchers

| Script | Purpose |
|--------|---------|
| `run_identity_shortcut_audit.sh` | Identity-only train-label shortcut audit. |
| `run_mask_sensitivity_audit.sh` | Frozen-model inference masks for identity groups and hub authorities. |
| `run_full_graph_communities.sh` | Full-graph Leiden detection, hierarchy analysis, profiling, bridge/hub authority analysis. |

## 🔧 Common Environment Overrides

| Variable | Meaning |
|----------|---------|
| `MAMBA_ENV` | Micromamba environment (default `thesis_work`). |
| `GPUS` | Comma-separated GPU IDs. |
| `OUTPUT_BASE` | Prefix used by the full workflow. |
| `EXPLAIN_DIR` / `PATTERN_DIR` / `FULL_GRAPH_DIR` | The three output directories. |
| `MODEL` / `PRED` / `GRAPH` / `CONFIG` | Checkpoint, predictions CSV, graph cache, experiment config. |
| `RUN_*` | Enable/skip individual stages in the full workflow. |
| `HOST` / `PORT` | Visualizer bind address (defaults `127.0.0.1` / `8899`). |

---

⬆️ Back to [`FINAL_EXPLANATION/`](../README.md)
