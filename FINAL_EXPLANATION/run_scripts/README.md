# Run Scripts

This folder contains shell launchers for the final explanation pipeline.

Run them from the repository root:

```bash
bash FINAL_EXPLANATION/run_scripts/run_default.sh
```

or from `FINAL_EXPLANATION/`:

```bash
bash run_scripts/run_default.sh
```

Each script resolves:

- `SCRIPT_DIR`: this `run_scripts/` folder.
- `APP_ROOT`: the parent `FINAL_EXPLANATION/` folder.
- `REPO_ROOT`: the parent repository folder.
- `SECTION_GNN`: `REPO_ROOT/section_GNN`.

## Primary Launchers

- `run_entity_resolved_section_sep_lr_decay_cross_bucket_all.sh`
  - Full current end-to-end explanation workflow.
  - Runs explanations, validation, audits, pattern analyses, full-graph
    analyses, and optional visualizer startup.

- `run_default.sh`
  - Single-process default fold-00 explanation run.
  - Useful for smoke tests and small local runs.

- `run_multi_gpu.sh`
  - Shards explanation cases over multiple GPUs.
  - Writes shard logs and outputs under the requested output directory.
  - Merges shard CSVs with `merge_outputs.py` unless `--no-merge` is passed.

- `run_validation_multi_gpu.sh`
  - Shards faithfulness and prediction-bucket validation over GPUs.
  - Merges validation outputs with `merge_validation_outputs.py` unless
    `--no-merge` is passed.

- `run_visualizer.sh`
  - Starts the local HTML/JS visualizer served by `visualizer.py`.

## Audit And Analysis Launchers

- `run_identity_shortcut_audit.sh`
  - Runs the identity-only train-label shortcut audit.

- `run_mask_sensitivity_audit.sh`
  - Runs frozen-model inference masks for identity groups and hub authorities.

- `run_full_graph_communities.sh`
  - Runs full-graph Leiden detection, hierarchy analysis, profiling, and
    bridge/hub authority analysis.

## Common Environment Overrides

- `MAMBA_ENV`: micromamba environment. Default: `thesis_work`.
- `GPUS`: comma-separated GPU IDs.
- `OUTPUT_BASE`: prefix used by the full workflow.
- `EXPLAIN_DIR`: explanation output directory.
- `PATTERN_DIR`: pattern analysis output directory.
- `FULL_GRAPH_DIR`: full-graph analysis output directory.
- `MODEL`: model checkpoint.
- `PRED`: predictions CSV.
- `GRAPH`: graph cache.
- `CONFIG`: experiment config.
- `RUN_*`: enable or skip individual stages in the full workflow.
