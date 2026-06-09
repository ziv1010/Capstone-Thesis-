# configs

This folder contains older legacy configs retained for reference. The main
current experiment configs live closer to their launchers:

- `runs/<bucket>/config.yaml`
- `runs_v2/<variant>/<bucket>/config.yaml`
- `ablations/<variant>/<bucket>/config.yaml`
- `experiments/fixed_open_pipeline/*.yaml`
- `multi_hearing_stage_test/config.yaml`

## Current Path Convention

Configs should use paths relative to `section_GNN`, such as:

```yaml
paths:
  raw_json_dir: ../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/cross_bucket_total_dataset
  processed_dir: data/timed_bucket_runs/cross_bucket_total_dataset/processed
  outputs_dir: outputs/timed_bucket_runs/cross_bucket_total_dataset
```

Python scripts should load configs through `src.utils.io.load_yaml` so those
relative paths resolve correctly.

## `old/`

`configs/old/` contains pre-refactor configs. Use them only for historical
comparison unless you have checked that their fields still match the current
pipeline.
