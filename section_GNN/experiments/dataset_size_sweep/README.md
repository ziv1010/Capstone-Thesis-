# dataset_size_sweep

This experiment measures how performance changes as the number of training
examples changes while keeping the graph/pipeline assumptions fixed.

## Files

- `run_fixed_open_dataset_size_sweep.py`: full sweep over configured sizes.
- `run_fixed_open_limited_experiment.py`: smaller limited run for quick checks.

## Inputs

The scripts usually start from:

```text
experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

They expect cleaned cases and graph-building settings compatible with the
fixed-open pipeline.

## Run

From `section_GNN`:

```bash
micromamba run -n thesis_work python experiments/dataset_size_sweep/run_fixed_open_dataset_size_sweep.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

For a quick limited run:

```bash
micromamba run -n thesis_work python experiments/dataset_size_sweep/run_fixed_open_limited_experiment.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

Outputs are written below the configured `outputs_dir`, usually under a
`dataset_size_sweep/` subfolder.
