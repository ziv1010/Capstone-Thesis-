# 📈 dataset_size_sweep — Training-Set Size Scaling

> Part of [`experiments/`](../README.md).

Measures how model performance changes as the number of training examples grows, while
keeping graph and pipeline assumptions fixed.

## 📄 Files

| File | Role |
|------|------|
| `run_fixed_open_dataset_size_sweep.py` | Full sweep over the configured sizes. |
| `run_fixed_open_limited_experiment.py` | Smaller limited run for quick checks. |

## ▶️ Run

From `section_GNN/` (inputs follow the fixed-open pipeline config):

```bash
micromamba run -n thesis_work python experiments/dataset_size_sweep/run_fixed_open_dataset_size_sweep.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml

# quick limited check:
micromamba run -n thesis_work python experiments/dataset_size_sweep/run_fixed_open_limited_experiment.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

Outputs are written below the configured `outputs_dir`, usually under a
`dataset_size_sweep/` subfolder.

---

⬆️ Back to [`experiments/`](../README.md)
