# 🗃️ configs — Legacy Configuration Archive

> Part of [`section_GNN/`](../README.md) · **reference only** — the active configs live next
> to their launchers.

Current experiment configs are located at:

- `runs/<bucket>/config.yaml`
- `runs_v2/<variant>/<bucket>/config.yaml`
- `ablations/<variant>/<bucket>/config.yaml`
- `experiments/fixed_open_pipeline/*.yaml`
- `multi_hearing_stage_test/config.yaml`

## 🗺️ Current Path Convention

Configs use paths relative to `section_GNN`:

```yaml
paths:
  raw_json_dir: ../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/cross_bucket_total_dataset
  processed_dir: data/timed_bucket_runs/cross_bucket_total_dataset/processed
  outputs_dir: outputs/timed_bucket_runs/cross_bucket_total_dataset
```

Load them via `src.utils.io.load_yaml` so relative paths resolve correctly.

## 🗄️ `old/`

`configs/old/` holds pre-refactor configs. Use them only for historical comparison after
checking that their fields still match the current pipeline.

---

⬆️ Back to [`section_GNN/`](../README.md)
