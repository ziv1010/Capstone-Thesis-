# 🎛️ run_scripts — Matrix Orchestration

> Part of [`section_GNN/`](../README.md) · top-level launchers for experiment matrices that
> span multiple buckets, encoders, or ablation families. Run from `section_GNN/`.

## 🚀 Launchers

| Script | Purpose |
|--------|---------|
| `run_all_experiments.sh` | Main BGE-M3 experiment matrix across buckets. |
| `run_complete_ablation_matrix.sh` | Fills older missing BGE-M3 ablation cells. |
| `run_baseline_party_args_lr_control.sh` | Controlled baseline vs party-argument LR-decay comparison. |
| `run_party_args_preamble_and_section_sep_lr_decay.sh` | BGE-M3 party+preamble and section-separated LR-decay runs. |
| `run_inlegalbert_experiments.sh` | Main InLegalBERT comparison matrix. |
| `run_remaining_table_experiments_8gpu.sh` | Remaining thesis-table BGE-M3 and InLegalBERT cells. |
| `run_remaining_non_cross_bucket_ablations.sh` | Non-cross-bucket ablation sync/run helper. |
| `run_timed_mistral_buckets_8gpu.sh` | Timed-bucket preprocess + build + train + eval launcher. |

## 🧾 Logging Pattern

Long runs should log into [`run_logs/`](../run_logs/README.md):

```bash
nohup bash run_scripts/run_inlegalbert_experiments.sh \
  > run_logs/run_inlegalbert_experiments.log 2>&1 &
tail -f run_logs/run_inlegalbert_experiments.log
```

## 🔧 Environment Overrides

Most scripts honour `MAMBA_ENV`, `BUILD_GPUS`, `TRAIN_GPUS`, `EVAL_GPUS`, `SKIP_BUILD`, and
`RUN_NAME_SUFFIX`. Check each script's header before a large launch.

---

⬆️ Back to [`section_GNN/`](../README.md)
