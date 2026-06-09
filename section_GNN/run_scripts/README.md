# run_scripts

Top-level orchestration scripts live here. They are for larger matrices that
span multiple buckets, encoders, or ablation families.

Run from `section_GNN`:

```bash
cd section_GNN
```

## Launchers

| Script | Purpose |
| --- | --- |
| `run_all_experiments.sh` | Main BGE-M3 experiment matrix across buckets. |
| `run_complete_ablation_matrix.sh` | Fills older missing BGE-M3 ablation cells. |
| `run_baseline_party_args_lr_control.sh` | Controlled baseline vs party-argument LR-decay comparison. |
| `run_party_args_preamble_and_section_sep_lr_decay.sh` | BGE-M3 party+preamble and section-separation LR-decay runs. |
| `run_inlegalbert_experiments.sh` | Main InLegalBERT comparison matrix. |
| `run_remaining_table_experiments_8gpu.sh` | Remaining thesis-table BGE-M3 and InLegalBERT cells. |
| `run_remaining_non_cross_bucket_ablations.sh` | Non-cross-bucket ablation sync/run helper. |
| `run_timed_mistral_buckets_8gpu.sh` | Timed-bucket preprocessing, graph building, training, and evaluation launcher. |

## Logging Pattern

Long runs should write to `run_logs/`:

```bash
nohup bash run_scripts/run_inlegalbert_experiments.sh \
  > run_logs/run_inlegalbert_experiments.log 2>&1 &
```

Monitor with:

```bash
tail -f run_logs/run_inlegalbert_experiments.log
```

## Environment Variables

Most scripts accept environment overrides such as:

- `MAMBA_ENV`
- `BUILD_GPUS`
- `TRAIN_GPUS`
- `EVAL_GPUS`
- `SKIP_BUILD`
- `RUN_NAME_SUFFIX`

Check the header of each script before launching a large run.
