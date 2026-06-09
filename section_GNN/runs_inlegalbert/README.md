# runs_inlegalbert

`runs_inlegalbert` mirrors the main BGE-M3 experiment matrix using InLegalBERT
features. The folder is organized by graph/input variant, then by dataset bucket.

## Variant Folders

- `baseline/`
- `baseline_lr_decay/`
- `party_args_no_lr/`
- `party_args_lr_decay/`
- `text_only/`
- `no_names/`
- `no_cross_case/`
- `hierarchical_enc/`
- `section_sep_enc/`
- `case_node_minimised/`

Each variant folder contains per-bucket `config.yaml` files. Top-level launch
scripts in `run_scripts/` orchestrate these configs.

## Main Launcher

```bash
bash run_scripts/run_inlegalbert_experiments.sh
```

## Outputs

Generated artifacts generally use:

```text
data/inlegalbert_runs/
outputs/inlegalbert_runs/
```

Use this folder when comparing model behavior across text encoders while
keeping graph variants aligned with the BGE-M3 matrix.
