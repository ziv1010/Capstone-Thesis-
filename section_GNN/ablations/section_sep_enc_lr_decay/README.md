# section_sep_enc_lr_decay

This folder stores section-separated encoding configs with LR-decay training
settings. It is used by later thesis-table runs where section-separated graphs
need to be compared under the same optimizer schedule as other LR-decay cells.

## Usage

These configs are usually launched by higher-level scripts rather than manually:

```bash
bash run_scripts/run_party_args_preamble_and_section_sep_lr_decay.sh
bash run_scripts/run_remaining_table_experiments_8gpu.sh
```

## Relationship to `section_sep_enc`

`section_sep_enc/` is the base section-separated ablation. This folder keeps the
same graph idea but changes training schedule/configuration for controlled
comparisons.
