# runs_inlegalbert_remaining

This folder contains InLegalBERT configs used to fill remaining thesis-table
cells after the main matrix was run.

## Variant Folders

- `central_section_lr_decay/`
- `central_section_no_lr/`
- `entity_section_lr_decay/`
- `entity_section_no_lr/`
- `party_args_preamble_lr_decay/`
- `party_args_preamble_no_lr/`
- `section_sep_lr_decay/`

Each variant contains per-bucket configs. These are normally launched by:

```bash
bash run_scripts/run_remaining_table_experiments_8gpu.sh
```

## Outputs

Generated artifacts generally use:

```text
data/inlegalbert_remaining/
outputs/inlegalbert_remaining/
```

Treat these configs as table-completion experiments rather than the primary
baseline matrix.
