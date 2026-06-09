# INPUT_DATA

Local raw and extracted input material for the OpenNyAI preprocessing stage.

This folder is the source side of the active pipeline:

```text
INPUT_DATA/ -> Fixed_GPU_OpenNyai/ -> DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/
```

## Layout

Domain folders usually appear in two forms:

- `<domain>/`: bucket-specific raw/extracted material from earlier collection
  steps.
- `<domain>_text/`: plain `.txt` judgment files consumed by
  `Fixed_GPU_OpenNyai/run_ner_rr_custom.py`.

Current domain names:

- `family_matrimonial`
- `financial_fraud`
- `food_safety`
- `land_property`
- `motor_accidents`
- `sexual_offences`

Some holdout folders are present for food-safety, land-property, and
motor-accident data. Those are for auxiliary/cross-domain checks, not the main
five-domain GNN training set.

## Usage

The normal entry point is:

```bash
bash Fixed_GPU_OpenNyai/run_scripts/run_ner_rr_all_categories.sh
```

That wrapper reads from the `*_text/` folders and writes generated annotations
under `Fixed_GPU_OpenNyai/final_outputs/`.

## Git Policy

The raw and extracted case corpora are large and ignored by Git. Keep scripts
and documentation tracked; keep the data local or restore it from external
storage when rerunning the pipeline.
