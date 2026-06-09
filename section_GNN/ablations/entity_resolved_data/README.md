# entity_resolved_data

This ablation uses externally resolved entity data instead of the baseline
entity extraction/canonicalization assumptions.

## Main Files

- `prepare_configs.py`: generates per-bucket configs for entity-resolved runs.
- `preprocess_fixed_open_resolved.py`: preprocessing entry point for resolved
  entity payloads.
- `run_entity_resolved_data_ablation.sh`: main launcher.
- `run_section_sep_no_names_both_lr.sh`: helper for combined section/no-name LR
  comparisons.

## Config Layout

Generated configs live under:

```text
ablations/entity_resolved_data/configs/
ablations/entity_resolved_data/configs_no_lr/
```

The subfolders `party`, `section`, and `section_no_names` identify which graph
variant the resolved-entity data is paired with.

## Outputs

Generated data and model outputs usually use:

```text
data/ablations/entity_resolved_data/
outputs/ablations/entity_resolved_data/
```
