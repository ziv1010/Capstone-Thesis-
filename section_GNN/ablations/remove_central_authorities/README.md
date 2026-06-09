# remove_central_authorities

This ablation removes or filters highly central authority nodes before training.

## Purpose

Very common statutes, provisions, or precedents can become high-degree hubs.
This variant tests whether performance depends on those hubs or whether they
inject noise/shortcut behavior.

## Main Files

- `analyze_central_authorities.py`: identifies high-centrality authority nodes.
- `filter_cleaned_cases.py`: removes selected central authorities from cleaned
  case records.
- `prepare_configs.py`: generates configs for filtered runs.
- `run_remove_central_authorities_ablation.sh`: main launcher.

## Config Layout

Generated configs live under:

```text
ablations/remove_central_authorities/configs/
ablations/remove_central_authorities/configs_no_lr/
```

Outputs usually land under:

```text
outputs/ablations/remove_central_authorities/
data/ablations/remove_central_authorities/
```
