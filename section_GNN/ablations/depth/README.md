# depth

The depth ablation tests different GNN layer counts.

## Layout

Each bucket can contain configs such as:

```text
config_depth1.yaml
config_depth2.yaml
config_depth3.yaml
run.sh
```

## Purpose

This measures whether performance depends on shallow local aggregation or
deeper multi-hop message passing.

## Run Example

```bash
bash ablations/depth/fin_fraud_timed_mistral/run.sh
```

The run script normally reuses the matching baseline graph cache and changes
only the model depth/training config.
