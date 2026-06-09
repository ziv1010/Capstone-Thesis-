# no_names_lr_decay

V2 no-names ablation with LR-decay training.

## Relationship to `party_args_lr_decay`

This folder uses the shared v2 builder/trainer from
`runs_v2/party_args_lr_decay/` but swaps in configs that remove name-bearing
features/nodes.

## Run All Buckets

```bash
bash runs_v2/no_names_lr_decay/run_all_buckets.sh
```

Outputs follow each config's `paths.outputs_dir`.
