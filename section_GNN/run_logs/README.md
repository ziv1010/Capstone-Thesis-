# run_logs

This folder stores logs from long-running `section_GNN` experiment launchers.
The `.log` files are generated artifacts and can be deleted or regenerated.

## Recommended Pattern

From `section_GNN`:

```bash
nohup bash run_scripts/<script>.sh > run_logs/<script>.log 2>&1 &
tail -f run_logs/<script>.log
```

## Naming

Use a descriptive name that includes the launcher or experiment family, for
example:

```text
run_logs/run_inlegalbert_experiments.log
run_logs/remove_central_authorities_fin_fraud.log
```

Logs are useful for diagnosing failed folds, GPU assignment issues, and missing
graph caches, but the authoritative metrics are the `kfold_summary.json` files
under `outputs/`.
