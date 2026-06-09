# Cross-Validated Outputs

This folder stores optional audit/validation labels. It is separate from the
main production outputs in `../final_outputs/`.

## Producer

The wrapper is:

```bash
bash ../run_scripts/run_crossval_all_buckets.sh
```

The underlying script is:

```text
../extra_scripts/add_case_outcome_labels_crossval_mistral.py
```

## Inputs

The cross-validation labeler reads enriched OpenNyAI summary JSONs from:

```text
../final_outputs/<bucket>_summary_opennyai/enriched_jsons/
```

## Outputs

Per-bucket outputs are written to:

```text
<bucket>/
```

Logs are written to:

```text
logs/
```

`label_comparison/` contains comparison artifacts used to inspect agreement or
differences between labeling approaches.
