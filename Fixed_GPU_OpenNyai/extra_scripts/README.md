# Extra Scripts

This folder contains optional labelers that are useful for audits or experiments
but are not part of the main production artifact chain.

## Files

- `add_case_outcome_labels_crossval_mistral.py`
  - Audit/validation outcome labeler.
  - Uses multiple yes/no checks and deterministic aggregation.
  - Reads enriched OpenNyAI summary JSONs.
  - Used by `../run_scripts/run_crossval_all_buckets.sh`.
  - Writes into `../cross_validated_outputs/`.

- `add_multi_label_outcome_from_enriched.py`
  - Experimental richer outcome labeler.
  - Produces six binary outcome flags plus a final ternary outcome label.
  - Reads enriched OpenNyAI summary JSONs.
  - Used by `../run_scripts/run_mistral_multi_labels_from_opennyai_summaries_all.sh`.
  - Writes separate multi-label folders under `../final_outputs/`.

## Main Labeler

For the current main pipeline, use:

```text
../add_case_outcome_labels_from_enriched.py
```

That script imports shared classification code from:

```text
../add_case_outcome_labels_mistral.py
```
