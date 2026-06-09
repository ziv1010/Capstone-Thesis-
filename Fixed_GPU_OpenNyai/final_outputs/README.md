# Final Outputs

This folder stores the main generated artifacts from the OpenNyAI and Mistral
pipeline.

## Folder Pattern

Each legal bucket has up to three output stages:

```text
<bucket>_extract/
<bucket>_summary_opennyai/
<bucket>_labelled_mistral/
```

Current bucket names include:

- `family_matrimonial`
- `fin_fraud`
- `food_safety`
- `land_property`
- `motor_accidents`
- `sexual_offences`

Some buckets may be missing a later stage if that stage has not been run or was
not part of the final selected dataset.

## Stage Contents

- `<bucket>_extract/annotations/`
  - OpenNyAI NER and rhetorical-role JSONs.

- `<bucket>_summary_opennyai/enriched_jsons/`
  - Extracted JSONs with OpenNyAI summary fields added.

- `<bucket>_labelled_mistral/labelled_jsons/`
  - Enriched JSONs with Mistral outcome labels added.

## Runtime Cache Folders

Parallel extraction runs may create folders like:

```text
<bucket>_extract/.worker_home_0/
<bucket>_extract/.worker_home_1/
```

These are per-worker runtime/cache homes. They can contain CUDA, CuPy, and
OpenNyAI model cache files. They are not the saved thesis artifacts; the saved
artifacts are in `annotations/`, `enriched_jsons/`, and `labelled_jsons/`.

Deleting `.worker_home_*` folders does not delete completed annotations,
summaries, or labels, but future reruns may recreate/download cache files.
