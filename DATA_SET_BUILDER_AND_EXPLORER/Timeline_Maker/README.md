# Timeline Maker

Merges per-hearing and per-stage case JSON files into thesis-ready case timelines.

## Key Scripts

- `merge_cases.py`, `merge_cases_v2.py`, `merge_cases_v3.py`: merge related case records into consolidated timelines.
- `build_cross_bucket_dataset.py`: assemble cross-domain datasets from bucket-specific merged outputs.
- `compare_merges.py`: compare merge versions and audit differences.
- `entity_resolver/resolve_entities.py`: canonicalize statutes, provisions, and precedent references.
- `visualiser.html` and `visualiser_dual.html`: static local viewers for inspecting merged case timelines.

## Generated Data

Large merged corpora such as `output_merged_v3/`, `output_merged_v3_resolved/`, `old_outputs/`, and bucket-specific timed datasets are ignored. They should be regenerated locally from the scripts above or restored from external storage.
