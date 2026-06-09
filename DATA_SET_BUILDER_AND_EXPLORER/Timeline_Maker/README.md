# Timeline Maker

Merges per-hearing and per-stage case JSON files into thesis-ready case timelines.

## Key Scripts

- `merge_cases.py`, `merge_cases_v2.py`, `merge_cases_v3.py`: merge related case records into consolidated timelines.
- `build_cross_bucket_dataset.py`: assemble cross-domain datasets from bucket-specific merged outputs.
- `compare_merges.py`: compare merge versions and audit differences.
- `entity_resolver/resolve_entities.py`: canonicalize statutes, provisions, and precedent references.
- `visualiser.html` and `visualiser_dual.html`: static local viewers for inspecting merged case timelines.

## Layout

- `output_merged_v3/`: current unresolved merged timeline corpus used by downstream GNN runs.
- `output_merged_v3_resolved/`: entity-resolved version with statute, provision, and precedent canonicalization.
- `old_outputs/`: older merge outputs kept for comparison audits.
- `dump/`: superseded intermediate outputs and older generated folders.
- `entity_resolver/logs/`: logs for entity-resolution runs.

The root output folders stay in place because existing `section_GNN` configs
reference them by relative path.

## Usage

From the repository root:

```bash
cd DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker
python merge_cases_v2.py \
  --input ../../Fixed_GPU_OpenNyai/final_outputs/fin_fraud_labelled_mistral/labelled_jsons \
  --output output_merged_v2
```

Serve the local visualiser from this folder:

```bash
python3 -m http.server 8080
```

## Generated Data

Large merged corpora such as `output_merged_v3/`, `output_merged_v3_resolved/`, `old_outputs/`, and bucket-specific timed datasets are ignored. They should be regenerated locally from the scripts above or restored from external storage.
