# Run Scripts

Shell wrappers for the `Fixed_GPU_OpenNyai` pipeline live here. They resolve
paths relative to this folder, so the repository can move without editing
machine-specific absolute paths.

Each wrapper derives:

- `PROJECT_ROOT`: the parent `Fixed_GPU_OpenNyai/` folder.
- `REPO_ROOT`: the parent `Capstone-Thesis-/` repository folder.

Prefer running these from `Fixed_GPU_OpenNyai/`.

## Main Pipeline Wrappers

Run in this order for the standard thesis artifact chain:

```bash
bash run_scripts/run_ner_rr_all_categories.sh --gpus 0,1,2,3
bash run_scripts/run_opennyai_summaries_all.sh --gpus 0,1,2,3
bash run_scripts/run_mistral_labels_from_opennyai_summaries_all.sh --gpus 0,1,2,3
bash run_scripts/run_merge_timeline_from_final_outputs.sh
```

- `run_ner_rr_all_categories.sh`
  - Reads `../INPUT_DATA/*_text`.
  - Writes `final_outputs/*_extract/annotations`.

- `run_opennyai_summaries_all.sh`
  - Reads `final_outputs/*_extract/annotations`.
  - Writes `final_outputs/*_summary_opennyai/enriched_jsons`.

- `run_mistral_labels_from_opennyai_summaries_all.sh`
  - Reads `final_outputs/*_summary_opennyai/enriched_jsons`.
  - Writes `final_outputs/*_labelled_mistral/labelled_jsons`.

- `run_merge_timeline_from_final_outputs.sh`
  - Reads `final_outputs/*_labelled_mistral/labelled_jsons`.
  - Writes `../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/*_timed_mistral`.

## Dataset-Building Wrappers

These consume the merged Timeline Maker folders:

```bash
bash run_scripts/run_build_cross_bucket_cases.sh
bash run_scripts/run_build_cross_bucket_remaining_cases.sh
```

- `run_build_cross_bucket_cases.sh`
  - Copies up to `8000` sorted case JSONs per bucket into
    `Timeline_Maker/cross_bucket_cases_8k_each_mistral`.

- `run_build_cross_bucket_remaining_cases.sh`
  - Skips the first `8000` sorted case JSONs per bucket and copies the
    remaining files into
    `Timeline_Maker/cross_bucket_cases_remaining_after_8k_each_mistral`.

## Optional Labeling Wrappers

These call scripts in `../extra_scripts/` and are not required for the main
pipeline:

```bash
bash run_scripts/run_crossval_all_buckets.sh
bash run_scripts/run_mistral_multi_labels_from_opennyai_summaries_all.sh
```

- `run_crossval_all_buckets.sh`
  - Runs the cross-validation/audit labeler.
  - Writes `cross_validated_outputs/<bucket>`.

- `run_mistral_multi_labels_from_opennyai_summaries_all.sh`
  - Runs the experimental multi-label outcome labeler.
  - Writes separate multi-label output folders under `final_outputs/`.

## Useful Overrides

Most wrappers accept command-line flags and environment variables. Common
overrides include:

- `CONDA_ENV`
- `GPUS`
- `MODEL` or `MODEL_ID`
- `BASE_INPUT`
- `BASE_OUTPUT`
- `FINAL_OUTPUTS_DIR`
- `TIMELINE_ROOT`
- `OUTPUT_DIR`
