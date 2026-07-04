# 🚀 run_scripts — Stage ② Shell Wrappers

> Part of [`Fixed_GPU_OpenNyai/`](../README.md) · the **recommended entry points** for Stage ②.

Every wrapper resolves its paths from its own location — `PROJECT_ROOT` (the parent
`Fixed_GPU_OpenNyai/`) and `REPO_ROOT` (the repository) — so the repo can move between
machines without editing any script. Prefer running these from `Fixed_GPU_OpenNyai/`.

---

## 🏭 Main Pipeline Wrappers (run in this order)

```bash
bash run_scripts/run_ner_rr_all_categories.sh --gpus 0,1,2,3
bash run_scripts/run_opennyai_summaries_all.sh --gpus 0,1,2,3
bash run_scripts/run_mistral_labels_from_opennyai_summaries_all.sh --gpus 0,1,2,3
bash run_scripts/run_merge_timeline_from_final_outputs.sh
```

| Script | Reads | Writes |
|--------|-------|--------|
| `run_ner_rr_all_categories.sh` | `../INPUT_DATA/*_text/` | `final_outputs/<bucket>_extract/annotations/` |
| `run_opennyai_summaries_all.sh` | `final_outputs/<bucket>_extract/annotations/` | `final_outputs/<bucket>_summary_opennyai/enriched_jsons/` |
| `run_mistral_labels_from_opennyai_summaries_all.sh` | `final_outputs/<bucket>_summary_opennyai/enriched_jsons/` | `final_outputs/<bucket>_labelled_mistral/labelled_jsons/` |
| `run_merge_timeline_from_final_outputs.sh` | `final_outputs/<bucket>_labelled_mistral/labelled_jsons/` | `../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/<bucket>_timed_mistral/` |

> The Mistral wrapper uses the `llm` micromamba environment and requires a Hugging Face token
> (`HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, or `--hf-token`).

---

## 🧱 Dataset-Building Wrappers

These consume the merged Timeline Maker folders and assemble the cross-bucket corpora used
by `section_GNN`:

| Script | Effect |
|--------|--------|
| `run_build_cross_bucket_cases.sh` | Copies up to **8 000** sorted case JSONs per bucket into `Timeline_Maker/cross_bucket_cases_8k_each_mistral/`. |
| `run_build_cross_bucket_remaining_cases.sh` | Copies everything **after** the first 8 000 per bucket into `Timeline_Maker/cross_bucket_cases_remaining_after_8k_each_mistral/`. |

---

## 🔬 Optional Labelling Wrappers

Call scripts in [`../extra_scripts/`](../extra_scripts/README.md); not required for the main chain:

| Script | Effect |
|--------|--------|
| `run_crossval_all_buckets.sh` | Audit/validation labeler → `../cross_validated_outputs/<bucket>/`. |
| `run_mistral_multi_labels_from_opennyai_summaries_all.sh` | Experimental multi-label labeler → separate multi-label folders under `../final_outputs/`. |

---

## 🔧 Useful Overrides

Most wrappers accept flags and environment variables: `CONDA_ENV`, `GPUS`,
`MODEL` / `MODEL_ID`, `BASE_INPUT`, `BASE_OUTPUT`, `FINAL_OUTPUTS_DIR`, `TIMELINE_ROOT`,
`OUTPUT_DIR`. Check each script's header before a large run.

---

⬆️ Back to [`Fixed_GPU_OpenNyai/`](../README.md)
