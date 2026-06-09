# Fixed GPU OpenNyAI Pipeline

This folder contains the OpenNyAI-based preprocessing pipeline used to turn raw
court judgment text files into structured case JSONs for the thesis experiments.

The active pipeline does three main things:

1. Run OpenNyAI NER and rhetorical-role extraction.
2. Add OpenNyAI summaries to the extracted JSONs.
3. Add Mistral-based case outcome labels to the enriched summary JSONs.

The output can then be merged into the timeline/case format used by
`DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker`.

## Current Folder Layout

```text
Fixed_GPU_OpenNyai/
  README.md
  environment.gpu.yml
  run_ner_rr_custom.py
  run_opennyai_summarizer_custom.py
  add_case_outcome_labels_mistral.py
  add_case_outcome_labels_from_enriched.py
  src/
    config.py
    io_utils.py
    output_formatter.py
    pipeline_runner.py
    validators.py
  run_scripts/
    run_ner_rr_all_categories.sh
    run_opennyai_summaries_all.sh
    run_mistral_labels_from_opennyai_summaries_all.sh
    run_merge_timeline_from_final_outputs.sh
    run_build_cross_bucket_cases.sh
    run_build_cross_bucket_remaining_cases.sh
    run_crossval_all_buckets.sh
    run_mistral_multi_labels_from_opennyai_summaries_all.sh
  extra_scripts/
    add_case_outcome_labels_crossval_mistral.py
    add_multi_label_outcome_from_enriched.py
  final_outputs/
  cross_validated_outputs/
  run_logs/
  .cache/
  .runtime_home/
```

## Main Scripts

- `run_ner_rr_custom.py`
  - Runs OpenNyAI named-entity recognition and rhetorical-role extraction over
    `.txt` files.
  - Writes per-case JSONs into an `annotations/` folder inside the selected
    output directory.

- `run_opennyai_summarizer_custom.py`
  - Reads extracted annotation JSONs.
  - Adds OpenNyAI summary fields.
  - Writes enriched JSONs into `enriched_jsons/`.

- `add_case_outcome_labels_from_enriched.py`
  - Main current labeler for enriched OpenNyAI summary JSONs.
  - Writes labelled JSONs into `labelled_jsons/`.
  - Imports shared Mistral classification logic from
    `add_case_outcome_labels_mistral.py`.

- `add_case_outcome_labels_mistral.py`
  - Shared Mistral outcome-classification implementation.
  - Keep this file in the root folder because the main enriched labeler imports
    from it.

## Optional Scripts

The scripts in `extra_scripts/` are not required for the main pipeline.

- `extra_scripts/add_case_outcome_labels_crossval_mistral.py`
  - Audit/validation labeler using multiple yes/no checks.
  - Used by `run_scripts/run_crossval_all_buckets.sh`.

- `extra_scripts/add_multi_label_outcome_from_enriched.py`
  - Experimental richer labeler that produces six binary outcome flags plus a
    final outcome label.
  - Used by `run_scripts/run_mistral_multi_labels_from_opennyai_summaries_all.sh`.

## Environment Setup

Create the validated GPU environment with micromamba:

```bash
cd Fixed_GPU_OpenNyai
micromamba env create -f environment.gpu.yml
```

This creates:

```bash
fixed_gpu_opennyai_final
```

Important environment notes:

- `spacy==3.2.4` is pinned.
- `pydantic==1.7.4` is pinned because newer `pydantic` builds can break this
  old spaCy/OpenNyAI stack.
- CUDA-enabled PyTorch and CuPy are installed through micromamba-compatible
  channels.
- The first run downloads large OpenNyAI model assets.
- Project-local cache paths are used so model/cache files do not have to live in
  the user's home directory.

## GPU Behavior

Use `--use_gpu` for OpenNyAI extraction and summarization runs.

When GPU mode is requested, the pipeline checks that `torch`, `CuPy`, and spaCy
can activate GPU support. Silent CPU fallback is intentionally avoided for the
model-heavy stages.

Some orchestration work still runs on CPU, including file I/O, process startup,
JSON parsing, and JSON writing.

## Main End-to-End Run Order

Run these from `Fixed_GPU_OpenNyai/`.

### 1. NER and Rhetorical Roles

```bash
bash run_scripts/run_ner_rr_all_categories.sh --gpus 0,1,2,3
```

This reads raw text from repository-relative `INPUT_DATA/*_text` folders and
writes:

```text
final_outputs/<bucket>_extract/annotations/
```

### 2. OpenNyAI Summaries

```bash
bash run_scripts/run_opennyai_summaries_all.sh --gpus 0,1,2,3
```

This reads:

```text
final_outputs/<bucket>_extract/annotations/
```

and writes:

```text
final_outputs/<bucket>_summary_opennyai/enriched_jsons/
```

### 3. Mistral Outcome Labels

```bash
bash run_scripts/run_mistral_labels_from_opennyai_summaries_all.sh --gpus 0,1,2,3
```

This reads:

```text
final_outputs/<bucket>_summary_opennyai/enriched_jsons/
```

and writes:

```text
final_outputs/<bucket>_labelled_mistral/labelled_jsons/
```

This script uses the `llm` micromamba environment by default and requires an
Hugging Face token through `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, or
`--hf-token`.

### 4. Merge Into Timeline Maker Format

```bash
bash run_scripts/run_merge_timeline_from_final_outputs.sh
```

This reads labelled Mistral outputs and writes merged case JSONs into:

```text
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/*_timed_mistral/
```

## Output Folders

- `final_outputs/`
  - Main generated artifacts from the current pipeline.
  - Contains extraction, summary, and labelled-output folders.

- `cross_validated_outputs/`
  - Optional audit/validation labels generated by the cross-validation labeler.
  - Not required for the main pipeline.

- `run_logs/`
  - Archived loose logs moved out of output directories during cleanup.

- `.cache/` and `.runtime_home/`
  - Runtime cache folders for model and GPU cache files.
  - These are reproducible caches, not thesis data artifacts.

- `final_outputs/*_extract/.worker_home_*`
  - Per-worker runtime homes created by parallel extraction runs.
  - These are also cache/runtime folders. Deleting them does not delete saved
    annotations, summaries, or labels, but reruns may recreate/download cache
    files.

## Single Custom Run Example

Use this when testing one input folder manually:

```bash
micromamba run -n fixed_gpu_opennyai_final python run_ner_rr_custom.py \
  --input_dir ../INPUT_DATA/financial_fraud_text \
  --output_dir final_outputs/test_fin_fraud_extract \
  --use_gpu \
  --gpus 0 \
  --pipeline_batch_size 1
```

## Path Reproducibility

The active shell wrappers resolve paths relative to the repository instead of
using machine-specific absolute paths. Environment variables can still override
defaults when needed:

- `CONDA_ENV`
- `GPUS`
- `BASE_INPUT`
- `BASE_OUTPUT`
- `FINAL_OUTPUTS_DIR`
- `TIMELINE_ROOT`
- `OUTPUT_DIR`
- `MODEL`, `MODEL_ID`

## Robustness Notes

- Empty files are skipped and logged.
- Short files are attempted, but warnings are logged.
- Failures are isolated per document so later documents can continue.
- The runner filters unsupported OpenNyAI kwargs across versions.
- The runner works around two upstream issues in `opennyai==0.0.13`:
  - the broken wheel filename published for `en_legal_ner_trf`
  - underscore-sensitive internal `file_id` handling in OpenNyAI combined
    pipeline output
