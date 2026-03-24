# OpenNyAI Local Pipeline

Production-style local pipeline for Indian legal NLP over `.txt` court-judgment files using the `opennyai` Python library directly.

## What it runs

- `NER`
- `Rhetorical_Role`
- `Summarizer`

The pipeline uses:

- `ner_model_name="en_legal_ner_trf"`
- `ner_do_sentence_level=True`
- `ner_do_postprocess=True`
- `Summarizer` after `Rhetorical_Role`

## Project layout

```text
opennyai_pipeline/
  README.md
  requirements.txt
  .env.example
  run_pipeline.py
  inspect_output.py
  01_extract_pdf_text.py
  src/
  input_txt/
  outputs/
    combined/
    annotations/
    ner/
    rhetorical_roles/
    summaries/
    logs/
```

## Environment setup

Create the validated GPU environment with micromamba:

```bash
cd /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai
micromamba env create -f environment.gpu.yml
```

This creates the final environment:

```bash
fixed_gpu_opennyai_final
```

Notes:

- `spacy==3.2.4` is pinned.
- `pydantic==1.7.4` is pinned because `spacy==3.2.4` can fail to import with newer allowed `pydantic` builds in this environment.
- This environment uses CUDA-enabled PyTorch and CuPy, provisioned through micromamba-compatible channels.
- The first pipeline run downloads large OpenNyAI model assets. This project redirects those caches into local project folders instead of your home directory.
- `requirements.txt` is now only a supplemental pip list. For this folder, prefer `environment.gpu.yml`.

## GPU-only behavior

- Run with `--use_gpu`.
- This folder now aborts the run if `torch`, `CuPy`, or `spaCy` cannot activate GPU support.
- Silent CPU fallback is intentionally disabled when GPU mode is requested.
- Some orchestration work still uses CPU, such as file I/O, process startup, and JSON writing. The enforced part is model execution, preprocessing, and NER/RR/summarization backends.

## Convert PDFs to `.txt`

The sample PDFs are under:

```text
test_data/test/
```

Extract them into `input_txt/`:

```bash
micromamba run -n fixed_gpu_opennyai_final python 01_extract_pdf_text.py \
  --input_dir test_data/test \
  --output_dir input_txt \
  --overwrite
```

## Run the pipeline

Example:

```bash
micromamba run -n fixed_gpu_opennyai_final python run_pipeline.py \
  --input_dir test_data \
  --output_dir outputs/test_gpu \
  --glob_pattern "*.txt" \
  --use_gpu \
  --gpu_devices 0 \
  --pipeline_batch_size 2 \
  --batch_size 40000 \
  --summary_length 0.0 \
  --preprocessing_model en_core_web_trf \
  --overwrite
```

For multiple GPUs, increase workers and pin devices explicitly:

```bash
micromamba run -n fixed_gpu_opennyai_final python run_pipeline.py \
  --input_dir test_data \
  --output_dir outputs/test_gpu_multi \
  --glob_pattern "*.txt" \
  --use_gpu \
  --worker_processes 2 \
  --gpu_devices 0,1 \
  --pipeline_batch_size 2 \
  --overwrite
```

## Output folders

- `outputs/combined/`
  - Wrapped combined payloads that preserve the exact raw OpenNyAI result under `raw_result`, plus stable `file_id` metadata.
- `outputs/annotations/`
  - Cleaned sentence-level annotations with rhetorical role, summary inclusion, and sentence entities.
- `outputs/ner/`
  - Sentence-grouped entities, flat entity lists, and deduplicated `unique_statutes`, `unique_provisions`, and `unique_precedents`.
- `outputs/rhetorical_roles/`
  - Sentence text plus rhetorical role labels and aggregate role counts.
- `outputs/summaries/`
  - JSON summary payloads and plain-text summary views.
- `outputs/logs/`
  - Timestamped run logs and `run_report.json`.

## Helper inspection command

Inspect one combined JSON file:

```bash
micromamba run -n opennyai_py38 python inspect_output.py outputs/combined/<file_id>.json
```

It prints:

- number of sentences
- rhetorical role counts
- total entities
- summary sections present

## Notes on robustness

- Empty files are skipped and logged.
- Short files are still attempted, but warnings are logged.
- Failures are isolated per document so the remaining documents continue.
- The runner defensively filters unsupported OpenNyAI kwargs across versions.
- The runner works around two upstream issues in `opennyai==0.0.13`:
  - the broken wheel filename published for `en_legal_ner_trf`
  - underscore-sensitive internal `file_id` handling in OpenNyAI's combined pipeline output
