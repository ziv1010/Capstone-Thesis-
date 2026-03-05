# Indian Court PDF Extraction Pipeline

Reproducible Python pipeline for born-digital Indian court judgments/orders.

## Features
- PDF text extraction in page order using PyMuPDF (`fitz`) with optional `pdfplumber` mode
- Deterministic paragraphization (blank lines + indentation + numbered/bullet starts)
- spaCy NER extraction (`PERSON`, `ORG`, `GPE`, `DATE`)
- Regex extraction for provisions, statutes, and precedents/citations
- Local Hugging Face LLM client (`Qwen/Qwen2.5-32B-Instruct` by default)
- Strict JSON schema validation with one automatic retry for JSON-fix
- Deterministic leakage firewall for `ml.input_text`
- Exports:
  - per-case JSON: `outputs/json/<case_id>.json`
  - aggregated JSONL: `outputs/cases.jsonl`
  - aggregated CSV: `outputs/cases.csv`

## Project Layout
- `src/main.py` CLI entrypoint
- `src/pdf_extract.py` PDF extraction
- `src/paragraphize.py` deterministic paragraph splitting
- `src/ner_extract.py` spaCy loading + NER
- `src/legal_regex.py` provisions/statutes/precedents regex extractors
- `src/llm_client.py` pluggable local LLM caller + prompt template
- `src/schema.py` strict schema and validation
- `src/postprocess.py` leakage firewall + ML input assembly
- `src/export.py` JSON/JSONL/CSV exporters
- `src/utils.py` shared helpers
- `configs/config.yaml` runtime config

## Setup
Use Python 3.11 or 3.12 for best spaCy compatibility.

```bash
micromamba create -y -n thesis_work python=3.11 pip
micromamba run -n thesis_work pip install -r requirements.txt
micromamba run -n thesis_work python -m spacy download en_core_web_sm
```

## Input
Default input directory is `./data/pdfs/`.

Run with explicit directory:
```bash
micromamba run -n thesis_work python -m src.main \
  --pdf_dir ./data/pdfs \
  --out_dir ./outputs \
  --config ./configs/config.yaml \
  --cuda_visible_devices 6,7
```

If your PDFs are currently in `./data/`, run:
```bash
micromamba run -n thesis_work python -m src.main \
  --pdf_dir ./data \
  --out_dir ./outputs \
  --config ./configs/config.yaml \
  --cuda_visible_devices 6,7
```

## Skip LLM Execution (parser + NER + regex only)
```bash
micromamba run -n thesis_work python -m src.main \
  --pdf_dir ./data \
  --out_dir ./outputs \
  --config ./configs/config.yaml \
  --skip_llm
```

## Notes
- `ml.input_text` is intended for training and excludes decision/outcome text.
- Any leaked outcome phrases detected in `ml.input_text` are removed and audited in `ml.removed_spans` with `ml.leakage_flag`.
