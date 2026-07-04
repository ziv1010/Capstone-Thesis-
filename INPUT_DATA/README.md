# 📥 INPUT_DATA — Stage ① · Raw Judgment Text

> **Pipeline position:** **① INPUT_DATA** ▸ ② Fixed_GPU_OpenNyai ▸ ③ DATA_SET_BUILDER_AND_EXPLORER ▸ ④ section_GNN ▸ ⑤ FINAL_EXPLANATION

This folder is the **source side of the entire pipeline**: the raw and extracted court-judgment
material for each legal domain, plus the PDF → text extraction utility. Everything downstream —
OpenNyAI extraction, timeline merging, graph training, and explanation — starts from the plain
`.txt` judgment files stored here.

---

## 🗂️ Layout

Each legal domain appears in up to three forms:

| Folder pattern | Contents | Consumed by |
|----------------|----------|-------------|
| `<domain>/` | Raw/collected case material (PDFs, scraped files) from earlier collection steps. | `01_extract_pdf_text.py` |
| `<domain>_text/` | Plain UTF-8 `.txt` judgment files — **the active pipeline input**. | `Fixed_GPU_OpenNyai/run_scripts/run_ner_rr_all_categories.sh` |
| `<domain>_text_holdout_big/` | Large held-out text sets for auxiliary / cross-domain tests. | cross-domain workflows in `section_GNN/` |

### Domains

| Domain | Role in the thesis |
|--------|--------------------|
| `family_matrimonial` | Core GNN training bucket |
| `financial_fraud` | Core GNN training bucket (downstream bucket name: `fin_fraud`) |
| `land_property` | Core GNN training bucket (+ big holdout) |
| `motor_accidents` | Core GNN training bucket (+ big holdout) |
| `sexual_offences` | Core GNN training bucket |
| `food_safety` | **Held-out cross-domain evaluation** domain (+ big holdout) — not part of the five-domain training set |

---

## 🛠️ PDF → Text Extraction

`01_extract_pdf_text.py` converts court-judgment PDFs into clean UTF-8 text files using
PyMuPDF (`fitz`). It normalises whitespace, preserves readable line breaks, and skips
documents that yield fewer than 200 characters of text (scanned/invalid PDFs).

```bash
python INPUT_DATA/01_extract_pdf_text.py \
  --input-dir INPUT_DATA/<domain> \
  --output-dir INPUT_DATA/<domain>_text
```

---

## ▶️ Feeding the Pipeline

The normal entry point into Stage ② reads every `*_text/` folder here and writes OpenNyAI
annotations under `Fixed_GPU_OpenNyai/final_outputs/`:

```bash
cd Fixed_GPU_OpenNyai
bash run_scripts/run_ner_rr_all_categories.sh --gpus 0,1,2,3
```

---

## 📦 Git Policy

The raw and extracted corpora are large and **intentionally ignored by Git** — only the
extraction script and README files are versioned. Keep the data local, or restore it from
external storage before rerunning the pipeline.

---

⬆️ Back to the [repository root](../README.md) · Next stage: [`Fixed_GPU_OpenNyai/`](../Fixed_GPU_OpenNyai/README.md)
