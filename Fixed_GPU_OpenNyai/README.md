# ⚙️ Fixed_GPU_OpenNyai — Stage ② · Extraction & Labelling

> **Pipeline position:** ① INPUT_DATA ▸ **② Fixed_GPU_OpenNyai** ▸ ③ DATA_SET_BUILDER_AND_EXPLORER ▸ ④ section_GNN ▸ ⑤ FINAL_EXPLANATION

The GPU-hardened OpenNyAI pipeline that turns raw judgment `.txt` files into **structured,
outcome-labelled case JSONs**. It performs three transformations, each with its own script
and output family:

1. 🏷️ **NER + rhetorical roles** — OpenNyAI named-entity recognition and rhetorical-role
   segmentation over raw text.
2. 📝 **Summaries** — OpenNyAI extractive summaries added to the annotation JSONs.
3. ⚖️ **Outcome labels** — Mistral-based case outcome classification added to the enriched JSONs.

The labelled output is then merged into the timeline/case format by
[`DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/`](../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/README.md).

---

## 🗂️ Folder Layout

| Path | Role |
|------|------|
| `run_ner_rr_custom.py` | OpenNyAI NER + rhetorical-role extraction over `.txt` files → `annotations/`. |
| `run_opennyai_summarizer_custom.py` | Adds OpenNyAI summary fields → `enriched_jsons/`. |
| `add_case_outcome_labels_from_enriched.py` | **Main labeler** — adds Mistral outcome labels → `labelled_jsons/`. |
| `add_case_outcome_labels_mistral.py` | Shared Mistral classification implementation imported by the main labeler (must stay in this folder). |
| [`src/`](src/README.md) | Shared helper modules (config, I/O, formatting, orchestration, validation). |
| [`run_scripts/`](run_scripts/README.md) | Repo-relative shell wrappers for every stage — the recommended entry points. |
| [`extra_scripts/`](extra_scripts/README.md) | Optional audit / multi-label labelers (not part of the main chain). |
| [`final_outputs/`](final_outputs/README.md) | 📤 Main generated artifacts (`*_extract`, `*_summary_opennyai`, `*_labelled_mistral`). |
| [`cross_validated_outputs/`](cross_validated_outputs/README.md) | Optional audit/validation labels from the cross-validation labeler. |
| [`run_logs/`](run_logs/README.md) | Archived loose logs from earlier runs. |
| `environment.gpu.yml` | Micromamba spec for the validated GPU environment. |
| `.cache/`, `.runtime_home/` | Reproducible runtime/model caches — not thesis artifacts. |

---

## 🧪 Environment Setup

```bash
cd Fixed_GPU_OpenNyai
micromamba env create -f environment.gpu.yml   # creates: fixed_gpu_opennyai_final
```

Key constraints baked into the environment:

- `spacy==3.2.4` and `pydantic==1.7.4` are **pinned** — newer pydantic builds break this
  spaCy/OpenNyAI stack.
- CUDA-enabled PyTorch and CuPy are installed through micromamba-compatible channels.
- The first run downloads large OpenNyAI model assets into project-local caches, so nothing
  needs to live in the user's home directory.

The Mistral labelling stage uses a separate `llm` environment and needs a Hugging Face token
(`HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, or `--hf-token`).

---

## 🚀 Main End-to-End Run Order

Run from `Fixed_GPU_OpenNyai/`:

```bash
# 1️⃣ NER + rhetorical roles      INPUT_DATA/*_text  →  final_outputs/<bucket>_extract/annotations/
bash run_scripts/run_ner_rr_all_categories.sh --gpus 0,1,2,3

# 2️⃣ OpenNyAI summaries          annotations/  →  final_outputs/<bucket>_summary_opennyai/enriched_jsons/
bash run_scripts/run_opennyai_summaries_all.sh --gpus 0,1,2,3

# 3️⃣ Mistral outcome labels      enriched_jsons/  →  final_outputs/<bucket>_labelled_mistral/labelled_jsons/
bash run_scripts/run_mistral_labels_from_opennyai_summaries_all.sh --gpus 0,1,2,3

# 4️⃣ Merge into Timeline Maker   labelled_jsons/  →  ../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/*_timed_mistral/
bash run_scripts/run_merge_timeline_from_final_outputs.sh
```

### Single custom run (one input folder)

```bash
micromamba run -n fixed_gpu_opennyai_final python run_ner_rr_custom.py \
  --input_dir ../INPUT_DATA/financial_fraud_text \
  --output_dir final_outputs/test_fin_fraud_extract \
  --use_gpu --gpus 0 --pipeline_batch_size 1
```

---

## 🖥️ GPU Behaviour

Pass `--use_gpu` for extraction and summarization. When GPU mode is requested, the pipeline
**verifies** that `torch`, CuPy, and spaCy can all activate GPU support — silent CPU fallback
is deliberately avoided for the model-heavy stages. Orchestration (file I/O, process startup,
JSON parsing/writing) still runs on CPU.

---

## 🛡️ Robustness Notes

- Empty files are skipped and logged; short files are attempted with a warning.
- Failures are isolated **per document**, so one bad judgment never kills a batch.
- The runner filters unsupported OpenNyAI kwargs across library versions.
- Two upstream bugs in `opennyai==0.0.13` are worked around: the broken wheel filename
  published for `en_legal_ner_trf`, and underscore-sensitive internal `file_id` handling in
  the combined pipeline output.

---

## 🔧 Path Reproducibility & Overrides

All wrappers resolve paths **relative to the repository** — no machine-specific absolute
paths. Common overrides: `CONDA_ENV`, `GPUS`, `BASE_INPUT`, `BASE_OUTPUT`,
`FINAL_OUTPUTS_DIR`, `TIMELINE_ROOT`, `OUTPUT_DIR`, `MODEL` / `MODEL_ID`.

---

⬆️ Back to the [repository root](../README.md) · Previous: [`INPUT_DATA/`](../INPUT_DATA/README.md) · Next: [`DATA_SET_BUILDER_AND_EXPLORER/`](../DATA_SET_BUILDER_AND_EXPLORER/README.md)
