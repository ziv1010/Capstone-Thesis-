# 📄 Sexual Offences — Plain-Text Judgments

> Part of [`INPUT_DATA/`](../README.md) · **Stage ①** of the pipeline.

Plain UTF-8 `.txt` judgment files for **sexual-offence prosecutions** — the direct input to the OpenNyAI
extraction stage.

- ⚙️ **Consumed by:** `Fixed_GPU_OpenNyai/run_scripts/run_ner_rr_all_categories.sh`
  (and `run_ner_rr_custom.py` for single-folder runs).
- 📤 **Produces:** OpenNyAI NER + rhetorical-role annotations under
  `Fixed_GPU_OpenNyai/final_outputs/sexual_offences_extract/annotations/`.
- 🧩 **Role in the thesis:** one of the five core GNN training buckets.
- 🚫 **Git policy:** local data only — this folder's contents are ignored by Git.
