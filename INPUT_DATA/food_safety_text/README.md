# 📄 Food Safety — Plain-Text Judgments

> Part of [`INPUT_DATA/`](../README.md) · **Stage ①** of the pipeline.

Plain UTF-8 `.txt` judgment files for **food-safety and regulatory prosecutions** — the direct input to the OpenNyAI
extraction stage.

- ⚙️ **Consumed by:** `Fixed_GPU_OpenNyai/run_scripts/run_ner_rr_all_categories.sh`
  (and `run_ner_rr_custom.py` for single-folder runs).
- 📤 **Produces:** OpenNyAI NER + rhetorical-role annotations under
  `Fixed_GPU_OpenNyai/final_outputs/food_safety_extract/annotations/`.
- 🧩 **Role in the thesis:** held-out cross-domain evaluation domain — used by
  `section_GNN/cross_domain_test/food_safety/`, **not** part of the five-domain training set.
- 🚫 **Git policy:** local data only — this folder's contents are ignored by Git.
