# 🗃️ Food Safety — Raw Case Material

> Part of [`INPUT_DATA/`](../README.md) · **Stage ①** of the pipeline.

Raw/collected Indian court-judgment material for **food-safety and regulatory prosecutions**.

- 📄 **Contents:** source PDFs / collected files from the original data-gathering step.
- ➡️ **Next step:** extract plain text with `INPUT_DATA/01_extract_pdf_text.py`; the active
  pipeline then consumes the sibling [`food_safety_text/`](../food_safety_text/README.md) folder.
- 🧩 **Role in the thesis:** held-out cross-domain evaluation domain — used by
  `section_GNN/cross_domain_test/food_safety/`, **not** part of the five-domain training set.
- 🚫 **Git policy:** local data only — this folder's contents are ignored by Git.
