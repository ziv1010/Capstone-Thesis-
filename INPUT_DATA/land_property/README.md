# 🗃️ Land & Property — Raw Case Material

> Part of [`INPUT_DATA/`](../README.md) · **Stage ①** of the pipeline.

Raw/collected Indian court-judgment material for **land, property, and revenue disputes (title, partition, tenancy, acquisition)**.

- 📄 **Contents:** source PDFs / collected files from the original data-gathering step.
- ➡️ **Next step:** extract plain text with `INPUT_DATA/01_extract_pdf_text.py`; the active
  pipeline then consumes the sibling [`land_property_text/`](../land_property_text/README.md) folder.
- 🧩 **Role in the thesis:** one of the five core GNN training buckets.
- 🚫 **Git policy:** local data only — this folder's contents are ignored by Git.
