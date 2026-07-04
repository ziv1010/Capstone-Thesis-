# 🧼 src/preprocessing — Leakage-Safe Case Preparation

> Part of [`section_GNN/src/`](../README.md).

Reusable utilities that turn extracted case JSON data into **leakage-safe model inputs** — a
core methodological point of the thesis: the model must never see outcome-bearing text.

## 📄 Files

| File | Role |
|------|------|
| `extract.py` | Text/entity extraction helpers for case payloads. |
| `leakage.py` | Conservative leakage detection and masking utilities. |
| `loader.py` | Typed loading/conversion helpers for cleaned cases. |
| `normalize.py` | Canonicalization of entity names and labels. |

## 📤 What Preprocessing Produces

Every cleaned case handed to the graph builders contains:

- a stable `case_id` + source-file metadata
- raw and mapped outcome labels
- text sections: `preamble`, `facts`, `arguments`, `petitioner_arguments`,
  `respondent_arguments`, `other_lawyer_arguments`
- normalized entities grouped by semantic type
- leakage-audit metadata

The fixed-open entry point is
[`experiments/fixed_open_pipeline/preprocess_fixed_open.py`](../../experiments/fixed_open_pipeline/README.md),
which writes this schema to `data/.../processed/cleaned_cases/`.

## 🛡️ Leakage Policy

Outcome-bearing rhetorical roles are dropped or masked **before** graph construction —
typically `ANALYSIS`, `ISSUE`, `RATIO`, `RLC`, and `RPC` (the judge's reasoning and
conclusions). The exact role policy is set in each YAML config.

---

⬆️ Back to [`src/`](../README.md)
