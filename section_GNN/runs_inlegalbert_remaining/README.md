# 🧩 runs_inlegalbert_remaining — Table-Completion Cells

> Part of [`section_GNN/`](../README.md) · InLegalBERT configs that fill the **remaining
> thesis-table cells** after the main matrix was run. Treat these as table-completion
> experiments, not the primary baseline matrix.

## 🗂️ Variant Folders

`central_section_lr_decay/` · `central_section_no_lr/` · `entity_section_lr_decay/` ·
`entity_section_no_lr/` · `party_args_preamble_lr_decay/` · `party_args_preamble_no_lr/` ·
`section_sep_lr_decay/`

Each contains per-bucket configs, normally launched via:

```bash
bash run_scripts/run_remaining_table_experiments_8gpu.sh
```

## 📤 Outputs

```text
data/inlegalbert_remaining/
outputs/inlegalbert_remaining/
```

---

⬆️ Back to [`section_GNN/`](../README.md)
