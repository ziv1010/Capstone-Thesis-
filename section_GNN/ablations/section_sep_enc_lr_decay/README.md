# 📑 section_sep_enc_lr_decay — Section-Separated + LR Decay

> Part of [`ablations/`](../README.md).

Section-separated encoding configs under the **LR-decay training schedule**, so
section-separated graphs can be compared against the other LR-decay thesis-table cells under
the same optimizer settings. Combined with entity-resolved data, this variant underlies the
model explained in `FINAL_EXPLANATION/`.

## ▶️ Run

Launched by the higher-level table scripts rather than manually:

```bash
bash run_scripts/run_party_args_preamble_and_section_sep_lr_decay.sh
bash run_scripts/run_remaining_table_experiments_8gpu.sh
```

## 🔗 Relationship

[`section_sep_enc/`](../section_sep_enc/README.md) is the base ablation; this folder keeps
the same graph idea and changes only the training schedule.

---

⬆️ Back to [`ablations/`](../README.md)
