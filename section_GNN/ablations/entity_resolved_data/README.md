# 🔗 entity_resolved_data — Canonicalized-Entity Ablation

> Part of [`ablations/`](../README.md).

Trains on the **entity-resolved dataset** produced by
[`DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/entity_resolver/`](../../../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/entity_resolver/README.md)
instead of the baseline entity extraction — testing the effect of statute/provision/precedent
canonicalization on graph quality. The entity-resolved **section-separated LR-decay
cross-bucket model** from this family is the one explained in `FINAL_EXPLANATION/`.

## 📄 Main Files

| File | Role |
|------|------|
| `prepare_configs.py` | Generates per-bucket configs for entity-resolved runs. |
| `preprocess_fixed_open_resolved.py` | Preprocessing entry point for resolved entity payloads. |
| `run_entity_resolved_data_ablation.sh` | Main launcher. |
| `run_section_sep_no_names_both_lr.sh` | Helper for combined section/no-name LR comparisons. |

## 🗂️ Configs & Outputs

Configs: `configs/` and `configs_no_lr/`, with `party`, `section`, and `section_no_names`
subfolders naming the paired graph variant.
Outputs: `data/ablations/entity_resolved_data/` and `outputs/ablations/entity_resolved_data/`.

---

⬆️ Back to [`ablations/`](../README.md)
