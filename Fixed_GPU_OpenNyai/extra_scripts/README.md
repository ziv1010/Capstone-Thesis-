# 🔬 extra_scripts — Optional Labelers

> Part of [`Fixed_GPU_OpenNyai/`](../README.md) · audit & experimental labelers, **not** part
> of the main production chain.

| Script | Purpose | Wrapper | Writes to |
|--------|---------|---------|-----------|
| `add_case_outcome_labels_crossval_mistral.py` | **Audit/validation labeler** — multiple yes/no checks with deterministic aggregation, for verifying the main labels. | `../run_scripts/run_crossval_all_buckets.sh` | [`../cross_validated_outputs/`](../cross_validated_outputs/README.md) |
| `add_multi_label_outcome_from_enriched.py` | **Experimental multi-label labeler** — six binary outcome flags plus a final ternary outcome label. | `../run_scripts/run_mistral_multi_labels_from_opennyai_summaries_all.sh` | separate multi-label folders under `../final_outputs/` |

Both read the enriched OpenNyAI summary JSONs
(`../final_outputs/<bucket>_summary_opennyai/enriched_jsons/`).

## ⭐ Main Labeler (for reference)

The production labeler lives one level up:

```text
../add_case_outcome_labels_from_enriched.py     ← main entry point
../add_case_outcome_labels_mistral.py           ← shared classification logic it imports
```

---

⬆️ Back to [`Fixed_GPU_OpenNyai/`](../README.md)
