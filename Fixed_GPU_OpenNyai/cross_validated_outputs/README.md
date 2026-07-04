# 🔁 cross_validated_outputs — Audit / Validation Labels

> Part of [`Fixed_GPU_OpenNyai/`](../README.md) · **optional generated data**, separate from
> the main production outputs in [`../final_outputs/`](../final_outputs/README.md).

Outcome labels produced by the **cross-validation labeler**, used to audit agreement with the
main Mistral labels rather than to feed the main pipeline.

## 🏭 Producer

```bash
bash ../run_scripts/run_crossval_all_buckets.sh
```

which drives `../extra_scripts/add_case_outcome_labels_crossval_mistral.py` — a labeler that
asks multiple yes/no checks and aggregates them deterministically.

## 🔄 Data Flow

- **Reads:** `../final_outputs/<bucket>_summary_opennyai/enriched_jsons/`
- **Writes:** `<bucket>/` (per-bucket labelled JSONs) and `logs/`
- **`label_comparison/`** holds agreement/difference artifacts comparing labelling approaches.

## 👀 Inspecting

The [Pipeline Stage Visualiser](../../STAGE_VISUALISER/README.md) (port `8053`) renders these
outputs side-by-side with the main pipeline stages.

---

⬆️ Back to [`Fixed_GPU_OpenNyai/`](../README.md)
