# 📤 outputs — Generated Explanation Artifacts

> Part of [`FINAL_EXPLANATION/`](../README.md) · **generated data** — large and mostly
> ignored by Git. Do not commit bulk CSV/JSON/HTML output without a specific reason.

## ⭐ Main Output Families

| Directory | Contents |
|-----------|----------|
| `entity_resolved_section_sep_lr_decay_cross_bucket_fold00/` | **Main** explanation + validation outputs. |
| `entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why/` | **Main** pattern-level community, embedding, and opposite-case outputs. |
| `entity_resolved_section_sep_lr_decay_cross_bucket_full_graph/` | **Main** full-graph community, bridge, hub, and authority outputs. |
| `traceability_reports_sample/` | One-case report sample for checking rendering. |
| `traceability_reports_all/` | Full traceability-report batch. |
| `benchmark_*`, `smoke_*`, `validation_*`, `pattern_why_*`, `target_*` | Earlier benchmark / smoke / working directories. |
| `old/` | 🗄️ Historical output material. |

## 🗃️ Typical Contents

**Explanation directories:**

```text
case_summary.csv · case_top_explanations.csv · case_counterfactual_groups.csv
typed_path_importance.csv · relation_type_importance.csv · evidence_type_importance.csv
manifest.json · run_summary.json
```

**Validation directories:**

```text
faithfulness_curves.csv · faithfulness_auc_by_case.csv · faithfulness_auc_summary.csv
prediction_bucket_cases.csv · prediction_bucket_summary.csv · prediction_bucket_evidence_types.csv
validation_manifest.json · validation_run_summary.json
```

**Traceability report directories:**

```text
index.html · cases/ · graphs/ · dot/ · json/ · manifest.json
```

Browse everything interactively with the ⭐ Final Explanation Visualizer
(`bash ../run_scripts/run_visualizer.sh`, port **8899**).

---

⬆️ Back to [`FINAL_EXPLANATION/`](../README.md)
