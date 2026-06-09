# Outputs

This folder contains generated explanation, validation, pattern-analysis,
full-graph-analysis, visualizer, and traceability-report artifacts.

It is large and mostly ignored by Git. Keep reusable documentation here, but do
not commit bulk CSV/JSON/HTML output unless there is a specific reason.

## Important Output Families

- `entity_resolved_section_sep_lr_decay_cross_bucket_fold00/`
  - Main current explanation and validation output directory.

- `entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why/`
  - Main current pattern-level community, embedding, and opposite-case output
    directory.

- `entity_resolved_section_sep_lr_decay_cross_bucket_full_graph/`
  - Main current full-graph community, bridge, hub, and authority output
    directory.

- `traceability_reports_sample/`
  - One-case report sample used for checking report rendering.

- `traceability_reports_all/`
  - Full traceability-report batch.

- `benchmark_*`, `smoke_*`, `validation_*`, `pattern_why_*`, and `target_*`
  - Earlier benchmark, smoke-test, validation, or working output directories.

- `old/`
  - Historical output material kept separate from the active output families.

## Common Files

Explanation directories usually contain:

```text
case_summary.csv
case_top_explanations.csv
case_counterfactual_groups.csv
typed_path_importance.csv
relation_type_importance.csv
evidence_type_importance.csv
manifest.json
run_summary.json
```

Validation directories usually contain:

```text
faithfulness_curves.csv
faithfulness_auc_by_case.csv
faithfulness_auc_summary.csv
prediction_bucket_cases.csv
prediction_bucket_summary.csv
prediction_bucket_evidence_types.csv
validation_manifest.json
validation_run_summary.json
```

Traceability report directories usually contain:

```text
index.html
cases/
graphs/
dot/
json/
manifest.json
```
