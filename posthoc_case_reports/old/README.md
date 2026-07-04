# 🗄️ old — Legacy Report Scripts

> Part of [`posthoc_case_reports/`](../README.md) · **archive** — kept isolated from the
> active pipeline code.

## Subfolders

| Folder | Contents |
|--------|----------|
| `aggregate_analysis/` | Built the aggregate test-set CSV from the legacy Graph_Analyser inference predictions and Phase-4 PGExplainer JSONs (no LLM stage). |
| `early_detection/` | Built multi-hearing case paths and hearing-level CSVs from `section_GNN/multi_hearing_stage_test` outputs. |

## Historical Run Order

```bash
python aggregate_analysis/build_aggregate_test_csv.py
python early_detection/build_early_detection_csvs.py
python aggregate_analysis/analyze_aggregate_visuals.py
python early_detection/analyze_early_detection_visuals.py
```

Outputs were written back into the same subfolders. These scripts reference the archived
`Graph_Analyser` tooling (now under `DUMP_MISC/`) — prefer the current
[`FINAL_EXPLANATION/`](../../FINAL_EXPLANATION/README.md) pipeline for new analyses.

---

⬆️ Back to [`posthoc_case_reports/`](../README.md)
