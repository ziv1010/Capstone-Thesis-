# Posthoc Case Reports

This folder is intentionally isolated from the existing training, inference,
and explainability pipeline code.

Subfolders:

- `aggregate_analysis/` builds the aggregate test-set CSV from
  `Graph_Analyser/outputs/phase1_2_inference/predictions.csv` and Phase 4
  PGExplainer JSONs. It does not use the LLM stage.
- `early_detection/` builds multi-hearing case paths and hearing-level CSVs
  from `section_GNN/multi_hearing_stage_test` outputs.

Run from anywhere:

```bash
python aggregate_analysis/build_aggregate_test_csv.py
python early_detection/build_early_detection_csvs.py
python aggregate_analysis/analyze_aggregate_visuals.py
python early_detection/analyze_early_detection_visuals.py
```

Outputs are written back into the same subfolders as the scripts.
