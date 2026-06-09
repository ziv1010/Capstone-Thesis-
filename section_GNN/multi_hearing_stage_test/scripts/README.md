# scripts

Step-by-step scripts for the multi-hearing stage-transition experiment.

## Order

1. `01_prepare_input.py`: copy and tag multi-hearing JSONs as stage files.
2. `02_preprocess.sh`: run fixed-open preprocessing on the staged inputs.
3. `03_build_graph.sh`: build the stage graph.
4. `04_run_inference.py`: run trained folds on stage cases.
5. `05_analyze_transitions.py`: compute prediction transitions.
6. `05b_aggregate_transitions.py`: aggregate transition statistics.
7. `05c_per_case_factors.py`: write per-case prediction-transition factors.
8. `05d_raw_outcome_factors.py`: write raw-outcome transition factors.
9. `05e_early_signal_test.py`: produce early-signal tables.
10. `06_explain_transitions.py`: optional Graph_Analyser explainability run.

`run_all.sh` runs the main sequence.

## Run

From `section_GNN`:

```bash
bash multi_hearing_stage_test/scripts/run_all.sh
```

Use individual scripts when debugging a single stage.
