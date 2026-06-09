# multi_hearing_stage_test

This experiment tests how predictions change across multiple hearings of the
same case.

## Goal

The workflow builds stage-tagged JSON files for cases with multiple hearings,
preprocesses them into the normal graph format, runs a trained GNN checkpoint
on the stage graph, and analyzes whether predictions change from earlier to
later hearings.

## Main Files

- `config.yaml`: graph/inference configuration.
- `scripts/01_prepare_input.py`: builds stage-tagged input JSONs.
- `scripts/02_preprocess.sh`: preprocesses stage-tagged JSONs.
- `scripts/03_build_graph.sh`: builds the stage graph.
- `scripts/04_run_inference.py`: runs configured trained folds on the stage graph.
- `scripts/05_analyze_transitions.py`: groups stage predictions by base case.
- `scripts/05b_aggregate_transitions.py`: aggregate transition statistics.
- `scripts/05c_per_case_factors.py`: per-case prediction-transition factor reports.
- `scripts/05d_raw_outcome_factors.py`: raw-outcome transition reports.
- `scripts/05e_early_signal_test.py`: early-detection signal tables.
- `scripts/06_explain_transitions.py`: optional Graph_Analyser explainability run.
- `scripts/run_all.sh`: full workflow wrapper.
- `visualiser/`: small app for browsing transition results.

## Run

From `section_GNN`:

```bash
bash multi_hearing_stage_test/scripts/run_all.sh
```

or step by step:

```bash
python multi_hearing_stage_test/scripts/01_prepare_input.py
bash multi_hearing_stage_test/scripts/02_preprocess.sh
bash multi_hearing_stage_test/scripts/03_build_graph.sh
python multi_hearing_stage_test/scripts/04_run_inference.py
python multi_hearing_stage_test/scripts/05_analyze_transitions.py
```

## Outputs

Generated artifacts live under:

```text
multi_hearing_stage_test/data/
multi_hearing_stage_test/outputs/
```

Key outputs:

- `outputs/stage_manifest.csv`
- `outputs/inference/predictions.csv`
- `outputs/analysis/stage_transitions.csv`
- `outputs/analysis/transition_counts.json`
- `outputs/analysis/per_case_diffs/`
