# 📜 scripts — Multi-Hearing Experiment Steps

> Part of [`multi_hearing_stage_test/`](../README.md) · run everything from `section_GNN/`.

## 🔢 Step Order

| # | Script | Effect |
|---|--------|--------|
| 1 | `01_prepare_input.py` | Copies and tags multi-hearing JSONs as stage files. |
| 2 | `02_preprocess.sh` | Runs fixed-open preprocessing on the staged inputs. |
| 3 | `03_build_graph.sh` | Builds the stage graph. |
| 4 | `04_run_inference.py` | Runs the configured trained folds on stage cases. |
| 5 | `05_analyze_transitions.py` | Computes prediction transitions per base case. |
| 5b | `05b_aggregate_transitions.py` | Aggregates transition statistics. |
| 5c | `05c_per_case_factors.py` | Per-case prediction-transition factor reports. |
| 5d | `05d_raw_outcome_factors.py` | Raw-outcome transition factor reports. |
| 5e | `05e_early_signal_test.py` | Early-detection signal tables. |
| 6 | `06_explain_transitions.py` | Optional legacy explainability pass (uses the archived Graph_Analyser tooling). |

## ▶️ Run

```bash
bash multi_hearing_stage_test/scripts/run_all.sh    # main sequence
```

Use individual scripts when debugging a single stage; results are browsable in the
[⭐ Multi-Hearing Stage Test Visualiser](../visualiser/README.md) (port 8050).

---

⬆️ Back to [`multi_hearing_stage_test/`](../README.md)
