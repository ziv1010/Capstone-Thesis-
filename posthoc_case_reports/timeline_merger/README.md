# ⏱️ timeline_merger — Explanation × Timeline Merge Outputs

> Part of [`posthoc_case_reports/`](../README.md) · connects model explanations to
> case-stage timelines. Use after the main GNN and final-explanation outputs exist.

## 📄 Artifacts

| File / folder | Contents |
|---------------|----------|
| `stage_predictions.csv` | Per-stage model predictions. |
| `stage_transitions.csv` | Stage-to-stage prediction transitions. |
| `stage_case_factors.csv` / `stage_decisive_factors_long.csv` | Case-level decisive factors per stage. |
| `stage_raw_outcome_factors.csv` | Raw-outcome transition factors. |
| `timeline_aggregate_metrics.csv` / `timeline_overall_outputs.csv` / `timeline_conversion_summary.csv` | Timeline-level aggregates. |
| `influence_connectivity_summary/` | Influence & connectivity summaries. |

The stage-level inputs originate from
[`section_GNN/multi_hearing_stage_test/`](../../section_GNN/multi_hearing_stage_test/README.md).

---

⬆️ Back to [`posthoc_case_reports/`](../README.md)
