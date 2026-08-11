# Paper additional tests — Reviewers 3 and 6

The studies below answer experimental comments from Reviewers 3 and 6. They reuse the
published run's **graph and case-level splits**. **The HGT was not re-run** — its recorded
five-fold numbers are the reference row in every comparison.

| Study | Reviewer point | Folder |
|---|---|---|
| **R3-03** | *"...not against simpler baselines such as SVM on TF-IDF, Logistic Regression on entity counts, or XGBoost. The absolute improvement over a trivial baseline is not shown."* | [`R3_03_non_llm_baselines/`](R3_03_non_llm_baselines/) |
| **R3-04** | *"The paper uses HGT [8] but does not compare against simpler GNNs (GCN, GraphSAGE, GAT) on the same graph."* | [`R3_04_gnn_architecture_ablation/`](R3_04_gnn_architecture_ablation/) |
| **R6-02** | Add a genuine flat-text comparator and correct the parsed-only InLegalLlama denominator. | [`R6_02_flat_text_llm_denominator/`](R6_02_flat_text_llm_denominator/) |

The two remaining Reviewer #3 items — R3-01 (LLM-derived labels) and R3-02 (cross-jurisdiction
generalisation) — are not addressed here. R6-02 reuses the leakage-audited R3-03 flat-text
experiment and recomputes LLM metrics from the stored per-case generations.

## The shared anchor: one split, read not re-derived

Both studies take fold membership verbatim from the published run's own per-fold
`predictions.csv`:

```
section_GNN/outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models/
  ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold/fold_{00..04}/predictions.csv
```

Each file lists all 71,813 cases in graph-node order with `split ∈ {train, val, test}`. The
five test folds are pairwise disjoint and their union is the whole corpus, so pooling them
gives exactly one held-out prediction per case per model — which is also what makes the paired
McNemar tests in R3-04 possible. This is the same split InLegalLlama was evaluated on.

Reference numbers, from `kfold_summary.json` (`aggregate`):

| | Accuracy | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|
| LegalGraph-LJP (HGT) | 0.8063 ± 0.0055 | 0.8002 ± 0.0050 | 0.8855 ± 0.0050 |

## Run everything

```bash
# R3-03 (CPU, env `llm`, ~1.5 h)
bash R3_03_non_llm_baselines/run.sh

# R3-04 (GPU, env `thesis_work`, ~1 h on five free GPUs)
bash R3_04_gnn_architecture_ablation/run_arch_ablation.sh --gpus 1,2,4,5,6
micromamba run -n thesis_work python R3_04_gnn_architecture_ablation/make_tables.py

# R6-02 (CPU, reads completed artifacts; no training)
python R6_02_flat_text_llm_denominator/evaluate_r6_02.py
```

Deliverables per study: a drop-in `.tex` table, a `report_*.md` with the numbers and suggested
paper edits, per-fold `metrics.json`/`predictions.csv`, and an aggregate JSON shaped like
`kfold_summary.json`.

## Note on `8pg_paper.tex`

The R6-02 revision updates the paper directly. It also corrects the best HGT run's macro-F1
from 0.796 to the 0.8002 recorded in `kfold_summary.json`; 0.796 belongs to the
*Party+arguments+LR decay* run.
