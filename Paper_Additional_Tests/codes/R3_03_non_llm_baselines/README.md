# R3-03 — non-LLM baselines and the absolute improvement over a trivial baseline

> **Reviewer 3:** *"The paper compares against InLegalLlama (generative LLM) and text-only
> ablations, but not against simpler baselines such as SVM on TF-IDF, Logistic Regression on
> entity counts, or XGBoost. The absolute improvement over a trivial baseline is not shown."*

## What this does

Trains seven valid classical/trivial models on the **same 71,813 cases and the same five
folds** as the published HGT run, and reports the absolute gain over a majority-class baseline.

| ID | Model | Features |
|---|---|---|
| B0 | Majority class | — (the "trivial baseline" the reviewer asks about) |
| B1 | Stratified random | — |
| B2 | **Logistic Regression on entity counts** | the GNN's own 12 case scalars + per-entity-type counts + retained-role histogram |
| B3 | **Linear SVM on TF-IDF** | sanitized word 1–2 grams, 300k features |
| B4 | Logistic Regression on TF-IDF | sanitized word 1–2 grams |
| B5 | **XGBoost** | entity counts + top-4000 canonical statute/provision/precedent counts |
| B6 | XGBoost on TF-IDF → SVD-256 | sanitized word 1–2 grams |
| B7 | ⚠️ **Optional diagnostic, never run/reported by default** | unfiltered raw judgment text |

## Why these inputs

Features are read from the **leakage-controlled cleaned cases the HGT itself consumes**
(`section_GNN/data/ablations/entity_resolved_data/cross_bucket_total_dataset/processed/cleaned_cases/`).
Rhetorical roles `ANALYSIS/ISSUE/NONE/RATIO/RLC/RPC` are already dropped there and 16 outcome
phrases already masked. The TXT source document is `preamble + facts + arguments` — exactly
the three sections the HGT case node encodes. A final deterministic TF-IDF guard then removes
the observable `[LEAKAGE_MASK]` artifact and direct operative-outcome vocabulary. This extra
guard is necessary because some retained PREAMBLE sentences contain cause-list markers such
as `[ALLOWED]` even though RPC/RLC/RATIO sentences were correctly removed.

The upstream annotated JSONs are deliberately *not* used as features: they still carry
`case_outcome_label`, `case_outcome_score`, `opennyai_summary["decision"]`,
`llm_case_outcome.{decision_text, rpc_texts, raw_model_response, crossval_*}` and every `RPC`
sentence. B7 is the single exception, and it reads raw text on purpose.

## Leakage controls

1. **Splits are read from the HGT run's own `fold_XX/predictions.csv`**, never re-derived. At
   runtime the five test folds are asserted pairwise disjoint and to cover all 71,813 cases,
   and every cached label is checked against `predictions.csv` (0 mismatches).
2. **Nothing is fit on val or test.** Vectorisers, scalers, SVD and authority vocabularies see
   `split == "train"` rows only; `val` picks the one hyperparameter; `test` is scored once.
3. **Label mirrors are blocked at the feature builder**: `metadata.source_label_field`,
   `source_label_value`, `source_decision_label`.
4. **`leakage_audit.dropped_sentence_role_counts` is excluded** — the RPC/RATIO counts proxy
   for decision length. Only `kept_sentence_role_counts` is used.
5. **Corpus-level entity statistics are excluded** — `global_case_frequency`, `degree`,
   `is_shared_node` are computed over test cases in the original artifact.
6. Metrics come from `section_GNN/src/training/metrics.py::compute_metrics`, the same function
   that produced the paper's numbers.

`audit_tfidf_inputs.py` checks all 71,813 rows against the cleaned cases and the HGT's own
`predictions.csv` artifacts. It fails unless case order, labels, retained sections and folds
match exactly, and unless the sanitized corpus has zero mask/outcome terms. Each fitted
vectorizer separately asserts that no forbidden feature entered its vocabulary.

## Run it

```bash
bash run.sh                 # build features + 7 valid rows + tables (~1.5 h, CPU)
bash run.sh --limit 2000    # smoke test on the first 2,000 cases
bash run.sh --skip-build    # reuse the cached features
```

Environment: **`llm`** — it already has scikit-learn 1.7.2 and xgboost 3.1.2, so nothing is
installed and the `thesis_work` env that produced the paper numbers is left untouched.

## Files

| File | Purpose |
|---|---|
| `build_features.py` | streams the 71,813 cleaned cases into TXT / ENT / AUTH / RAWTXT caches |
| `audit_tfidf_inputs.py` | full-corpus provenance, role, split and leakage audit |
| `text_sanitization.py` | deterministic final TF-IDF-only leakage guard |
| `run_baselines.py` | the valid models on the HGT's own folds |
| `make_tables.py` | LaTeX table + markdown report, incl. the InLegalLlama rows for context |
| `run.sh` | the whole pipeline |
| `outputs/features/` | cached features + `build_report.json` (leakage audit) |
| `outputs/per_fold/<model>/fold_XX/metrics.json` | per-fold metrics |
| `outputs/baselines_summary.json` | aggregate, shaped like `kfold_summary.json` |
| `outputs/top_features_*.json` | 40 highest-weight n-grams per class, per fold |
| `outputs/table_r3_03_baselines.tex` | drop-in LaTeX table |
| `outputs/report_r3_03.md` | results and suggested paper changes |
