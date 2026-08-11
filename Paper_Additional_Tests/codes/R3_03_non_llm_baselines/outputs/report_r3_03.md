# R3-03 — non-LLM baselines and the absolute improvement over a trivial baseline

> *Reviewer 3: "The paper compares against InLegalLlama (generative LLM) and text-only
> ablations, but not against simpler baselines such as SVM on TF-IDF, Logistic Regression on
> entity counts, or XGBoost. The absolute improvement over a trivial baseline is not shown."*

## Protocol

All 71,813 cases, the **same five folds as the HGT run** — read verbatim from
its own `fold_XX/predictions.csv`, not re-derived. The five test folds were asserted at runtime
to be pairwise disjoint and to cover the corpus exactly, and every cached label was checked
against `predictions.csv` (0 mismatches).

Features come from the **leakage-controlled cleaned cases the HGT itself consumes**:
rhetorical roles `ANALYSIS/ISSUE/NONE/RATIO/RLC/RPC` already removed. Before TF-IDF, all
18,301 surviving
`[LEAKAGE_MASK]` artifacts and direct operative-outcome terms are removed. Runtime assertions
require zero such terms in both the sanitized corpus and every fitted vocabulary.

Per fold: vectorisers, scalers, SVD and authority vocabularies are fit on `split == "train"`
rows only; the single regularisation hyperparameter is chosen on `split == "val"` macro-F1;
`split == "test"` is scored once. Class weighting is on everywhere, matching the HGT's balanced
cross-entropy. Metrics come from `section_GNN/src/training/metrics.py::compute_metrics`, the
same function that produced the paper's numbers.

Excluded as features by construction: `metadata.source_label_*` (literal copies of the label),
`leakage_audit.dropped_sentence_role_counts` (an RPC-length proxy), and the entity fields
`global_case_frequency` / `degree` / `is_shared_node` (corpus-level statistics computed over
test cases).

## Results (5-fold mean ± population s.d.)

| Model | Features | Acc (%) | Macro-F1 | Δ vs majority | Δ vs HGT |
|---|---|---:|---:|---:|---:|
| Majority class | -- | 60.83 ± 0.00 | 0.3782 ± 0.0000 | +0.00 | +19.80 |
| Stratified random | -- | 52.35 ± 0.28 | 0.4997 ± 0.0022 | -8.48 | +28.28 |
| Logistic Regression | entity counts + case scalars | 60.20 ± 0.28 | 0.5962 ± 0.0034 | -0.64 | +20.44 |
| Linear SVM | sanitized TF-IDF (1--2 gram) | 83.13 ± 0.25 | 0.8251 ± 0.0021 | +22.30 | -2.50 |
| Logistic Regression | sanitized TF-IDF (1--2 gram) | 83.12 ± 0.34 | 0.8245 ± 0.0038 | +22.29 | -2.49 |
| XGBoost | entity counts + authority counts | 76.89 ± 0.33 | 0.7620 ± 0.0037 | +16.06 | +3.74 |
| XGBoost | sanitized TF-IDF $\to$ SVD-256 | 82.61 ± 0.28 | 0.8180 ± 0.0029 | +21.77 | -1.98 |
| **LegalGraph-LJP (HGT, published)** | heterogeneous case graph | **80.63 ± 0.55** | **0.8002 ± 0.0050** | +19.80 | — |

## Headline answers to the reviewer

- **Absolute improvement over the trivial baseline.** The majority-class baseline scores 60.83% accuracy / 0.3782 macro-F1 (the corpus is 60.8% positive). LegalGraph-LJP scores 80.63% / 0.8002, i.e. **+19.80 accuracy points and +0.4219 macro-F1** over chance-level prediction.
- **⚠️ The strongest non-LLM baseline outperforms the GNN.** Linear SVM on sanitized TF-IDF (1--2 gram) reaches 83.13% / 0.8251, i.e. **2.50 accuracy points above** LegalGraph-LJP's 80.63%. This is reported as found after the TF-IDF corpus and fitted vocabulary both passed zero-tolerance checks for mask artifacts and direct operative-outcome terms.
- For reference, InLegalLlama (CPT) reaches 33.2% / 0.391 on the full denominator (unparseable generations count as incorrect abstentions), with 53.2% coverage and parsed-only diagnostic 62.5% / 0.565.
- For reference, InLegalLlama (SFT) reaches 48.4% / 0.358 on the full denominator (unparseable generations count as incorrect abstentions), with 78.7% coverage and parsed-only diagnostic 61.5% / 0.396.

## Where the flat-text signal actually comes from

Same folds, same sanitizer, same Linear SVM as B3 — only the sections given to TF-IDF change.
This separates pre-decision case-type base rates (the preamble carries party names, court and
petition type) from the case narrative (facts, arguments).

| Sections used | Acc (%) | Macro-F1 | Features | Δ vs majority | Δ vs HGT |
|---|---:|---:|---:|---:|---:|
| preamble only | 74.45 ± 0.44 | 0.7354 | 103,176 | +13.61 | -6.19 |
| facts only | 75.27 ± 0.35 | 0.7489 | 212,159 | +14.44 | -5.36 |
| arguments only | 74.97 ± 0.23 | 0.7481 | 276,666 | +14.14 | -5.66 |
| facts + arguments | 81.41 ± 0.13 | 0.8092 | 300,000 | +20.57 | +0.77 |
| all three (fixed C) | 82.97 ± 0.13 | 0.8241 | 300,000 | +22.14 | +2.34 |

**Reading.** The preamble alone reaches 74.45% — +13.61 points over the
majority class from party names, court and petition type, with no case narrative at all. That is a
registry-level prior, the same shortcut family the paper's `no_names` ablation and identity-shortcut
audits already police on the GNN side — so it is worth checking whether the flat-text baseline simply
exploits it.

It does not. The preamble is largely redundant with the narrative: adding it on top of facts and
arguments is worth only +1.56 points. Facts and arguments alone — every party
name, court and petition type stripped — already reach 81.41%,
+0.77 against the HGT.

This matters for how the result is framed. The flat-text advantage is **not** an identity shortcut;
it comes from the case narrative, which is precisely the evidence the graph model is supposed to
reason over. The implication is that the frozen general-purpose text encoder, not the graph, is the
binding constraint.

## Suggested change to the paper

1. Add Table `tab:non-llm-baselines` and one paragraph stating the absolute gain over the
   majority-class baseline and over the strongest classical model.
2. Replace the old flat-text limitation with the result that the strongest TF-IDF model
   exceeds HGT, while retaining the missing fine-tuned-transformer limitation.
3. Do not include the raw-judgment oracle in the paper table; it contains decision text and
   is not a valid baseline.
