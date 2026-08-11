# R6-02 — flat-text baseline and corrected InLegalLlama denominator

> Reviewer 6: the text-only ablation is still an HGT case-star graph, and the
> InLegalLlama accuracy is computed only on its parseable subset.

## Resolution

The first point is resolved with genuine flat-document classifiers on the same 71,813 cases
and the same five case-level folds as the HGT. Inputs contain only the retained preamble,
facts, and argument text. The TF–IDF pipeline removes mask artifacts and direct operative-outcome
vocabulary, fits its vectorizer on training rows only, selects regularization on validation
macro-F1, and evaluates the held-out test rows once.

| Model | Accuracy | Macro-F1 | Difference from HGT |
|---|---:|---:|---:|
| Majority class | 60.83% | 0.3782 | -19.80 pp |
| Linear SVM, sanitized TF–IDF | **83.13%** | **0.8251** | **+2.50 pp** |
| Logistic regression, sanitized TF–IDF | 83.12% | 0.8245 | +2.49 pp |
| LegalGraph-LJP (HGT) | 80.63% | 0.8002 | — |

The result changes the paper's conclusion: the strongest flat-text baseline exceeds HGT by
2.50 accuracy points and 0.0249 macro-F1. The revision therefore
does not claim a predictive advantage over flat text; it frames the graph's contribution as
explicit typed structure and intervention-based auditability.

For InLegalLlama, an unparseable generation is now an abstention and is counted as an error in
the primary full-denominator metrics. Parsed-only performance is retained only as a selective
diagnostic:

| Model | n | Coverage | Full-denominator accuracy / macro-F1 | Parsed-only diagnostic |
|---|---:|---:|---:|---:|
| InLegalLlama (CPT) | 14,363 | 53.2% | **33.22% / 0.3914** | 62.48% / 0.5649 |
| InLegalLlama (SFT) | 14,363 | 78.7% | **48.42% / 0.3585** | 61.55% / 0.3959 |

The LLM runs cover only fold 0, whereas the flat-text and HGT values are five-fold means. The
revision states this explicitly and avoids treating the operational zero-shot run as an
exhaustive or fine-tuned LLM comparison.

## Artifacts and paper changes

- `outputs/r6_02_results.json`: machine-readable recomputation and provenance.
- `outputs/table_r6_02.tex`: drop-in table with coverage and both metric policies.
- `../../../../8pg_paper.tex`: abstract, results table/text, limitations, and conclusion revised.
- `../../../../model comparison/summarize_inlegalllama.py`: primary metrics now use the full denominator.
