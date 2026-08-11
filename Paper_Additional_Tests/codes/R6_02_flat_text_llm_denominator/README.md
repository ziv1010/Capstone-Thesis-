# R6-02 — genuine flat-text baseline and a fair LLM denominator

This study resolves Reviewer 6's second comment. It combines the already completed,
leakage-audited five-fold flat-text experiment in `R3_03_non_llm_baselines` with a
coverage-aware recomputation of the two cross-bucket InLegalLlama runs.

Run:

```bash
python3 evaluate_r6_02.py
```

The script reads existing per-case predictions; it does not train or download a model. It
asserts that its recomputed LLM metrics exactly match the corrected metrics emitted by
`model comparison/summarize_inlegalllama.py`.

Primary LLM metrics use all 14,363 evaluation cases. An unparseable generation is an
abstention and counts as an error. Conditional performance on parseable generations is kept
under the explicit `selective_*` names and is never presented as headline accuracy.

Outputs:

- `outputs/r6_02_results.json` — machine-readable evidence and provenance
- `outputs/table_r6_02.tex` — paper-ready comparison table
- `outputs/report_r6_02.md` — reviewer-response narrative
