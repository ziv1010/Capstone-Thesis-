# FINAL_EXPLANATION

Self-contained HGT explanation pipeline for the frozen legal graph model. It does not edit training code or old outputs.

Default target:

```bash
Capstone-Thesis-/FINAL_EXPLANATION/run_default.sh
```

The default run explains the fold-00 test cases for:

```text
section_GNN/outputs/timed_bucket_runs/cross_bucket_total_dataset/models/ablation_section_sep_enc_cross_bucket_kfold/kfold/fold_00/model.pt
```

To switch models, pass the matching artifacts:

```bash
micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/explain_hgt.py \
  --model-path /path/to/model.pt \
  --graph-cache /path/to/graph.pt \
  --config /path/to/config.yaml \
  --predictions-csv /path/to/predictions.csv \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/my_model \
  --split test \
  --overwrite
```

Main outputs:

- `case_counterfactual_groups.csv`: all typed counterfactual group masks.
- `case_top_explanations.csv`: top-k counterfactual groups per case.
- `connected_case_label_distribution.csv`: training WIN/LOSS neighbourhoods for surfaced evidence nodes.
- `typed_path_importance.csv`: aggregate meta-path/path-family importance.
- `relation_type_importance.csv`: aggregate relation masking importance.
- `evidence_type_importance.csv`: aggregate node/evidence type importance.
- `leakage_sensitivity_summary.csv`: judge/court/name sensitivity audit.
- `identity_shortcut_summary.csv`: held-out identity-only shortcut audit with split overlap, domain baseline deltas, and permutation p-values.
- `identity_shortcut_case_scores.csv`: per-test-case scores from train-label identity priors.
- `identity_shortcut_top_skewed_identities.csv`: normalized identity names with strong train-label skew that also appear in the eval split.
- `mask_sensitivity_summary.csv`: post-hoc inference stress test for no-judge, no-parties, no-lawyers, no-court, all-identity, and top-k hub-authority masks.
- `mask_sensitivity_by_domain.csv`: same masking audit broken down by domain bucket to identify the largest drops.
- `mask_sensitivity_hub_authorities.csv`: top-k hub authorities removed by the hub stress test.
- `attention_counterfactual_overlap.csv`: HGT attention vs counterfactual overlap diagnostic.
- `run_summary.json` and `manifest.json`: reproducibility metadata and output inventory.

Validation outputs, after running `validate_explanations.py` or `run_validation_multi_gpu.sh`:

- `faithfulness_curves.csv`: per-case sufficiency and comprehensiveness curve points for counterfactual, attention, and random rankings.
- `faithfulness_auc_by_case.csv`: per-case AUCs for the faithfulness curves.
- `faithfulness_auc_summary.csv`: paper-level comparison table across ranking methods.
- `prediction_bucket_cases.csv`: case-level high-confidence correct / high-confidence wrong / low-confidence assignment.
- `prediction_bucket_summary.csv`: bucket-level evidence purity, top-delta, support, and accuracy summary.
- `prediction_bucket_evidence_types.csv`: most common top evidence types per bucket.

Visualizer:

```bash
Capstone-Thesis-/FINAL_EXPLANATION/run_visualizer.sh \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --pattern-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --host 127.0.0.1 \
  --port 8899
```

Open:

```text
http://127.0.0.1:8899
```

The visualizer is organized experiment-by-experiment:

- `Summary`: high-level experiment map and headline metrics.
- `1. Faithfulness`: sufficiency/comprehensiveness AUCs, curves, and prediction buckets.
- `2. Evidence Signal`: connected training-case support and statistically label-discriminative evidence.
- `Identity Shortcuts`: train-label identity-prior audit for judges, parties, courts, and lawyers.
- `3. Communities`: Leiden legal communities, community evidence profiles, and success/failure regions.
- `4. Opposite Cases`: closest opposite-label training cases and structural evidence differences.
- `5. Embeddings`: HDBSCAN embedding clusters and structural-vs-embedding alignment.
- `Case Explorer`: per-case drilldown that joins counterfactual evidence with community, embedding cluster, and nearest opposite case.
- `Raw Tables`: direct CSV table browser for appendix and custom analysis.

It reads merged output CSVs server-side and does not load the large full counterfactual CSV into the browser.

If validation CSVs are present, the visualizer also shows sufficiency/comprehensiveness AUCs, curve plots, and the correct/wrong/low-confidence bucket breakdown.

If `outputs/pattern_why` is present, the visualizer adds a Patterns tab for Leiden communities, community success/failure, statistically label-discriminative evidence, nearest opposite-label case comparisons, and HGT embedding clusters.

For interpretation details, see [VISUALIZER_GUIDE.md](VISUALIZER_GUIDE.md).

Traceability reports:

```bash
# One-time environment creation.
micromamba create -y -f Capstone-Thesis-/FINAL_EXPLANATION/traceability_reports_env.yml

# Sample first: choose a high-impact legal-authority case and render one report.
micromamba run -n hgt_trace_reports python Capstone-Thesis-/FINAL_EXPLANATION/generate_traceability_reports.py \
  --explanation-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/traceability_reports_sample \
  --case-limit 1 \
  --overwrite

# Full batch after checking the sample.
micromamba run -n hgt_trace_reports python Capstone-Thesis-/FINAL_EXPLANATION/generate_traceability_reports.py \
  --explanation-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/traceability_reports_all \
  --all \
  --overwrite
```

Each traceability report writes machine JSON, a self-contained HTML report with an embedded PyVis typed subgraph, a standalone graph HTML, and a DOT file. When the graph cache and predictions CSV can be inferred from `run_summary.json`, the report also lists connected training-case provenance for each concrete evidence node.

Useful controls:

```bash
# Smoke test on 25 test cases.
Capstone-Thesis-/FINAL_EXPLANATION/run_default.sh --case-limit 25 --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/smoke_25

# Multi-GPU smoke. This runs 2 cases per shard, then merges shard outputs.
Capstone-Thesis-/FINAL_EXPLANATION/run_multi_gpu.sh \
  --gpus 0,1 \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/smoke_multigpu \
  -- --case-limit 2 --progress-every 1

# Faithfulness + prediction-bucket validation smoke.
Capstone-Thesis-/FINAL_EXPLANATION/run_validation_multi_gpu.sh \
  --gpus 0,1 \
  --explanation-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/validation_smoke \
  -- --case-limit 2 --k-values 0,1,2 --random-trials 1 --progress-every 1

# Identity shortcut audit for the entity-resolved fold-00 run.
Capstone-Thesis-/FINAL_EXPLANATION/run_identity_shortcut_audit.sh \
  --permutations 100

# Post-hoc identity and hub-removal masking audit for the entity-resolved fold-00 run.
MASK_AUDIT_DEVICE=cuda:0 Capstone-Thesis-/FINAL_EXPLANATION/run_mask_sensitivity_audit.sh

# In the mask-sensitivity summary, confidence_drop is the drop in probability
# assigned to the original unmasked predicted class. mean_confidence_drop is
# the change in max probability after masking.

# Full fold-00 test split on 5 GPUs. Run this yourself for the bulk output.
Capstone-Thesis-/FINAL_EXPLANATION/run_multi_gpu.sh \
  --gpus 0,1,2,3,4 \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_multigpu

# Full validation on all 8 GPUs, merged into the same directory used by the visualizer.
Capstone-Thesis-/FINAL_EXPLANATION/run_validation_multi_gpu.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --explanation-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  -- --k-values 0,1,2,3,5,10,20 --random-trials 3

# Explain all 71,813 cases rather than only fold test cases.
Capstone-Thesis-/FINAL_EXPLANATION/run_default.sh --split all --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_all_cases

# Disable the supporting attention diagnostic if you only need faithful counterfactuals.
Capstone-Thesis-/FINAL_EXPLANATION/run_default.sh --no-attention-audit
```

Multi-GPU layout:

- Each GPU gets a deterministic shard via `--num-shards` / `--shard-index`.
- Per-shard outputs are written to `OUTPUT_DIR/shards/shard_XX`.
- `merge_outputs.py` writes merged CSVs and recomputed aggregate tables into `OUTPUT_DIR`.
- `--case-limit` in `run_multi_gpu.sh -- ...` is per shard, so `--gpus 0,1 -- --case-limit 2` processes up to 4 cases total.

Method mapping:

- Typed counterfactual masking masks one legal evidence group at a time inside the model's L-hop receptive field and records probability deltas from the frozen HGT. The local field is the L-hop undirected node neighbourhood with all directed typed edges induced among those nodes, which preserves the original and reverse relation directions used by HGT.
- Connected-case label distribution traverses from surfaced evidence nodes to connected training cases and reports label support.
- Typed path tracing assigns each explanation group to a legal path family and aggregates importance.
- HGT attention audit records internal HGT attention and reports overlap with the counterfactual ranking as a diagnostic only.
- Post-hoc mask sensitivity runs full-graph inference with selected identity or hub-authority nodes disconnected by removing all incident typed edges. It does not retrain the model.

Relation naming note:

- A `rev_` prefix means HGT's reverse message-passing edge. For example, `precedent->rev_cites_precedent->arguments` is the reverse direction of `arguments->cites_precedent->precedent`, so read it as information flowing from the precedent node back into the argument node.

Pattern-level "why" analyses:

```bash
# Required once for Leiden community detection in the thesis_work environment.
micromamba run -n thesis_work python -m pip install igraph leidenalg

# Smoke test: communities + skew + HGT embeddings + all-GPU nearest opposite + HDBSCAN.
micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/structural_why_analysis.py \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why_smoke \
  --case-limit 512 --max-feature-cases 500 --top-k-neighbors 40

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/extract_hgt_case_embeddings.py \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why_smoke \
  --split all --case-limit 512 --device cuda:0

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/counterfactual_neighborhoods.py \
  --pattern-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why_smoke \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why_smoke \
  --case-limit 16 --gpus all

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/embedding_cluster_characterization.py \
  --pattern-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why_smoke \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why_smoke \
  --case-limit 512 --min-cluster-size 20 --n-jobs -1
```

Full run:

```bash
micromamba run -n thesis_work python -m pip install igraph leidenalg

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/structural_why_analysis.py \
  --explanation-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --split all --max-feature-cases 0 --top-k-neighbors 80

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/extract_hgt_case_embeddings.py \
  --explanation-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/target_fold_00_8gpu \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --split all --device cuda:0

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/counterfactual_neighborhoods.py \
  --pattern-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --query-split test --candidate-split train --gpus all

micromamba run -n thesis_work python Capstone-Thesis-/FINAL_EXPLANATION/embedding_cluster_characterization.py \
  --pattern-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --output-dir Capstone-Thesis-/FINAL_EXPLANATION/outputs/pattern_why \
  --min-cluster-size 80 --n-jobs -1
```

Main pattern outputs:

- `case_communities.csv`, `community_profiles.csv`, `community_feature_profiles.csv`, `community_success_failure.csv`
- `evidence_label_skew.csv`, `case_top_explanations_with_skew.csv`
- `hgt_case_embeddings.npz`
- `counterfactual_neighborhoods.csv`, `counterfactual_neighborhood_feature_differences.csv`
- `case_embedding_clusters.csv`, `embedding_cluster_profiles.csv`, `community_embedding_splits.csv`, `structural_embedding_alignment.csv`
