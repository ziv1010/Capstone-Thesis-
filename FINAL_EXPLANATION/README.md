# FINAL_EXPLANATION

This folder contains the final post-hoc explanation, validation, visualization,
and paper-figure pipeline for the frozen HGT legal graph model.

It does not train models. It reads trained model artifacts from `../section_GNN`,
runs explanation and validation analyses, writes CSV/JSON/HTML outputs under
`outputs/`, and serves those outputs through a local visualizer.

## Folder Layout

```text
FINAL_EXPLANATION/
  README.md
  *.py
  traceability_reports_env.yml
  run_scripts/
  docs/
  figures/
  outputs/
  logs/
  paper/
  visualizer_static/
```

- `*.py`: explanation, validation, community, report, and visualizer modules.
- `run_scripts/`: shell launchers for common single-GPU, multi-GPU, audit, and
  visualizer workflows.
- `docs/`: human reading guides, including the visualizer guide.
- `figures/`: final PNG/PDF figures generated from output tables.
- `outputs/`: generated experiment outputs. This is large and mostly ignored by
  Git.
- `logs/`: loose runtime logs moved out of the root folder.
- `paper/`: LaTeX source and build artifacts for the experiment overview note.
- `visualizer_static/`: frontend assets served by `visualizer.py`.

## Main Pipeline

Run commands from the repository root unless noted otherwise.

The full current entity-resolved thesis explanation workflow is:

```bash
bash FINAL_EXPLANATION/run_scripts/run_entity_resolved_section_sep_lr_decay_cross_bucket_all.sh
```

That orchestrates:

1. Typed counterfactual HGT explanations.
2. Faithfulness and prediction-bucket validation.
3. Identity shortcut audit.
4. Pattern-level community and structural analyses.
5. HGT case embedding extraction.
6. Nearest opposite-label case analysis.
7. Embedding cluster characterization.
8. Full-graph community, bridge, and hub analysis.
9. Post-hoc identity and hub-removal masking audit.

The default output prefix is:

```text
outputs/entity_resolved_section_sep_lr_decay_cross_bucket
```

The main output directories derived from that prefix are:

```text
outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00/
outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why/
outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph/
```

## Smoke Test

For a small explanation run:

```bash
bash FINAL_EXPLANATION/run_scripts/run_default.sh \
  --case-limit 25 \
  --output-dir FINAL_EXPLANATION/outputs/smoke_25
```

For a small multi-GPU run:

```bash
bash FINAL_EXPLANATION/run_scripts/run_multi_gpu.sh \
  --gpus 0,1 \
  --output-dir FINAL_EXPLANATION/outputs/smoke_multigpu \
  -- --case-limit 2 --progress-every 1
```

For a small validation run:

```bash
bash FINAL_EXPLANATION/run_scripts/run_validation_multi_gpu.sh \
  --gpus 0,1 \
  --explanation-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --output-dir FINAL_EXPLANATION/outputs/validation_smoke \
  -- --case-limit 2 --k-values 0,1,2 --random-trials 1 --progress-every 1
```

## Main Outputs

Explanation outputs include:

- `case_counterfactual_groups.csv`: all typed counterfactual group masks.
- `case_top_explanations.csv`: top-k counterfactual groups per case.
- `connected_case_label_distribution.csv`: training label support around
  surfaced evidence nodes.
- `typed_path_importance.csv`: aggregate legal path-family importance.
- `relation_type_importance.csv`: aggregate relation masking importance.
- `evidence_type_importance.csv`: aggregate evidence type importance.
- `leakage_sensitivity_summary.csv`: judge, court, party, and lawyer
  sensitivity audit.
- `attention_counterfactual_overlap.csv`: attention versus counterfactual
  overlap diagnostic.
- `manifest.json` and `run_summary.json`: output inventory and run metadata.

Validation outputs include:

- `faithfulness_curves.csv`
- `faithfulness_auc_by_case.csv`
- `faithfulness_auc_summary.csv`
- `prediction_bucket_cases.csv`
- `prediction_bucket_summary.csv`
- `prediction_bucket_evidence_types.csv`

Pattern-level outputs include:

- `case_communities.csv`
- `community_profiles.csv`
- `community_feature_profiles.csv`
- `community_success_failure.csv`
- `evidence_label_skew.csv`
- `case_top_explanations_with_skew.csv`
- `hgt_case_embeddings.npz`
- `counterfactual_neighborhoods.csv`
- `counterfactual_neighborhood_feature_differences.csv`
- `case_embedding_clusters.csv`
- `embedding_cluster_profiles.csv`
- `community_embedding_splits.csv`
- `structural_embedding_alignment.csv`

Full-graph outputs include:

- `full_graph_node_communities_long.csv`
- `full_graph_node_communities_wide.csv`
- `full_graph_community_profiles_res_1.00.csv`
- `full_graph_community_authorities_res_1.00.csv`
- `bridge_authority_pairs_res_1.00.csv`
- `authority_role_classification_res_1.00.csv`
- `authority_role_summary_res_1.00.csv`
- `resolution_sweep_summary.csv`

## Visualizer

Start the local viewer with:

```bash
bash FINAL_EXPLANATION/run_scripts/run_visualizer.sh \
  --output-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --pattern-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why \
  --full-graph-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph \
  --host 127.0.0.1 \
  --port 8899
```

Then open:

```text
http://127.0.0.1:8899
```

The visualizer reads merged CSVs server-side and presents:

- summary metrics
- faithfulness curves
- evidence support
- identity shortcut audits
- communities
- opposite-label case comparisons
- embedding clusters
- case-level traceability drilldowns
- raw table browsing

For interpretation guidance, see `docs/VISUALIZER_GUIDE.md`.

## Traceability Reports

Create the report environment once:

```bash
micromamba create -y -f FINAL_EXPLANATION/traceability_reports_env.yml
```

Generate one sample report:

```bash
micromamba run -n hgt_trace_reports python FINAL_EXPLANATION/generate_traceability_reports.py \
  --explanation-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --output-dir FINAL_EXPLANATION/outputs/traceability_reports_sample \
  --case-limit 1 \
  --overwrite
```

Generate a full report batch:

```bash
micromamba run -n hgt_trace_reports python FINAL_EXPLANATION/generate_traceability_reports.py \
  --explanation-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --output-dir FINAL_EXPLANATION/outputs/traceability_reports_all \
  --all \
  --overwrite
```

Each report run writes machine JSON, case HTML, standalone graph HTML, DOT
files, and a browsable `index.html`.

## Paper Figures

Generate the final static figures from the existing output tables:

```bash
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_paper_figures.py
```

Figures are written to:

```text
figures/
```

## Path Reproducibility

The active scripts resolve paths from their own location:

- `APP_ROOT`: `FINAL_EXPLANATION/`
- `REPO_ROOT`: repository root
- `SECTION_GNN`: `../section_GNN` from this folder

This means the pipeline does not depend on the original machine path. Model,
graph, config, prediction, and output paths can still be overridden with command
line flags or environment variables documented in `run_scripts/README.md`.

## Interpretation Notes

- Counterfactual importance is the primary faithful explanation score.
- Attention overlap is diagnostic only.
- `rev_` relation names are reverse message-passing edges added for HGT.
- Identity shortcut outputs test whether names, judges, courts, or lawyers can
  predict held-out labels from train-label priors.
- Mask sensitivity outputs test whether removing identity or hub-authority nodes
  changes frozen-model predictions.
