# 🔍 FINAL_EXPLANATION — Stage ⑤ · Post-Hoc Explanation & Validation

> **Pipeline position:** ① INPUT_DATA ▸ ② Fixed_GPU_OpenNyai ▸ ③ DATA_SET_BUILDER_AND_EXPLORER ▸ ④ section_GNN ▸ **⑤ FINAL_EXPLANATION**

The final analysis stage. It **does not train models** — it loads the frozen HGT checkpoint
and graph cache from [`../section_GNN/`](../section_GNN/README.md) and produces the thesis'
explanation results:

- 🧩 **Typed counterfactual explanations** — which typed node/edge groups flip a prediction
- ✅ **Faithfulness validation** — sufficiency/comprehensiveness curves vs attention & random
- 🕵️ **Identity-shortcut audits** — can names/judges/courts/lawyers alone predict outcomes?
- 🧭 **Pattern & community analyses** — structural communities, embedding clusters, opposite-label neighbours
- 🌉 **Full-graph analyses** — Leiden communities, bridges, hub authorities
- 📜 **Traceability reports** — per-case HTML/JSON/DOT evidence reports
- 🖥️ **Final Explanation Visualizer** — interactive results browser on port **8899**
- 🖼️ **Paper figures** — the static figures used in the paper/thesis

---

## 🗂️ Folder Layout

| Path | Role |
|------|------|
| `*.py` | Explanation, validation, community, report, and visualizer modules (entry points below). |
| `presentation_graphs.py` | Shared data layer for the case-similarity graphs: nearest same/opposite-label cases, shared-evidence sets, counterfactual factor lookup, case display names. Used by the visualizer *and* the figure scripts. |
| `presentation_figure_style.py` | Matplotlib primitives (evidence cards, case nodes, connectors) shared by `generate_paper_figures.py` and `generate_presentation_figures.py`. |
| `build_counterfactual_factor_index.py` | One-off: compacts the 390 MB `case_counterfactual_groups.csv` into the per-case factor index the evidence badges read. |
| [`run_scripts/`](run_scripts/README.md) | Shell launchers: single-GPU, multi-GPU, audits, full workflow, visualizer. |
| [`docs/`](docs/README.md) | Reading guides — how to *interpret* the outputs (`VISUALIZER_GUIDE.md`). |
| [`figures/`](figures/README.md) | Final PNG/PDF paper figures generated from output tables. |
| [`outputs/`](outputs/README.md) | 📤 Generated experiment outputs (large, mostly Git-ignored). |
| [`logs/`](logs/README.md) | 📤 Loose runtime logs. |
| [`paper/`](paper/README.md) | LaTeX experiment-overview note + build artifacts. |
| [`visualizer_static/`](visualizer_static/README.md) | Frontend assets served by `visualizer.py`. |
| `traceability_reports_env.yml` | Micromamba spec for the `hgt_trace_reports` environment. |

---

## 🚀 Main Pipeline

The full current entity-resolved thesis explanation workflow (run from the repository root):

```bash
bash FINAL_EXPLANATION/run_scripts/run_entity_resolved_section_sep_lr_decay_cross_bucket_all.sh
```

This orchestrates, in order:

1. Typed counterfactual HGT explanations (`explain_hgt.py`)
2. Faithfulness + prediction-bucket validation (`validate_explanations.py`)
3. Identity shortcut audit (`identity_shortcut_audit.py`)
4. Pattern-level community & structural analyses (`structural_why_analysis.py`)
5. HGT case-embedding extraction (`extract_hgt_case_embeddings.py`)
6. Nearest opposite-label case analysis (`counterfactual_neighborhoods.py`)
7. Embedding cluster characterization (`embedding_cluster_characterization.py`)
8. Full-graph community/bridge/hub analysis (`full_graph_community_detection.py`, `full_graph_community_profiling.py`, `bridge_hub_authority_analysis.py`, `community_hierarchy_analysis.py`)
9. Post-hoc identity & hub-removal masking audit (`mask_sensitivity_audit.py`)

Default output prefix → three main output directories:

```text
outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00/        # explanations + validation
outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why/   # pattern/community/embedding
outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph/    # full-graph communities/bridges/hubs
```

### 🧪 Smoke Tests

```bash
# Small explanation run:
bash FINAL_EXPLANATION/run_scripts/run_default.sh \
  --case-limit 25 --output-dir FINAL_EXPLANATION/outputs/smoke_25

# Small multi-GPU run:
bash FINAL_EXPLANATION/run_scripts/run_multi_gpu.sh \
  --gpus 0,1 --output-dir FINAL_EXPLANATION/outputs/smoke_multigpu \
  -- --case-limit 2 --progress-every 1

# Small validation run:
bash FINAL_EXPLANATION/run_scripts/run_validation_multi_gpu.sh \
  --gpus 0,1 \
  --explanation-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --output-dir FINAL_EXPLANATION/outputs/validation_smoke \
  -- --case-limit 2 --k-values 0,1,2 --random-trials 1 --progress-every 1
```

---

## 🖥️ Visualizer — Port 8899

The ⭐ **Final Explanation Visualizer** is one of the repository's two main visualisers
(the other is the [Multi-Hearing Stage Test Visualiser](../section_GNN/multi_hearing_stage_test/visualiser/README.md), port 8050).

```bash
bash FINAL_EXPLANATION/run_scripts/run_visualizer.sh \
  --output-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --pattern-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why \
  --full-graph-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph \
  --host 127.0.0.1 --port 8899
```

Open **http://127.0.0.1:8899**. The server (`visualizer.py`, env `thesis_work`) merges the
CSVs server-side and presents: summary metrics · faithfulness curves · evidence support ·
identity-shortcut audits · communities · similar/opposite-label case comparisons · embedding
clusters · case-level traceability drilldowns · raw table browsing.

**Experiment 6** builds its three graphs live from `hgt_case_embeddings.npz` — a case
similarity network (3 closest same-label + 3 closest opposite-label cases, edges labelled with
the evidence they share) followed by the two contrast diagrams. Evidence boxes carry their
counterfactual rank, and every case node shows its true label *and* the model's prediction.

⚠️ **Read the labels carefully.** Pairing defaults to the *true* label, and the nearest
opposite-true-label case gets the **same** prediction from the model ~93% of the time. Such a
pair shows that the model could not separate the two cases — it does **not** show what made
their outcomes differ. The **Compare this case against…** switch at the top of the Case Selector
flips the pairing to *the closest case the model decided differently* (`--match pred` in the
figure script) — the only setting where "what drove the difference" is a real question.

Run the one-off index builder first so the counterfactual badges appear:

```bash
micromamba run -n thesis_work python FINAL_EXPLANATION/build_counterfactual_factor_index.py
```

The launcher defaults are baked in, so a bare
`bash FINAL_EXPLANATION/run_scripts/run_visualizer.sh` works once the main outputs exist.
📖 Interpretation guide: [`docs/VISUALIZER_GUIDE.md`](docs/VISUALIZER_GUIDE.md).

---

## 📤 Main Outputs (per family)

**Explanation** — `case_counterfactual_groups.csv` (all typed counterfactual group masks),
`case_top_explanations.csv` (top-k groups per case), `connected_case_label_distribution.csv`,
`typed_path_importance.csv`, `relation_type_importance.csv`, `evidence_type_importance.csv`,
`leakage_sensitivity_summary.csv`, `attention_counterfactual_overlap.csv`, `manifest.json`,
`run_summary.json`.

**Validation** — `faithfulness_curves.csv`, `faithfulness_auc_by_case.csv`,
`faithfulness_auc_summary.csv`, `prediction_bucket_cases.csv`,
`prediction_bucket_summary.csv`, `prediction_bucket_evidence_types.csv`.

**Pattern** — `case_communities.csv`, `community_profiles.csv`,
`community_feature_profiles.csv`, `community_success_failure.csv`,
`evidence_label_skew.csv`, `hgt_case_embeddings.npz`, `counterfactual_neighborhoods.csv`,
`case_embedding_clusters.csv`, `embedding_cluster_profiles.csv`,
`structural_embedding_alignment.csv`, and more.

**Full graph** — `full_graph_node_communities_{long,wide}.csv`,
`full_graph_community_profiles_res_1.00.csv`, `full_graph_community_authorities_res_1.00.csv`,
`bridge_authority_pairs_res_1.00.csv`, `authority_role_classification_res_1.00.csv`,
`authority_role_summary_res_1.00.csv`, `resolution_sweep_summary.csv`.

---

## 📜 Traceability Reports

Create the dedicated report environment once, then generate:

```bash
micromamba create -y -f FINAL_EXPLANATION/traceability_reports_env.yml   # env: hgt_trace_reports

# One sample report:
micromamba run -n hgt_trace_reports python FINAL_EXPLANATION/generate_traceability_reports.py \
  --explanation-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --output-dir FINAL_EXPLANATION/outputs/traceability_reports_sample \
  --case-limit 1 --overwrite

# Full batch:
micromamba run -n hgt_trace_reports python FINAL_EXPLANATION/generate_traceability_reports.py \
  --explanation-dir FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00 \
  --output-dir FINAL_EXPLANATION/outputs/traceability_reports_all \
  --all --overwrite
```

Each run writes machine JSON, case HTML, standalone graph HTML, DOT files, and a browsable
`index.html`.

---

## 🖼️ Paper & Presentation Figures

```bash
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_paper_figures.py

# Slide figures for one case: similarity network + same-label and opposite-label contrasts
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_presentation_figures.py \
  --case-index 51419 --formats png pdf svg

# Rank cases by how legible their figures come out
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_presentation_figures.py --suggest 20
```

Writes PNG + PDF (+ SVG) into [`figures/`](figures/README.md). Both scripts and the visualizer
share `presentation_graphs.py` (data) and `presentation_figure_style.py` (drawing), so the
figures and the on-screen panels always agree.

---

## 🗺️ Path Reproducibility

Scripts resolve paths from their own location — `APP_ROOT` (`FINAL_EXPLANATION/`),
`REPO_ROOT` (repository), `SECTION_GNN` (`../section_GNN`) — so nothing depends on the
original machine path. Model, graph, config, prediction, and output paths can be overridden
via CLI flags or the environment variables documented in
[`run_scripts/README.md`](run_scripts/README.md).

## 🧠 Interpretation Notes

- **Counterfactual importance** is the primary *faithful* explanation score; attention
  overlap is diagnostic only.
- `rev_` relation names are reverse message-passing edges added for HGT.
- Identity-shortcut outputs test whether names/judges/courts/lawyers can predict held-out
  labels from train-label priors; mask-sensitivity outputs test whether removing identity or
  hub-authority nodes changes frozen-model predictions.

---

⬆️ Back to the [repository root](../README.md) · Previous: [`section_GNN/`](../section_GNN/README.md)
