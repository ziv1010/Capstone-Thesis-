# Capstone Thesis Repository

This repository contains the full thesis pipeline for legal case processing,
dataset construction, graph neural network experiments, and final post-hoc
explanation analysis.

The codebase is organized as a staged workflow. A new user should generally
read and run the folders in this order:

1. `Fixed_GPU_OpenNyai/`
2. `DATA_SET_BUILDER_AND_EXPLORER/`
3. `section_GNN/`
4. `FINAL_EXPLANATION/`

The remaining folders are supporting inputs, visualizers, archives, paper
materials, and diagnostic analyses. They are documented after the main workflow
below.

## Repository Policy

The repository is source-first. Scripts, configs, README files, LaTeX sources,
and compact figures are intended to be versioned. Large generated datasets,
model checkpoints, graph caches, embedding arrays, runtime homes, and logs are
kept locally and ignored by Git where appropriate.

Use relative paths when adding new scripts or configs. The project has been
cleaned so active paths point to sibling repository folders with `../...`
instead of machine-specific absolute paths.

## Main Workflow

### 1. OpenNyAI Extraction and Labelling

Start with:

```text
Fixed_GPU_OpenNyai/
```

This folder converts raw judgment text into enriched case JSONs. It runs:

- OpenNyAI named-entity recognition
- OpenNyAI rhetorical-role extraction
- OpenNyAI summarization
- Mistral-based outcome labelling

Important outputs:

```text
Fixed_GPU_OpenNyai/final_outputs/
Fixed_GPU_OpenNyai/cross_validated_outputs/
```

Typical output families inside `final_outputs/` are:

- `<bucket>_extract/annotations/`
- `<bucket>_summary_opennyai/enriched_jsons/`
- `<bucket>_labelled_mistral/labelled_jsons/`

Read:

```text
Fixed_GPU_OpenNyai/README.md
Fixed_GPU_OpenNyai/run_scripts/README.md
Fixed_GPU_OpenNyai/final_outputs/README.md
```

### 2. Dataset Building and Timeline Merging

Then use:

```text
DATA_SET_BUILDER_AND_EXPLORER/
```

The active part of this folder is:

```text
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/
```

This stage merges labelled OpenNyAI outputs into the case format used by the
graph experiments. It also builds combined cross-bucket datasets and resolved
entity variants.

Important generated outputs:

```text
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3/
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/
```

The resolved output is the preferred final dataset source for visualisation and
some downstream analysis:

```text
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved/
```

Read:

```text
DATA_SET_BUILDER_AND_EXPLORER/README.md
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/README.md
```

### 3. GNN Training, Ablations, and Evaluation

Next use:

```text
section_GNN/
```

This is the main modelling folder. It preprocesses merged case JSONs, builds
heterogeneous graph caches, trains graph neural networks, and runs the
ablation/cross-domain experiments used in the thesis.

The standard modelling flow is:

1. Preprocess cases.
2. Build graph caches and embeddings.
3. Train/evaluate K-fold GNN models.
4. Run ablations and InLegalBERT/BGE comparison matrices.

Important areas:

- `section_GNN/src/`: reusable graph, preprocessing, model, training, and utility code
- `section_GNN/runs/`: baseline timed-bucket runs
- `section_GNN/runs_v2/`: later run variants
- `section_GNN/runs_inlegalbert/`: InLegalBERT experiment matrix
- `section_GNN/ablations/`: controlled ablation experiments
- `section_GNN/final_graph/`: final reasoning-focused graph builders
- `section_GNN/embedding_analysis/`: post-hoc embedding and probing tools

Generated data and model outputs are written under:

```text
section_GNN/data/
section_GNN/outputs/
section_GNN/run_logs/
```

Read:

```text
section_GNN/README.md
section_GNN/src/README.md
section_GNN/runs/README.md
section_GNN/ablations/README.md
section_GNN/final_graph/README.md
```

### 4. Final Explanation and Thesis Visualisation

Finally use:

```text
FINAL_EXPLANATION/
```

This folder does not train models. It reads trained model artifacts from
`section_GNN/` and runs final post-hoc analyses:

- typed counterfactual explanations
- faithfulness validation
- identity shortcut audits
- pattern/community analysis
- full-graph bridge and hub analysis
- final traceability reports
- local explanation visualizer
- paper/thesis figure generation

The main current output families are:

```text
FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_fold00/
FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_pattern_why/
FINAL_EXPLANATION/outputs/entity_resolved_section_sep_lr_decay_cross_bucket_full_graph/
```

Read:

```text
FINAL_EXPLANATION/README.md
FINAL_EXPLANATION/docs/README.md
FINAL_EXPLANATION/run_scripts/README.md
FINAL_EXPLANATION/outputs/README.md
```

## Supporting Folders

| Folder | Purpose |
| --- | --- |
| `INPUT_DATA/` | Local raw and extracted text inputs used before OpenNyAI processing. Large domain folders are intentionally ignored by Git. |
| `Thesis_FINAL_DATA/` | Local curated final data bundle for figures, graph info, timelines, embeddings, and experiment-result snapshots. |
| `GRAPH_VISUALISER/` | Dash visualiser and static plotting tools for the final resolved Timeline Maker graph data. |
| `Graph_Analyser_OLD/` | Legacy HGT explanation/graph-analysis prototype kept for reference and old figure generation. |
| `STAGE_VISUALISER/` | Small visualiser for hearing-stage or transition outputs. |
| `posthoc_case_reports/` | Case-level post-hoc report generation and timeline-merger diagnostics. |
| `model comparison/` | LLM/model comparison experiments against held-out GNN test cases. |
| `Latex_Documentation/` | LaTeX thesis/paper source material and final documentation bundles. |
| `configs/` | Legacy/root-level config files retained for reference. Most active configs now live inside their pipeline folders. |
| `DUMP_MISC/` | Archive for deprecated scripts, old experiments, exploratory utilities, and sample data moved out of active folders. |

## Folder-Level Documentation

Each active top-level folder has a `README.md`. Many important subfolders also
have local README files that explain their role, inputs, outputs, and whether
they are active, generated, or archived.

Generated data/output/cache folders are documented by their parent README or by
a short marker README when the folder is intended to be browsed directly. Do not
treat files under `DUMP_MISC/`, old output folders, cache folders, or generated
model-output folders as active entry points unless their local README says so.

## Environment Summary

Different stages use different environments because OpenNyAI, graph modelling,
and report generation have incompatible dependency constraints.

Common environment names:

- `fixed_gpu_opennyai_final`: OpenNyAI extraction, summaries, and GPU pipeline.
- `llm`: local/Hugging Face Mistral outcome labelling.
- `thesis_work`: `section_GNN` preprocessing, graph building, and training.
- `graph_vis`: `GRAPH_VISUALISER` Dash apps and static graph plots.
- `hgt_trace_reports`: final traceability report generation.

Prefer the folder README for exact setup commands.

## Reproducibility Notes

- Run commands from the folder specified by the local README, or use the provided
  shell wrappers.
- Keep configs relative to the repository.
- Do not commit large generated outputs, model checkpoints, caches, or raw case
  corpora.
- When adding a new experiment, place its README next to its scripts/configs and
  document input folders, output folders, environment, and expected run command.

## Quick Navigation

Start here for the main thesis pipeline:

```text
Fixed_GPU_OpenNyai/README.md
DATA_SET_BUILDER_AND_EXPLORER/README.md
DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/README.md
section_GNN/README.md
FINAL_EXPLANATION/README.md
```

Then use the supporting README files only when you need visualisation, paper
assets, post-hoc reports, model comparisons, or archived scripts.
