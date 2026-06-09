# section_GNN

`section_GNN` contains the graph-neural-network pipeline used for leakage-aware
legal outcome prediction. It converts extracted case JSON files into cleaned
case records, builds heterogeneous PyTorch Geometric graphs, trains/evaluates
GNN models, and runs the ablation and cross-domain experiments used for thesis
tables.

The folder is intentionally self-contained: paths in configs are relative to
`section_GNN`, and paths to sibling repository folders use `../...`. The Python
config loader resolves those relative paths at runtime.

## Quick Start

Run commands from this folder:

```bash
cd section_GNN
```

For a baseline timed-bucket run:

```bash
bash runs/fin_fraud_timed_mistral/01_preprocess.sh
bash runs/fin_fraud_timed_mistral/02_build_graph.sh
bash runs/fin_fraud_timed_mistral/03_kfold_8gpu.sh
```

For the active text-only cross-bucket ablation:

```bash
bash ablations/text_only/cross_bucket_total_dataset/run.sh
```

Most scripts use the micromamba environment named `thesis_work` by default.
Override it with:

```bash
MAMBA_ENV=my_env bash runs/fin_fraud_timed_mistral/run_all.sh
```

## Core Workflow

The standard pipeline has three stages.

1. Preprocess raw JSON files.

   Entry points:

   - `experiments/fixed_open_pipeline/preprocess_fixed_open.py`
   - bucket wrappers such as `runs/<bucket>/01_preprocess.sh`

   Main outputs:

   - `data/.../processed/cleaned_cases/*.json`
   - `data/.../processed/normalized_entities/*.json`
   - `data/.../audits/*.json`

2. Build graph caches.

   Entry points:

   - `src/scripts/build_graph.py`
   - `final_graph/build_graph.py`
   - `final_graph/build_graph_section_sep.py`
   - `runs_v2/party_args_lr_decay/graph/build_graph_v2.py`

   Main outputs:

   - `data/.../graph_cache/*.pt`
   - `data/.../graph_cache/graph_metadata*.json`
   - `data/.../graph_cache/node_mappings*.json`
   - `data/.../embeddings_cache/*.npz`

3. Train and evaluate.

   Entry points:

   - `src/scripts/kfold_cv.py`
   - `runs_v2/party_args_lr_decay/scripts/kfold_cv_v2.py`
   - `src/scripts/train_gnn.py`
   - `src/scripts/evaluate_saved_model.py`

   Main outputs:

   - `outputs/.../models/<run_name>/kfold/fold_*/`
   - `outputs/.../models/<run_name>/kfold/kfold_summary.json`
   - predictions, metrics, confusion matrices, and training-history plots

## Path Convention

Configs store portable paths. Examples:

```yaml
paths:
  raw_json_dir: ../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/cross_bucket_total_dataset
  processed_dir: data/timed_bucket_runs/cross_bucket_total_dataset/processed
  graph_cache_dir: data/timed_bucket_runs/cross_bucket_total_dataset/graph_cache
  outputs_dir: outputs/timed_bucket_runs/cross_bucket_total_dataset
```

When a Python script calls `src.utils.io.load_yaml`, these values are resolved
relative to `section_GNN`. Shell scripts that inspect YAML directly change into
`section_GNN` first for the same reason.

Avoid adding machine-specific absolute paths to configs or scripts.

## Main Folders

| Folder | Purpose |
| --- | --- |
| `src/` | Reusable preprocessing, graph construction, model, training, and utility code. |
| `src/scripts/` | Python command-line entry points for build/train/eval/audit helpers. |
| `runs/` | Baseline BGE-M3 timed-bucket configs and shell wrappers. |
| `runs_v2/` | Later run families with party-argument case text and LR-decay controls. |
| `runs_inlegalbert/` | InLegalBERT versions of the main comparison matrix. |
| `runs_inlegalbert_remaining/` | InLegalBERT runs used to fill remaining thesis-table cells. |
| `ablations/` | Controlled graph/input/model ablations. |
| `experiments/` | Standalone experiments such as fixed-open preprocessing, size sweep, and encoder comparison. |
| `final_graph/` | Reasoning-focused and section-separated graph builders. |
| `embedding_analysis/` | Post-hoc embedding, SHAP, t-SNE, probing, and node-importance tools. |
| `cross_domain_test/` | Cross-domain evaluation workflows, currently including food safety. |
| `multi_hearing_stage_test/` | Multi-hearing/stage-transition experiment and visualiser. |
| `run_scripts/` | Top-level orchestration scripts for larger matrices. |
| `data/` | Generated cleaned cases, graph caches, embedding caches, and audits. |
| `outputs/` | Generated predictions, metrics, model checkpoints, and summary tables. |
| `run_logs/` | Long-running experiment logs. |
| `configs/` | Older legacy configs retained for reference. |
| `FINAL_DUMP/` | Archived dump material. Treat it as read-only and out of the main pipeline. |

## Dataset Buckets

Most run families use the same six dataset names:

- `family_matrimonial_timed_mistral`
- `fin_fraud_timed_mistral`
- `land_property_timed_mistral`
- `motor_accidents_timed_mistral`
- `sexual_offences_timed_mistral`
- `cross_bucket_total_dataset`

The first five are domain-specific timed buckets. `cross_bucket_total_dataset`
is the combined dataset used for broader training/evaluation.

## Graph Variants

The main graph family is a case-star heterogeneous graph. A case node connects
to text-section nodes, party/court/lawyer nodes, and legal authority nodes.
Graph variants change which nodes and edges are included:

- baseline: full reasoning-focused graph
- `text_only`: case and text-section nodes only
- `no_names`: removes identity/name-bearing nodes and features
- `no_cross_case`: removes cross-case sharing
- `hierarchical_enc`: changes text encoding/aggregation strategy
- `section_sep_enc`: separates section embeddings
- `case_node_minimised`: reduces case-node text/features
- `entity_resolved_data`: uses externally resolved entity data
- `remove_central_authorities`: filters overly central authority nodes
- `runs_v2/party_args_*`: uses party-argument/preamble case-node text variants

See `ablations/README.md`, `runs/README.md`, and `runs_v2/README.md` for more
detail.

## Configuration Fields

Important sections in each YAML config:

- `project`: run name and seed.
- `paths`: input, intermediate, cache, audit, and output directories.
- `data`: file glob, limits, and optional sample case IDs.
- `preprocessing`: label mapping, role mapping, dropped roles, and leakage masks.
- `graph`: graph build mode, included node types, section handling, cache name.
- `features`: text encoder backend, scalar features, embedding settings.
- `labels`: binary or multiclass label handling.
- `model`: architecture, hidden size, layer count, heads, dropout.
- `training`: epochs, optimizer, LR schedule, early stopping, folds/repeats.

## Common Commands

Run one baseline bucket:

```bash
bash runs/fin_fraud_timed_mistral/run_all.sh
```

Run all baseline/ablation families:

```bash
nohup bash run_scripts/run_all_experiments.sh > run_logs/run_all_experiments.log 2>&1 &
```

Run the main InLegalBERT matrix:

```bash
nohup bash run_scripts/run_inlegalbert_experiments.sh > run_logs/run_inlegalbert_experiments.log 2>&1 &
```

Run the timed-bucket pipeline:

```bash
bash run_scripts/run_timed_mistral_buckets_8gpu.sh
```

Run fixed-open preprocessing directly:

```bash
micromamba run -n thesis_work python experiments/fixed_open_pipeline/preprocess_fixed_open.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

Build a reasoning-focused graph directly:

```bash
CUDA_VISIBLE_DEVICES=0,1 micromamba run -n thesis_work python final_graph/build_graph.py \
  --config runs/cross_bucket_total_dataset/config.yaml
```

Run K-fold CV directly:

```bash
micromamba run -n thesis_work python src/scripts/kfold_cv.py \
  --config runs/cross_bucket_total_dataset/config.yaml \
  --run-name cross_bucket_kfold
```

## Reading Results

Primary summaries usually live at:

```text
outputs/<family>/<bucket>/models/<run_name>/kfold/kfold_summary.json
```

Per-fold artifacts usually include:

- `metrics.json`
- `predictions.csv`
- `model.pt`
- `run_config_snapshot.yaml`
- `training_history.png`
- `split_metrics.png`
- `confusion_matrix_*.png`

Aggregate result tables are stored under `outputs/`, including:

- `outputs/master_ablation_results.csv`
- `outputs/inlegalbert_vs_bge_comparison.csv`

## Development Notes

- Keep source code in `src/`; keep experiment-specific orchestration in
  `runs*/`, `ablations/`, `experiments/`, or `run_scripts/`.
- Generated artifacts belong under `data/`, `outputs/`, or `run_logs/`.
- Do not hand-edit generated graph metadata unless you are explicitly repairing
  an artifact.
- Prefer updating YAML configs and wrapper scripts over hardcoding paths inside
  Python modules.
- Keep `FINAL_DUMP/`, `dump/`, and `__pycache__/` out of operational docs and
  new experiment wiring.
