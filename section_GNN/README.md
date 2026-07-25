# 🕸️ section_GNN — Stage ④ · Graph Neural Network Modelling

> **Pipeline position:** ① INPUT_DATA ▸ ② Fixed_GPU_OpenNyai ▸ ③ DATA_SET_BUILDER_AND_EXPLORER ▸ **④ section_GNN** ▸ ⑤ FINAL_EXPLANATION

The modelling heart of the thesis. `section_GNN` converts the merged case JSONs from Stage ③
into **leakage-safe cleaned cases**, builds **heterogeneous PyTorch Geometric case-star
graphs**, trains and evaluates **HGT-style GNN models** with K-fold cross-validation, and
runs the full **ablation, encoder-comparison, cross-domain, and multi-hearing** experiment
matrix behind the thesis tables.

The folder is self-contained: every config uses paths relative to `section_GNN`
(sibling repo folders via `../...`), resolved at runtime by the config loader.

---

## ⚡ Quick Start

```bashV
cd section_GNN                      # all commands run from here

# One baseline bucket, end to end (preprocess → graph → 8-GPU K-fold):
bash runs/fin_fraud_timed_mistral/run_all.sh

# …or step by step:
bash runs/fin_fraud_timed_mistral/01_preprocess.sh
bash runs/fin_fraud_timed_mistral/02_build_graph.sh
bash runs/fin_fraud_timed_mistral/03_kfold_8gpu.sh
```

Most scripts default to the `thesis_work` micromamba environment; override with
`MAMBA_ENV=my_env bash ...`.

---

## 🔄 Core Workflow

| Step | Entry points | Main outputs |
|:----:|--------------|--------------|
| 1️⃣ **Preprocess** | `experiments/fixed_open_pipeline/preprocess_fixed_open.py` · bucket wrappers `runs/<bucket>/01_preprocess.sh` | `data/.../processed/cleaned_cases/` · `normalized_entities/` · `audits/` |
| 2️⃣ **Build graph** | `src/scripts/build_graph.py` · `final_graph/build_graph.py` · `final_graph/build_graph_section_sep.py` · `runs_v2/party_args_lr_decay/graph/build_graph_v2.py` | `data/.../graph_cache/*.pt` (+ metadata & node mappings) · `data/.../embeddings_cache/*.npz` |
| 3️⃣ **Train / evaluate** | `src/scripts/kfold_cv.py` · `runs_v2/party_args_lr_decay/scripts/kfold_cv_v2.py` · `src/scripts/train_gnn.py` · `src/scripts/evaluate_saved_model.py` | `outputs/.../models/<run_name>/kfold/fold_*/` · `kfold_summary.json` · predictions, metrics, plots |

---

## 🗂️ Folder Map

| Folder | Purpose |
|--------|---------|
| [`src/`](src/README.md) | Reusable preprocessing, graph, model, training, and utility packages. |
| [`runs/`](runs/README.md) | Baseline BGE-M3 timed-bucket experiment definitions (config + 3-step wrappers). |
| [`runs_v2/`](runs_v2/README.md) | Later run families: party-argument case text, LR-decay controls. |
| [`runs_inlegalbert/`](runs_inlegalbert/README.md) | InLegalBERT mirror of the main comparison matrix. |
| [`runs_inlegalbert_remaining/`](runs_inlegalbert_remaining/README.md) | InLegalBERT table-completion cells. |
| [`ablations/`](ablations/README.md) | Controlled graph/input/model ablations (10 variants). |
| [`experiments/`](experiments/README.md) | Standalone workflows: fixed-open preprocessing, dataset-size sweep. |
| [`final_graph/`](final_graph/README.md) | Reasoning-focused & section-separated graph builders. |
| [`embedding_analysis/`](embedding_analysis/README.md) | Post-hoc t-SNE, SHAP, probing, attention, node-importance tools. |
| [`cross_domain_test/`](cross_domain_test/README.md) | Held-out-domain evaluation (food safety). |
| [`multi_hearing_stage_test/`](multi_hearing_stage_test/README.md) | ⭐ Multi-hearing stage-transition experiment + **Multi-Hearing Stage Test Visualiser (port 8050)**. |
| [`run_scripts/`](run_scripts/README.md) | Top-level orchestration for the big experiment matrices. |
| [`configs/`](configs/README.md) | Legacy configs kept for reference. |
| [`data/`](data/README.md) | 📤 Generated cleaned cases, graph caches, embeddings, audits. |
| [`outputs/`](outputs/README.md) | 📤 Generated checkpoints, predictions, metrics, summary tables. |
| [`run_logs/`](run_logs/README.md) | 📤 Long-running experiment logs. |
| `summarize_bge_vs_inlegalbert.py` | Builds the BGE-M3 vs InLegalBERT comparison table from run outputs. |

---

## 🪣 Dataset Buckets

Most run families use the same six dataset names:

| Dataset | Description |
|---------|-------------|
| `family_matrimonial_timed_mistral` | Family & matrimonial bucket |
| `fin_fraud_timed_mistral` | Financial fraud bucket |
| `land_property_timed_mistral` | Land & property bucket |
| `motor_accidents_timed_mistral` | Motor accidents bucket |
| `sexual_offences_timed_mistral` | Sexual offences bucket |
| `cross_bucket_total_dataset` | Combined five-domain corpus for cross-bucket training/evaluation |

---

## 🧬 Graph Variants

The main graph family is a **case-star heterogeneous graph**: each `case` node connects to
text-section nodes (`preamble`, `facts`, `arguments`, party-specific argument sections),
party/court/judge/lawyer nodes, and legal-authority nodes (`statute`, `provision`,
`precedent`), with authorities **shared across cases**. Variants change what is included:

| Variant | Question it isolates |
|---------|----------------------|
| baseline | Full reasoning-focused graph |
| `text_only` | Text encodings alone (no entity/authority structure) |
| `no_names` | Contribution of identity/name-bearing nodes |
| `no_cross_case` | Value of cross-case node sharing |
| `hierarchical_enc` | Hierarchical text encoding |
| `section_sep_enc` (+ `_lr_decay`) | Separate per-section embeddings |
| `case_node_minimised` | Reduced case-node features |
| `entity_resolved_data` | Externally resolved (canonicalized) entities |
| `remove_central_authorities` | Dependence on high-degree hub authorities |
| `runs_v2/party_args_*` | Party-argument / preamble case-node text policies |

---

## ⚙️ Configuration Anatomy

Each YAML config has these sections: `project` (run name, seed) · `paths` (all directories,
relative to `section_GNN`) · `data` (glob, limits) · `preprocessing` (label/role mapping,
leakage masks) · `graph` (node types, section handling, cache name) · `features`
(text encoder, scalars) · `labels` · `model` (architecture, hidden size, layers, heads,
dropout) · `training` (epochs, optimizer, LR schedule, early stopping, folds).

Load configs through `src.utils.io.load_yaml` — it resolves relative paths against
`section_GNN` so configs stay portable. **Never add machine-specific absolute paths.**

---

## 🧮 Common Commands

```bash
# Full BGE-M3 experiment matrix (background + log):
nohup bash run_scripts/run_all_experiments.sh > run_logs/run_all_experiments.log 2>&1 &

# Main InLegalBERT matrix:
nohup bash run_scripts/run_inlegalbert_experiments.sh > run_logs/run_inlegalbert_experiments.log 2>&1 &

# Timed-bucket pipeline (preprocess + build + train + eval, 8 GPUs):
bash run_scripts/run_timed_mistral_buckets_8gpu.sh

# Direct invocations:
micromamba run -n thesis_work python experiments/fixed_open_pipeline/preprocess_fixed_open.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml

CUDA_VISIBLE_DEVICES=0,1 micromamba run -n thesis_work python final_graph/build_graph.py \
  --config runs/cross_bucket_total_dataset/config.yaml

micromamba run -n thesis_work python src/scripts/kfold_cv.py \
  --config runs/cross_bucket_total_dataset/config.yaml --run-name cross_bucket_kfold
```

---

## 📊 Reading Results

The authoritative per-run summary is:

```text
outputs/<family>/<bucket>/models/<run_name>/kfold/kfold_summary.json
```

Per-fold folders contain `metrics.json`, `predictions.csv`, `model.pt`,
`run_config_snapshot.yaml`, `training_history.png`, `split_metrics.png`, and
`confusion_matrix_*.png`. Aggregate tables live directly under `outputs/`
(e.g. `master_ablation_results.csv`, `inlegalbert_vs_bge_comparison.csv`).

---

## 🧑‍💻 Development Notes

- Keep reusable code in `src/`; experiment orchestration in `runs*/`, `ablations/`,
  `experiments/`, or `run_scripts/`.
- Generated artifacts belong under `data/`, `outputs/`, or `run_logs/`.
- Never hand-edit generated graph metadata; rerun the preprocessing/build instead.
- Prefer updating YAML configs and wrappers over hardcoding paths in Python modules.

---

⬆️ Back to the [repository root](../README.md) · Previous: [`DATA_SET_BUILDER_AND_EXPLORER/`](../DATA_SET_BUILDER_AND_EXPLORER/README.md) · Next: [`FINAL_EXPLANATION/`](../FINAL_EXPLANATION/README.md)
