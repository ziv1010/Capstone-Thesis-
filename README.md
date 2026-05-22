# Legal Case Outcome Prediction — Thesis Pipeline

A modular, reproducible ML pipeline for predicting Indian Supreme/High Court case outcomes from extracted PDF case records.

---

## Repository Hygiene

This repository is kept source-first for GitHub. Scripts, configs, LaTeX sources, compact figures, and README files are tracked. Large generated corpora, local model weights, graph caches, embedding arrays, LLM/Hugging Face caches, and experiment output folders are ignored and should be regenerated locally or restored from external storage.

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Repository Structure](#2-repository-structure)  
3. [Dataset & Extraction](#3-dataset--extraction)  
4. [Configuration](#4-configuration)  
5. [Phase 3 — Exploratory Data Analysis](#5-phase-3--exploratory-data-analysis)  
6. [Phase 4 — Baselines](#6-phase-4--baselines)  
7. [Phase 5-D1 — Text / Embeddings Model](#7-phase-5-d1--text--embeddings-model)  
8. [Phase 5-D2 — Graph Neural Network](#8-phase-5-d2--graph-neural-network)  
9. [Phase 5-D3 — LLM + RAG with FAISS](#9-phase-5-d3--llm--rag-with-faiss)  
10. [Output Structure](#10-output-structure)  
11. [Label System](#11-label-system)  
12. [GNN: Current Implementation & Improvement Suggestions](#12-gnn-current-implementation--improvement-suggestions)  
13. [Current Results (Pilot Dataset)](#13-current-results-pilot-dataset)  
14. [Requirements](#14-requirements)

---

## 1. Project Overview

This repository implements a full ML research pipeline for predicting legal case outcomes from extracted Supreme/High Court case records. The pipeline proceeds in stages:

| Phase | Description |
|-------|-------------|
| **Phase 2** *(existing)* | PDF extraction → structured `cases.jsonl` |
| **Phase 3** | Exploratory Data Analysis (EDA) |
| **Phase 4** | Baseline classifiers (TF-IDF + Logistic Regression) |
| **Phase 5-D1** | Text Embeddings + Classifier |
| **Phase 5-D2** | Graph Neural Network (case×entity heterogeneous graph) |
| **Phase 5-D3** | LLM + FAISS Retrieval-Augmented Generation (RAG) |

All new ML code lives exclusively in `src_ml/` — the original extraction code in `src/` is untouched.

---

## 2. Repository Structure

```
Capstone-Thesis-/
├── src/                        # [EXISTING] PDF extraction pipeline (do not modify)
│   └── ...
│
├── src_ml/                     # [NEW] ML pipeline — all phases 3–5
│   ├── common/
│   │   ├── io.py               # JSONL streaming → DataFrame; split load/save
│   │   ├── text_utils.py       # char/word stats, field joining, null-safe ops
│   │   ├── labels.py           # Canonical label mapper (configurable)
│   │   ├── metrics.py          # accuracy, macro-f1, weighted-f1, AUC, conf-matrix
│   │   ├── config_utils.py     # YAML loading + CLI override merging
│   │   ├── logging_utils.py    # Structured logging → outputs/logs/
│   │   ├── seed.py             # Global seed for reproducibility
│   │   ├── serialization.py    # joblib / json / torch / yaml save/load
│   │   └── sklearn_compat.py   # sklearn version-agnostic helpers
│   │
│   ├── eda/
│   │   ├── eda_runner.py       # Core EDA logic & aggregations
│   │   ├── eda_plots.py        # Matplotlib plot functions
│   │   └── eda_report.py       # Markdown report builder
│   │
│   ├── baselines/
│   │   ├── tfidf_logreg.py     # TF-IDF + LogReg baseline
│   │   ├── structured_classifier.py  # Structured features (statutes, court, year...)
│   │   └── baseline_runner.py  # Orchestrates both baselines
│   │
│   ├── models/
│   │   ├── text/
│   │   │   ├── embedder.py         # SentenceTransformer & HF mean-pooling backends
│   │   │   ├── classifier.py       # LogReg / MLP head on embeddings
│   │   │   └── train_text_model.py # Full text pipeline
│   │   ├── gnn/
│   │   │   ├── graph_builder.py    # Heterogeneous graph: case↔statute↔provision↔precedent
│   │   │   ├── gnn_model.py        # GCN via PyTorch Geometric (+ fallback LR)
│   │   │   └── train_gnn.py        # Full GNN pipeline
│   │   └── llm_rag/
│   │       ├── faiss_index.py      # FAISS index build/save/load/search
│   │       ├── prompt_builder.py   # RAG prompt construction
│   │       ├── rag_predictor.py    # Per-query: embed → retrieve → prompt → parse
│   │       └── eval_rag.py         # Full RAG pipeline orchestration
│   │
│   └── runners/
│       ├── run_eda.py          # CLI: python -m src_ml.runners.run_eda
│       ├── run_baselines.py    # CLI: python -m src_ml.runners.run_baselines
│       ├── run_text.py         # CLI: python -m src_ml.runners.run_text
│       ├── run_gnn.py          # CLI: python -m src_ml.runners.run_gnn
│       └── run_rag.py          # CLI: python -m src_ml.runners.run_rag
│
├── configs/
│   ├── config.yaml             # [EXISTING] extraction config
│   ├── config.fast.yaml        # [EXISTING] fast extraction config
│   └── ml.yaml                 # [NEW] unified ML config for all phases
│
├── outputs/
│   ├── cases.jsonl             # Source dataset (output of Phase 2)
│   ├── eda/
│   │   ├── tables/             # CSV summary tables
│   │   ├── figures/            # PNG plots
│   │   └── eda_report.md       # Auto-generated markdown report
│   ├── splits/
│   │   └── split_ids.json      # Deterministic train/val/test case IDs
│   ├── models/
│   │   ├── baselines/          # Saved baseline models (.joblib)
│   │   ├── text/               # Saved embeddings + classifier
│   │   └── gnn/                # Saved GNN model state or fallback
│   ├── results/
│   │   ├── baselines/          # metrics.json + preds.csv per baseline
│   │   ├── text/               # metrics.json + preds.csv for text model
│   │   └── gnn/                # metrics.json + preds.csv for GNN
│   └── rag/
│       ├── faiss/              # FAISS index + metadata
│       └── results/            # preds.csv + metrics.json + retrieval_audit.jsonl
│
├── requirements.txt
└── README.md
```

---

## 3. Dataset & Extraction

The pipeline reads from:

```
outputs/cases.jsonl
```

Each line is a JSON record with the following key fields (flattened by `src_ml/common/io.py`):

| Flat Column | Original Path | Type | Description |
|---|---|---|---|
| `case_id` | `case_id` | str | Unique case identifier |
| `court` | `court` | str | Court name |
| `year` | parsed from `date` | int | Year of case |
| `statutes` | `statutes` | list[str] | Cited statutes |
| `provisions` | `provisions` | list[str] | Cited provisions |
| `precedents` | `precedents` | list[str] | Cited precedent cases |
| `facts_text` | `texts.facts_text` | str | Facts section |
| `arguments_petitioner` | `texts.arguments_petitioner` | str | Petitioner arguments |
| `arguments_respondent` | `texts.arguments_respondent` | str | Respondent arguments |
| `reasoning_text` | `texts.reasoning_text` | str | Court reasoning |
| `decision_text` | `texts.decision_text` | str | Court decision text |
| `ml_input_text` | `ml.input_text` | str | Precomputed input for ML |
| `ml_leakage_flag` | `ml.leakage_flag` | bool | Contains decision info |
| `outcome_label` | `outcome.label` | str | Raw outcome label |
| `outcome_winner` | `outcome.winner` | str | Raw outcome winner |

**JSONL loading** supports streaming (chunk-based) to avoid memory issues on large datasets. Set `dataset.limit` in `configs/ml.yaml` for quick experiments.

---

## 4. Configuration

All ML phases are controlled by a single config file:

```bash
configs/ml.yaml
```

### Key Config Sections

```yaml
dataset:
  jsonl_path: outputs/cases.jsonl
  limit: null              # Set to e.g. 500 for quick experiments
  chunk_size: 4096

splits:
  seed: 42
  train: 0.70
  val: 0.15
  test: 0.15
  force_rebuild: false     # Set true to regenerate splits

labels:
  source_priority: [decision, outcome.winner, outcome.label]
  canonical_order: [for_appellant, against_appellant, dismissed, delayed, other]

text_model:
  embedder:
    backend: sentence_transformers
    model_name: sentence-transformers/all-MiniLM-L6-v2
    batch_size: 32
    max_length: 512

gnn:
  use_pyg: true            # Falls back to sklearn LR if PyG unavailable
  hidden_dim: 128
  epochs: 100

rag:
  top_k: 5
  llm:
    enabled: false         # Enable when local LLM is running
    endpoint_url: http://127.0.0.1:11434/api/generate
```

**CLI overrides** are supported by all runners (e.g. `--limit 100`, `--run_name my_test`).

---

## 5. Phase 3 — Exploratory Data Analysis

### Command

```bash
python -m src_ml.runners.run_eda --config configs/ml.yaml
```

### What it does

- Loads and flattens all cases from `cases.jsonl`
- Computes **class distribution** after canonical label mapping
- Computes **year distribution**, **court distribution**, and **case type distribution**
- Computes **missingness rates** for 8 key fields (judge_names, statutes, facts_text, etc.)
- Computes **text length statistics** (char/word mean, median, p95) for facts, arguments, and ml_input_text
- Computes **leakage summary**: % cases with `ml.leakage_flag == true` and non-empty `decision_text`
- Computes **top-K frequency tables** for statutes, provisions, precedents (K configured in `ml.yaml`)
- Generates **7 matplotlib plots** (no seaborn dependency)

### Outputs

| Location | Content |
|---|---|
| `outputs/eda/tables/class_distribution.csv` | Label counts |
| `outputs/eda/tables/cases_by_year.csv` | Year-wise case count |
| `outputs/eda/tables/cases_by_court.csv` | Court-wise case count |
| `outputs/eda/tables/missingness.csv` | % missing per field |
| `outputs/eda/tables/text_length_stats.csv` | Char/word stats |
| `outputs/eda/tables/top_statutes.csv` | Top cited statutes |
| `outputs/eda/tables/top_provisions.csv` | Top cited provisions |
| `outputs/eda/tables/top_precedents.csv` | Top cited precedents |
| `outputs/eda/figures/class_distribution.png` | Bar chart |
| `outputs/eda/figures/top_courts.png` | Bar chart |
| `outputs/eda/figures/cases_by_year.png` | Bar chart |
| `outputs/eda/figures/ml_input_text_wordcount_hist.png` | Histogram |
| `outputs/eda/figures/top_statutes.png` | Bar chart |
| `outputs/eda/figures/top_provisions.png` | Bar chart |
| `outputs/eda/figures/top_precedents.png` | Bar chart |
| `outputs/eda/eda_report.md` | Auto-generated summary report |
| `outputs/eda/eda_summary.json` | Machine-readable summary |

---

## 6. Phase 4 — Baselines

### Command

```bash
python -m src_ml.runners.run_baselines --config configs/ml.yaml
```

### Baseline A: TF-IDF + Logistic Regression (text-only)

- **Input**: Concatenated `facts_text + arguments_petitioner + arguments_respondent` (configurable `text_mode`)
- **Vectorizer**: `TfidfVectorizer` with configurable `max_features`, `ngram_range`, `sublinear_tf`
- **Classifier**: `LogisticRegression` with configurable `C`, `class_weight` (`balanced` by default), `solver`

### Baseline B: Structured + Logistic Regression

- **Features**: OneHot for `court`, `year`, `case_type`; MultiLabel indicator for `statutes`, `provisions`, `precedents`; count features for each
- **Classifier**: `LogisticRegression`

### Shared behaviour

- **Splits**: Both baselines reuse the exact same deterministic 70/15/15 split stored in `outputs/splits/split_ids.json`. This file is generated once and shared by ALL pipelines.
- **Stratification**: Stratified on label. Falls back to random if any class is too rare.
- **Config snapshot** saved alongside results for full reproducibility.

### Outputs

| Location | Content |
|---|---|
| `outputs/models/baselines/tfidf_logreg.joblib` | Vectorizer + classifier |
| `outputs/models/baselines/structured_logreg.joblib` | Feature transformer + classifier |
| `outputs/results/baselines/tfidf_logreg_metrics.json` | Accuracy, F1, AUC per split |
| `outputs/results/baselines/tfidf_logreg_preds.csv` | case_id, split, y_true, y_pred, y_pred_name |
| `outputs/results/baselines/structured_logreg_metrics.json` | Same format |
| `outputs/results/baselines/structured_logreg_preds.csv` | Same format |
| `outputs/results/baselines/run_config_snapshot.yaml` | Config used |
| `outputs/splits/split_ids.json` | Train/val/test case IDs (shared by all pipelines) |

---

## 7. Phase 5-D1 — Text / Embeddings Model

### Command

```bash
python -m src_ml.runners.run_text --config configs/ml.yaml
```

### How it works

1. Loads cases and derives canonical labels
2. Reuses split IDs from `outputs/splits/split_ids.json`
3. Selects text via `text_mode` config:
   - `facts_plus_args` *(default)*: facts + petitioner args + respondent args
   - `facts_only`: facts section only
   - `args_only`: petitioner + respondent arguments
   - `input_text_only`: precomputed `ml.input_text` field
4. Encodes text via one of two backends:
   - **`sentence_transformers`**: Uses `SentenceTransformer.encode()` directly (default: `all-MiniLM-L6-v2`)
   - **`hf_encoder`**: HuggingFace `AutoModel` with attention-mask-weighted mean pooling
5. **Caches embeddings** to `.npz` files (keyed by namespace + SHA hash of case IDs + config). Subsequent runs are instant.
6. Trains classifier:
   - **`logreg`**: `LogisticRegression` on embeddings (default)
   - **`mlp`**: Small MLP (configurable hidden dims and dropout)
7. Evaluates on train / val / test

### Switching to a legal BERT model

To use a domain-specific model, update `configs/ml.yaml`:

```yaml
text_model:
  run_name: legal_bert_run
  embedder:
    backend: hf_encoder
    model_name: law-ai/InLegalBERT    # or nlpaueb/legal-bert-base-uncased
    batch_size: 16
    max_length: 512
    device: cuda
```

### Outputs

| Location | Content |
|---|---|
| `outputs/models/text/embeddings_cache/` | Cached `.npz` embedding files |
| `outputs/models/text/<run_name>/classifier.joblib` | Trained classifier |
| `outputs/models/text/<run_name>/embeddings.npy` | All embeddings (numpy) |
| `outputs/models/text/<run_name>/run_config_snapshot.yaml` | Config used |
| `outputs/models/text/<run_name>/summary.json` | Run summary |
| `outputs/results/text/<run_name>_metrics.json` | Per-split metrics |
| `outputs/results/text/<run_name>_preds.csv` | Predictions CSV |

---

## 8. Phase 5-D2 — Graph Neural Network

### Command

```bash
python -m src_ml.runners.run_gnn --config configs/ml.yaml
```

### Graph Construction

The pipeline builds a **heterogeneous bipartite graph** from the dataset:

```
Node types:
  • case nodes      → one per case, feature = text embedding (from D1 cache)
  • entity nodes    → statute / provision / precedent (one node per unique entity)

Edge types:
  • case ↔ statute    (bidirectional)
  • case ↔ provision  (bidirectional)
  • case ↔ precedent  (bidirectional)
  • case ↔ case       (if a precedent string matches another case's ID or title)

Node features:
  • case nodes:   d-dim sentence embedding (reused from D1 cache)
  • entity nodes: random Gaussian init ~ N(0, 0.01) with same dimension d
```

### GNN Architecture (PyTorch Geometric)

When PyG is available (`use_pyg: true` in config):

- **Model**: 2-layer GCN (`GCNConv → ReLU → Dropout → GCNConv`)
- **Input**: node feature matrix (all nodes)
- **Output**: per-node logits (only case nodes used for loss/metrics)
- **Training**: Cross-entropy loss on train-split case nodes only
- **Val monitoring**: Accuracy on val-split case nodes; best-state checkpoint saved
- **Device**: CUDA if available, else CPU

### Fallback Mode (no PyG)

If `torch_geometric` is not importable, the pipeline **automatically falls back** to:

- Concatenating text embeddings with one-hot structured features (court, year, statutes, etc.) using `DictVectorizer`
- Training `LogisticRegression` (sklearn) on the concatenated sparse matrix

This fallback keeps the same output format so downstream evaluation is unaffected.

> **Current status**: Running in **fallback mode** (PyG not available in the current environment). See Section 12 for how to enable the full GNN.

### Outputs

| Location | Content |
|---|---|
| `outputs/models/gnn/<run_name>/model.pt` | GNN model state (or `{type: fallback_logreg}` marker) |
| `outputs/models/gnn/<run_name>/fallback_model.joblib` | Fallback classifier (when no PyG) |
| `outputs/models/gnn/<run_name>/run_config_snapshot.yaml` | Config snapshot |
| `outputs/models/gnn/<run_name>/summary.json` | Run summary |
| `outputs/results/gnn/<run_name>_metrics.json` | Per-split metrics |
| `outputs/results/gnn/<run_name>_preds.csv` | Predictions CSV |

---

## 9. Phase 5-D3 — LLM + RAG with FAISS

### Command

```bash
python -m src_ml.runners.run_rag --config configs/ml.yaml
```

### How RAG Works

```
INDEXING PHASE (train split only):
  1. Embed all TRAIN cases using the same SentenceTransformer from D1
  2. L2-normalize embeddings (enables cosine similarity via inner product)
  3. Build FAISS IndexFlatIP
  4. Save: index.faiss + train_case_ids.json + train_metadata.json

PREDICTION PHASE (per TEST case):
  1. Embed the query case text
  2. L2-normalize the query vector
  3. Search FAISS top-K (default K=5) — returns nearest TRAIN neighbours
  4. Build a prompt:
       - Query case: court, year, truncated facts/args snippet
       - Each retrieved case: case_id, court, year, snippet, known outcome
  5. Send to local LLM (Ollama / any OpenAI-compatible endpoint)
  6. Parse strict JSON response: {pred_label, pred_winner, confidence, rationale, cited_case_ids}
  7. Map prediction to canonical label set
  8. Fallback: if JSON malformed, retry once; if still fails, use majority-vote from retrieved cases
```

### Enabling the LLM

Update `configs/ml.yaml`:

```yaml
rag:
  llm:
    enabled: true
    endpoint_url: http://127.0.0.1:11434/api/generate  # Ollama
    model_name: qwen2.5:7b-instruct                     # or llama3, mistral, etc.
    timeout_sec: 120
    temperature: 0.0
```

Run Ollama in background: `ollama serve` then `ollama pull qwen2.5:7b-instruct`.

> When `llm.enabled: false`, the RAG pipeline still builds the FAISS index and saves retrieval results, using majority-vote of retrieved neighbours as prediction.

### Outputs

| Location | Content |
|---|---|
| `outputs/rag/faiss/index.faiss` | FAISS binary index |
| `outputs/rag/faiss/train_metadata.json` | Per-train-case metadata store |
| `outputs/rag/faiss/train_case_ids.json` | Ordered case ID list |
| `outputs/results/rag/preds.csv` | Predictions CSV |
| `outputs/results/rag/metrics.json` | Accuracy, F1, AUC |
| `outputs/results/rag/retrieval_audit.jsonl` | Per-query: retrieved IDs + prompt + raw LLM response |

---

## 10. Output Structure

All outputs from every pipeline are cleanly separated:

```
outputs/
├── cases.jsonl                         ← Source (Phase 2)
├── splits/
│   └── split_ids.json                  ← Shared by ALL pipelines
├── eda/
│   ├── tables/*.csv
│   ├── figures/*.png
│   ├── eda_report.md
│   └── eda_summary.json
├── models/
│   ├── baselines/
│   │   ├── tfidf_logreg.joblib
│   │   └── structured_logreg.joblib
│   ├── text/
│   │   ├── embeddings_cache/           ← Shared embedding cache (D1 + D2 + D3)
│   │   └── <run_name>/
│   │       ├── classifier.joblib
│   │       ├── embeddings.npy
│   │       └── run_config_snapshot.yaml
│   └── gnn/
│       └── <run_name>/
│           ├── model.pt
│           └── run_config_snapshot.yaml
├── results/
│   ├── baselines/
│   │   ├── tfidf_logreg_metrics.json
│   │   ├── tfidf_logreg_preds.csv
│   │   ├── structured_logreg_metrics.json
│   │   └── structured_logreg_preds.csv
│   ├── text/
│   │   ├── <run_name>_metrics.json
│   │   └── <run_name>_preds.csv
│   └── gnn/
│       ├── <run_name>_metrics.json
│       └── <run_name>_preds.csv
└── rag/
    ├── faiss/
    └── results/
```

**Every output `preds.csv`** has consistent columns:

```
case_id | split | y_true | y_pred | y_pred_name | prob_or_confidence
```

**Every output `metrics.json`** includes (per split):

```
n_samples | accuracy | macro_f1 | weighted_f1 | roc_auc | confusion_matrix | classification_report
```

---

## 11. Label System

The label mapping is fully configurable in `configs/ml.yaml` under the `labels:` section.

### Canonical Labels (in priority order)

| ID | Label Name | Meaning |
|---|---|---|
| 0 | `for_appellant` | Decision in favour of petitioner/appellant |
| 1 | `against_appellant` | Decision against petitioner (respondent wins) |
| 2 | `dismissed` | Appeal/petition dismissed, quashed, or disposed |
| 3 | `delayed` | Adjourned, deferred, or postponed |
| 4 | `other` | Anything else mappable to this bucket |

### Mapping Logic (`src_ml/common/labels.py`)

1. Try sources in `source_priority` order: `decision` → `outcome.winner` → `outcome.label`
2. For each source value, apply:
   - **Exact match** (after normalisation): `petitioner` → `for_appellant`, etc.
   - **Contains match**: substring patterns like `"bail granted"` → `for_appellant`
   - **Canonical passthrough**: if already a canonical label name
3. If no match and `drop_unknown: true`, the case is **dropped** with a log warning
4. Unknown / unmapped cases are never silently included

---

## 12. GNN: Current Implementation & Improvement Suggestions

### Current Implementation Summary

The GNN pipeline (`src_ml/models/gnn/`) implements:

1. **`graph_builder.py`**: Builds a heterogeneous bipartite graph with case nodes (text embeddings) and entity nodes (statute, provision, precedent — random init). Bidirectional edges connect cases to their cited entities. An optional case↔case edge is added when a precedent string matches another known case ID or title.

2. **`gnn_model.py`**: A 2-layer **GCN** using `torch_geometric.nn.GCNConv`. Trains on train-split case nodes, tracks val accuracy, saves best checkpoint.

3. **`train_gnn.py`**: Orchestrates the full pipeline — loads data, builds texts, loads/computes embeddings (shared cache with D1), builds graph, runs GCN or falls back to `sklearn` LR.

---

### What's Working Well ✅

- **Embedding reuse**: Case node features are the same frozen embeddings computed by D1, ensuring the GNN and text model are directly comparable.
- **Entity nodes**: Statute and provision nodes allow the GNN to propagate shared-statute signal across cases — sensible inductive bias for legal data.
- **Best-state checkpointing**: Saves the best val-accuracy model, not the final epoch.
- **Graceful fallback**: Automatically falls back to sklearn LR if `torch_geometric` is unavailable (HPC-friendly).
- **Deterministic splits**: Fully consistent with all other pipelines via `split_ids.json`.

---

### Identified Issues & Improvement Suggestions 🔧

#### Issue 1: Graph convolves entity-node features into case nodes, but entity features are random noise

**Problem**: Entity nodes (statutes, provisions, precedents) are initialised with tiny Gaussian noise `~N(0, 0.01)`. After one GCN layer, case nodes average in these near-zero features — this adds noise rather than signal, especially on a small dataset.

**Fix options**:
- **Option A (quick)**: Make entity node initial features **learnable embeddings** rather than fixed random noise. Add them as `nn.Embedding` parameters in the GCN so they get trained:

  ```python
  self.entity_embedding = nn.Embedding(n_entity_nodes, dim)
  nn.init.xavier_uniform_(self.entity_embedding.weight)
  ```

- **Option B (better)**: Initialise entity nodes using a **TF-IDF or count embedding** of the statute/provision text strings themselves, so they carry semantic information before any message passing.

- **Option C (best for legal domain)**: Use **typed/heterogeneous graph convolution** (`HeteroConv` in PyG) to have separate weight matrices for case→statute aggregation vs. case→precedent aggregation, avoiding feature mixing across semantically different entity types.

---

#### Issue 2: Homogeneous GCN treats all edges as equivalent

**Problem**: The current `GCNConv` does not distinguish between `case→statute`, `case→provision`, `case→precedent`, and `case→case` edges. In a legal graph, these relations carry very different semantic meaning.

**Suggested fix**: Switch to **`SAGEConv` (GraphSAGE)** or **`HeteroConv`**:

```python
# GraphSAGE (still homogeneous but more robust aggregation)
from torch_geometric.nn import SAGEConv

self.conv1 = SAGEConv(in_dim, hidden_dim)
self.conv2 = SAGEConv(hidden_dim, out_dim)
```

For heterogeneous convolution:

```python
from torch_geometric.nn import HeteroConv, SAGEConv

self.conv = HeteroConv({
    ('case', 'cites_statute', 'statute'): SAGEConv((-1, -1), hidden_dim),
    ('statute', 'cited_by', 'case'):      SAGEConv((-1, -1), hidden_dim),
    ('case', 'cites_prec', 'precedent'):  SAGEConv((-1, -1), hidden_dim),
    ('case', 'references', 'case'):       SAGEConv((-1, -1), hidden_dim),
}, aggr='sum')
```

---

#### Issue 3: Small dataset → risk of over-smoothing with 2 GCN layers

**Problem**: With ~90 cases and many entity nodes, a 2-layer GCN can cause over-smoothing — case node representations collapse toward a global mean, losing discriminative power.

**Suggestions**:
- **Reduce to 1 GCN layer** for the pilot dataset. Add a skip-connection (residual) to preserve the original features:

  ```python
  h = self.conv1(x, edge_index)
  h = F.relu(h + x[:, :hidden_dim])  # residual if dims match
  ```

- **Add dropout on the input features** (input dropout), not just between layers.
- Consider **PairNorm** or **DiffGroupNorm** to combat over-smoothing explicitly.

---

#### Issue 4: Class imbalance not handled in GNN training

**Problem**: The current GCN uses unweighted `F.cross_entropy`. With `for_appellant` dominant (~58% of cases), minority classes (`against_appellant`, `dismissed`, `delayed`) get poor recall.

**Fix**: Pass class weights into the loss:

```python
from sklearn.utils.class_weight import compute_class_weight
import torch

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

loss = F.cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
```

---

#### Issue 5: No early stopping based on val loss (only val accuracy tracked)

**Problem**: The model saves best val accuracy, but accuracy can plateau while val loss diverges — a sign of overfitting. On small datasets, this distinction matters.

**Fix**: Track `val_loss` and use it for early stopping:

```python
val_loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
```

Also add configurable patience: stop if no improvement for N epochs.

---

#### Issue 6: No `BatchNorm` or `LayerNorm` in the GCN

**Suggestion**: Add `BatchNorm1d` after each graph convolution layer. This stabilises training, especially at small batch sizes (full-batch GCN):

```python
self.bn1 = nn.BatchNorm1d(hidden_dim)
...
h = F.relu(self.bn1(self.conv1(x, edge_index)))
```

---

#### Issue 7: Graph builder silently ignores `case_title` if missing

**Note (minor)**: In `graph_builder.py`, `title_to_case_node` is only populated if `rec.get("case_title")` is non-null. This is correct behaviour, but the case-to-case edge lookup (`norm_prec in title_to_case_node`) will just always miss if titles are absent — no real bug, just worth knowing.

---

#### Summary Table of GNN Improvements

| Priority | Issue | Recommended Fix |
|---|---|---|
| 🔴 High | Unweighted loss despite class imbalance | Add `class_weight` tensor to `F.cross_entropy` |
| 🔴 High | Entity features are random noise | Make entity features learnable `nn.Embedding` |
| 🟡 Medium | Homogeneous GCN conflates edge types | Switch to `SAGEConv` or `HeteroConv` |
| 🟡 Medium | No early stopping | Track val_loss with patience counter |
| 🟡 Medium | Over-smoothing on small graph | Use 1 GCN layer + skip connection |
| 🟢 Low | No batch normalisation | Add `BatchNorm1d` after each conv layer |
| 🟢 Low | No input-feature dropout | Add dropout on x before conv1 |

---

## 13. Current Results (Pilot Dataset)

> ⚠️ These results are from a **pilot dataset of ~90 labeled cases** (after filtering). Numbers will change significantly on the full dataset. Treat these as **sanity-check numbers** only.

### Text Model (D1) — `text_d1`

| Split | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Train | 80.9% | 0.732 | 0.817 |
| Val   | 53.8% | 0.397 | 0.575 |
| **Test** | **50.0%** | **0.279** | **0.545** |

Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, text mode: `facts_plus_args`.  
The strong train→test gap is entirely expected given 90 samples. Minority classes (`delayed`, `other`) get 0 recall on test.

### GNN (D2) — `gnn_d2` (Fallback mode, no PyG)

| Split | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Train | **100%** | **1.000** | **1.000** |
| Val   | 84.6% | 0.633 | 0.777 |
| **Test** | **78.6%** | **0.552** | **0.722** |

> The GNN fallback achieves higher accuracy than the text model on this pilot, but with perfect train accuracy — a clear sign of **overfitting** on 63 training cases.

**Key observations**:
- Perfect train accuracy = memorisation, not generalisation
- `for_appellant` has 100% recall in both val/test (dominant class, ~58% of data)
- `against_appellant` gets 0 recall in both val/test
- `delayed`/`other` have no support in val/test
- **Next step**: run on the full extraction dataset to get reliable estimates

---

## 14. Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Core ML requirements (added in `requirements.txt`)

```
# Core ML
torch>=2.0.0
sentence-transformers>=2.2.0
transformers>=4.30.0
scikit-learn>=1.3.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0

# Graphs
torch-geometric>=2.3.0   # optional but recommended for full GNN
faiss-cpu>=1.7.4         # or faiss-gpu for GPU support

# Visualisation
matplotlib>=3.7.0

# Config
PyYAML>=6.0

# Saving models
joblib>=1.3.0
```

### HPC Notes

- All runners support `dataset.limit` in config for quick development runs
- Embeddings are cached to disk — compute once, reuse across all pipelines
- The GNN fallback (sklearn LR) uses no GPU and minimal memory
- For full GNN, `torch_geometric` requires matching CUDA/torch versions — see [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- All file I/O uses `pathlib` (no hardcoded paths beyond `configs/ml.yaml`)

---

## Quick Reference: All Pipeline Commands

```bash
# Phase 3: EDA
python -m src_ml.runners.run_eda --config configs/ml.yaml

# Phase 4: Baselines
python -m src_ml.runners.run_baselines --config configs/ml.yaml

# Phase 5-D1: Text / Embeddings model
python -m src_ml.runners.run_text --config configs/ml.yaml

# Phase 5-D2: GNN
python -m src_ml.runners.run_gnn --config configs/ml.yaml

# Phase 5-D3: RAG with FAISS
python -m src_ml.runners.run_rag --config configs/ml.yaml

# Quick experiment with limited data (override limit in config)
python -m src_ml.runners.run_text --config configs/ml.yaml --limit 200 --run_name quick_test
```

All pipelines run from the repo root (`Capstone-Thesis-/`) and write outputs relative to `configs/ml.yaml`'s `outputs.root` setting.
