# CODEBASE_DOCUMENTATION
Comprehensive Technical Documentation for the Legal Outcome Prediction GNN Pipeline

---

## 1. Executive Overview
This codebase is designed to predict the outcome of legal cases (e.g., whether an appellant/petitioner wins or loses) using a **Heterogeneous Graph Neural Network (HGT / HeteroConv)**. 

Instead of treating a legal case as a flat sequence of text, the pipeline models it as a highly structured "Star Graph", where the focal point is the `case` node, surrounded by rich semantic entities (text sections, parties, lawyers, judges, statutes, precedents). 

**The Workflow (Raw to Trained Model):**
1. **Raw JSON Ingestion:** Expects deeply annotated NLP output from OpenNyai (containing rhetoric labels, entity extractions, and summary text).
2. **Leakage-Safe Extraction:** Strips out post-judgment leakage (e.g., words like "allowed", "dismissed") and explicit decision sections.
3. **Graph Construction:** Creates a PyTorch Geometric (PyG) Heterogeneous Graph.
   - Text chunks (preamble, facts, arguments) are embedded via `sentence-transformers`.
   - Structural and citation edges relate lawyers to arguments, arguments to provisions, and cases to entities.
4. **Training:** Trains a Graph Transformer (HGT) or HeteroConv. Information propagates across the network up to the `case` node. 
5. **Prediction:** An MLP head on top of the `case` node outputs the final prediction (Win / Lose / Procedural).

---

## 2. Repository Map

The codebase is highly modular, split by purpose.

```text
section_GNN/
├── configs/
│   ├── gnn_case_star.yaml                 # Base configuration
│   ├── gnn_case_star_food_law_final.yaml  # Experiment-specific config (Food Law dataset)
│   └── gnn_case_star_sanity.yaml          # Quick structural testing
├── scripts/                               # Entrypoints for execution
│   ├── preprocess_cases.py                # Pipeline Step 1 (JSON -> Cleaned Case schemas)
│   ├── build_graph.py                     # Pipeline Step 2 (Cleaned Cases -> PyG HeteroData Cache)
│   ├── train_gnn.py                       # Pipeline Step 3 (Model Training and Evaluation)
│   ├── run_ablation.py                    # Runs multiple configs automatically
│   ├── visualize_graph.py                 # Graph visualization utility
│   └── generate_final_visualisations.py   # Plotting and stats extraction
└── src/                                   # Core application modules
    ├── graph/
    │   ├── schema.py                      # Defines Data Classes (CleanedCase, GraphNode), relation lists
    │   ├── pyg_builder.py                 # Compiles PyTorch Geometric HeteroData, injects embeddings
    │   ├── global_graph_builder.py        # Merges individual case graphs into one transductive super-graph
    │   └── case_star_builder.py           # Logic to build a local graph for a single case
    ├── models/
    │   ├── hetero_gnn.py                  # PyTorch definition of the Heterogeneous GNN (HGTConv / SAGEConv)
    │   └── mlp_head.py                    # Simple Readout block
    ├── preprocessing/
    │   ├── extract.py                     # Heavy-lifting for parsing OpenNyai outputs
    │   ├── leakage.py                     # Rigorous RegEx and Rule-based leakage masking
    │   ├── loader.py                      # JSON Loading Utility
    │   └── normalize.py                   # Canonical entity merging (e.g. mapping "Smt. X" to "X")
    ├── training/
    │   ├── dataset.py                     # Stratified / Group-based splitting (Train/Val/Test)
    │   ├── evaluate.py                    # Wraps evaluation functions
    │   ├── metrics.py                     # Scikit-learn wrapping, computes Acc, F1, and plots plots
    │   └── train.py                       # The actual core training loop (loss fn, early stopping logic)
    └── utils/
        ├── pipeline.py                    # Orchestrates the transition from extracted JSON -> PyG Graph
        └── text_encoder.py                # Interfaces with sentence-transformers for text embeddings
```

- **Entry Point for Graph Creation:** `scripts/build_graph.py`
- **Entry Point for Training:** `scripts/train_gnn.py`
- **Core Graph Logic:** `src/graph/schema.py` and `case_star_builder.py`

---

## 3. End-to-End Pipeline Walkthrough

1. **Preprocessing (`scripts/preprocess_cases.py`):**
   - **Input:** OpenNyai Augmented JSON files.
   - **Process:** Reads `.json`. Loads it to `extract.py`. Identifies summary sections. Masked leakage is replaced with `[LEAKAGE_MASK]`. Extracts `Entities` using Spacy annotations matching allowed types (e.g., `PETITIONER`, `JUDGE`). Merges names via `normalize.py`.
   - **Output:** Saves heavily reduced, flattened `CleanedCase` schemas into `data/.../cleaned_cases/`.

2. **Graph Compilation (`scripts/build_graph.py`):**
   - **Input:** `CleanedCase` objects.
   - **Process:** Calls `pipeline.py:build_graph_bundle`. For each case, `case_star_builder.py` attaches text nodes and entity nodes. `global_graph_builder.py` unifies entities matching globally across cases (like common judges or courts) if `share_party_nodes` / `share_across_cases` is enabled. `pyg_builder.py` fetches `sentence-transformers` text embeddings and computes structural scalars (degree, frequency).
   - **Output:** Saves a `.pt` PyTorch bundle containing the compiled PyTorch Geometric `HeteroData`.

3. **Training Iteration (`scripts/train_gnn.py`):**
   - **Input:** The `cache.pt` HeteroData containing `train_mask`, `val_mask`, `test_mask`.
   - **Process:** Loads config parameters. Initializes `HeteroLegalOutcomeGNN`. 
   - **Forward Pass:** Node embeddings run through multiple layers of `HGTConv`. The target node (`case`) aggregates inputs. `MLPHead` computes unnormalized logits.
   - **Evaluation:** Uses `metrics.py` to compare predictions vs true labels (Loss: CrossEntropy).
   - **Output:** Models, CSV predictions, and generated confusion matrix plots.

---

## 4. Data Schema and Input Format

The pipeline expects output JSON files from the highly structured **OpenNyai augmentation pipeline**.

### Required Structure
- `case_outcome_label`: The true status (e.g., `appellant_won`). Note: This is dropped out of text features to prevent leakage.
- `raw_result.summary`: Must contain fields like `PREAMBLE`, `facts`, and `arguments`.
- `raw_result.data.text`: The complete string of the court case.

### Annotation Fields (Extraction Targets)
The JSON contains an `annotations` list with a schema like so:
```json
{
  "summary_section": "arguments",
  "text": "The learned counsel for petitioner submitted...",
  "labels": ["ARGUMENT"],
  "entities": [
    { "text": "Mr. Singh", "labels": ["LAWYER"], "start": 4, "end": 13 }
  ]
}
```
**Handling Missing Data:** If `summary.facts` is missing, `extract.py` looks at all annotations where `summary_section == "facts"` and strings them together. If a label is unmapped, it throws an error unless `drop_procedural=True`.

---

## 5. Graph Construction Documentation

The schema of the graph is strictly defined in `src/graph/schema.py`. It is a true Heterogeneous network.

### Node Creation Logic
Nodes obtain features through `pyg_builder.py`.
- **Text Nodes** (`preamble`, `facts`, `arguments`): Receive an dense embedding from `Sentence Transformers` (like `all-MiniLM-L6-v2`), appended with scalar counts (e.g., text length, number of cited precedents).
- **Entity Nodes** (`judge`, `statute`, `lawyer`): Typically do not have text, but receive a synthetic embedding if necessary, and use heavily structural feature vectors (degree, global frequency, first seen section).
- **Case Nodes**: Receive an array of descriptive structural statistics (total lawyer count, case year, respondent count).

### Edge Construction
Edges are assembled in `case_star_builder.py`.
Edges connect structural domains to text. For example, rather than just linking a `petitioner_lawyer` to the `case`, the builder actively creates a `citation` edge connecting the `petitioner_lawyer` to the `petitioner_arguments` text node.
Reverse edges are injected later by PyG `ToUndirected()`.

---

## 6. Node-by-Node Meaning

- **`case`**: The central anchor for the legal dispute. Aggregates all structural and textual context. Final readout occurs here.
- **`preamble` / `facts`**: Raw textual sections. Provides historical background. Highly safe from leakage.
- **`arguments`**: Extract of legal rationale. Split dynamically into `petitioner_arguments` and `respondent_arguments` by looking at nearby lawyer annotations in the text.
- **`petitioner` / `respondent`**: The entities battling. In some configs, kept strictly local. In others, optionally shared.
- **`court` / `judge`**: Global structural variables. Highly relevant if certain judges have strong biases. Always shared.
- **`*_lawyer`**: Extracted via heuristic lookup (e.g., if "for the respondent" appears near a LAWYER entity, it becomes a `defence_lawyer`).
- **`statute` / `provision`**: Structural points of law. `provision` edges up into `statute` (hierarchical).
- **`precedent`**: Previous cases cited in the argument.

---

## 7. Relation-by-Relation Meaning

- **Structural Setup**: 
  - `(case, has_facts, facts)`
  - `(case, decided_by_bench, judge)` 
  *Meaning:* Definitional mapping anchoring nodes to their parent case.
  *Leakage Risk:* None.
- **Bridging / Citation Edges**: 
  - `(petitioner_lawyer, citation, petitioner_arguments)`
  - `(statute, used_in_arguments, arguments)`
  *Meaning:* Shortens the message-passing hop distance from an entity to its textual justification. Derived relation, computationally mapped from text overlap.
  *Leakage Risk:* Negligible.
- **Hierarchy Edges**:
  - `(provision, belongs_to_statute, statute)`
  *Meaning:* Groups laws together structurally.

---

## 8. Model Architecture

The framework is a PyTorch representation implemented in `src/models/hetero_gnn.py`.

### The Core
- **`HeteroLegalOutcomeGNN`**: The primary wrapper model.
- **Input Projections**: Every arbitrary feature size for every node type is projected down to `hidden_dim` (e.g. `128`) using a typed `nn.Linear` layer.
- **Core Layers**: Defaults to **HGT (Heterogeneous Graph Transformer)** via PyG's `HGTConv`. HGT is excellent here because it handles highly disjoint features dynamically using relative attention based on node and edge types. `HeteroConv` wrapped around `SAGEConv` is provided as an alternative fallback.
- **Message Passing**: Over `N` layers (Default: 3), nodes update their states by aggregating messages from neighbors. A LayerNorm + ReLU + Dropout activation closes each hop.
- **Classification / Readout**: Extracted from the `case` node's final updated embedding. Passed through `mlp_head.py` (a multi-layer perceptron) directly to the un-normalized logits for class sizes.

---

## 9. Training Logic

Run via `scripts/train_gnn.py`, the training script establishes reproducible benchmarking.
- **Splits**: Implemented in `src/training/dataset.py`. `mode=random` splits dynamically with stratification (`stratify=True`), ensuring class balance. There are also `year` or `court` based GroupSplits available to prevent leakage via duplicate courts or timespans.
- **Loss Computation**: Pure `CrossEntropy`. To handle huge class imbalances (e.g., Appellants winning 80% of the time), a `class_weight` scaler assigns weights based on the inverse inverse frequencies of classes in the `train` set.
- **Optimizer**: `AdamW`. Used over Adam for better regularization via weight coupling (`1e-5`).
- **Optimization Strategy**: Standard epochs + `early_stopping_patience`.
- **Reproducibility**: `set_global_seed()` is strictly enforced. It runs repeated runs (`seed_stride`) picking the model with the highest `val.macro_f1`.

---

## 10. Evaluation Logic

Triggered via `src/training/evaluate.py` / `metrics.py`.
- The evaluation focuses heavily on **Macro F1**. Why? Because raw accuracy is deceiving if a dataset represents an unbalanced legal landscape.
- **Metrics Tracked**: Accuracy, Macro F1, Micro F1, ROC-AUC, Class-specific Precision/Recall/Support.
- **Artifacts**: Generates deep visual evidence. Confusion Matrices and Split Bar Charts are output directly to `runs/models/.../`.

---

## 11. Config and Hyperparameters

Managed via hierarchical YAMLs. Example configurations from `gnn_case_star...yaml`:
- **`graph.build_mode`**: Toggles topological ablations (`text_only` vs `full_star_global`).
- **`preprocessing.leakage_phrases`**: An explicit array of hardcoded "leakage phrases" (e.g., `petition dismissed`).
- **`features.text_encoder`**: Defines the HuggingFace cache pipeline. Default: `sentence-transformers/all-MiniLM-L6-v2`.
- **`labels.binary_map`**: Maps strings like `appellant_won` to structural classes like `win`.
- **`training.lr`**: Learning Rate. Expected standard: `0.001`, `epochs`: 60. `dropout`: `0.25`.

---

## 12. Call Graph / Code Flow

1. You run `python scripts/preprocess_cases.py` -> calls `extract.py` logic -> loops `json` parsing.
2. You run `python scripts/build_graph.py` -> initializes `pipeline.py` -> triggers `case_star_builder.py` -> triggers `pyg_builder.py` to bake the PyTorch HeteroData bundle -> dumps to `cache.pt`.
3. You run `python scripts/train_gnn.py` -> loads `cache.pt` -> initializes model `HeteroLegalOutcomeGNN` -> invokes `src/training/train.py:train_model` loop -> computes outputs and `metrics.py` plots.

---

## 13. Important Classes and Functions

| Component | Path | Purpose | Centrality |
| :--- | :--- | :--- | :--- |
| `CleanedCase` | `schema.py` | Data structure storing post-extraction content before graphification. | Central |
| `HeteroLegalOutcomeGNN` | `hetero_gnn.py` | Implementation of graph propagation. Drives the forward pass. | Central |
| `extract_prejudgment_text` | `extract.py` | Filters `summary` keys to discard judgments and maps leakage. | Key Helper |
| `build_case_star_graph` | `case_star_builder.py` | Actually computes and maps Graph nodes/edges per Case. | Central |
| `train_model` | `train.py` | Core PyTorch backprop loop. | Central |

---

## 14. Design Choices and Rationale

**Confirmed from Code:**
1. **Dynamic Lawyer Siding:** Extracting lawyers into `petitioner_lawyer` vs `defence_lawyer` is dynamically guessed using a 300 character rolling context window (`_infer_lawyer_side`).
2. **Heavy Masking over Removal:** The codebase *masks* strings like `[LEAKAGE_MASK]` for `petition allowed` instead of deleting the sentence.
3. **Graph Fallbacks:** If a node doesn't exist, it is permitted to be synthesized (e.g. Provisions synthesizing missing Statutes).

**Inferred Structure:**
1. **Why HGT?** Legal documents are intrinsically highly heterogeneous. A judge's relationship to a case behaves mathematically differently than an argument's semantic relation. HGT resolves this natively by having distinct matrices for every combination.
2. **Why Bridging Edges?** Graph diameter issues (Over-smoothing limit). An argument node is far from a lawyer, but `citation` edges bridge them directly allowing gradients to backprop cleanly in < 2 hops.

---

## 15. Leakage / Safety / Methodology Audit

> [!WARNING]
> **This is the most dangerous aspect of predictive legal ML, and this codebase dedicates enormous logic to it.**

- **Use of Post-Decision Text:** Extensively forbidden. Fields like `decision_text`, JSON key `case_outcome_label`, and NLP label `RPC` (Ratio / Ruling) are explicitly filtered in `extract.py` and `leakage.py`.
- **Word-Level Leakage:** The framework maps explicitly phrases like "appeal dismissed".
  *Risk:* The regex only covers `DEFAULT_LEAKAGE_PHRASES`. A subtle phrasing like "We find no merit in the appeal" will pass through unfiltered resulting in 100% predictive accuracy (leakage).
  *Fix:* Regularly audit the `retained_texts` metrics output against validation failures.
- **Transductive Global Nodes:** `share_party_nodes` is set to `False` organically. If it were `True`, an entity appearing in the *Train* set and the *Test* set might allow information bleed. However, sharing `judge` and `court` is safe and recommended (helps predict external biases).

---

## 16. Practical Guide: Modifying the Pipeline

- **Add a new structural Node (`expert_witness`):**
  1. Add to `ENTITY_NODE_TYPES` in `src/graph/schema.py`.
  2. Map it in `ALLOWED_ENTITY_LABELS` in `src/preprocessing/extract.py`.
  3. Wire its structural connection (`has_expert`) in `src/graph/schema.py:RELATION_DEFINITIONS`.
- **Change the Model to GAT / Relational GCN:**
  Modify `hetero_gnn.py` nested within the `if architecture ==` block. Map relationships to `GATConv`.
- **Run a feature ablation:**
  Copy and paste an ablation key in `configs/...yaml` within the `ablation:` tree. `graph.include_node_types` explicitly lists what builds the model.

---

## 17. Reproduction Guide

**How to run start to finish:**
*(Assuming micromamba environment `gnn` is active)*

```bash
# 1. Clean the messy jsons
python scripts/preprocess_cases.py --config configs/gnn_case_star_food_law_final.yaml

# 2. Build the graph (will download sentence-transformers implicitly if empty cache)
python scripts/build_graph.py --config configs/gnn_case_star_food_law_final.yaml

# 3. Train
python scripts/train_gnn.py --config configs/gnn_case_star_food_law_final.yaml --run-name food_trial_1
```

---

## 18. Glossary

- **HGT**: *Heterogeneous Graph Transformer.* A transformer designed for multi-type node structures.
- **Case Node**: Central pseudo-node from which an MLP predicts final states.
- **Rhetorical Role**: NLP assignment of sentences (Precedent, Facts, Arguments). Used extensively to segment the graph.
- **RPC / RLC**: Ratio clause/Ruling clauses. These represent final judgments and represent extreme leakage hazards.

---

## 19. File-by-File Deep Notes Appendix

1. **`src/preprocessing/extract.py`**
   - **Note:** Holds the heuristic magic that converts a highly nested OpenNyai Spacy dump into something usable. **Very fragile**. The definition of `_infer_lawyer_side` determines the success of `petitioner_lawyer` assignments.
2. **`src/training/metrics.py`**
   - **Note:** Hardcoded for evaluating `Macro F1`. Provides free automatic artifact graphing `save_split_metric_bar_plot`, saving a researcher endless hours writing plotting code.
3. **`src/graph/global_graph_builder.py`**
   - **Note:** This determines transductive logic. It deduplicates node keys across multiple files. Meaning if `(statute, Contract Act)` appears in 5 cases, there is only ONE node for it bridging the cases, unless disabled via config.

---

## 20. Final Summary

The most important modules to understand are `pyg_builder.py` (for data tensors) and `schema.py` (configuration bounds).
The single biggest risk in predictive legal analysis is predicting perfectly without generalizability: **Text masking in `leakage.py` needs to be aggressively monitored.**

*The easiest wins for improvement are expanding the array of `leakage_phrases` within the config files and incorporating advanced embeddings instead of `sentence-transformers/all-MiniLM-L6-v2` down the pipeline.*
