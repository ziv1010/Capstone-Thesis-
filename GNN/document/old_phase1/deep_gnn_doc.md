# Deep GNN Architecture & Training Documentation

> **Directory:** `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN`
> **Config:** `configs/gnn_case_star.yaml`
> **Model:** Heterogeneous Graph Transformer (HGT), 3 layers, `hidden_dim=128`

---

## Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Full Graph Schema — Nodes & Edges](#2-full-graph-schema--nodes--edges)
3. [Exactly What is Stored in Each Node](#3-exactly-what-is-stored-in-each-node)
4. [What Every Edge Does in the GNN](#4-what-every-edge-does-in-the-gnn)
5. [Layer-by-Layer Tensor State](#5-layer-by-layer-tensor-state)
6. [Receptive Field — Multi-Hop Information Flow](#6-receptive-field--multi-hop-information-flow)
7. [Train / Val / Test — Exact Procedure](#7-train--val--test--exact-procedure)
8. [Outputs](#8-outputs)

---

## 1. Pipeline Overview

```
Raw JSON files (OpenNyAI-style)
        │
        ▼ src/preprocessing/extract.py
  ┌─────────────────────────────────────┐
  │  Leakage Filter & Audit             │
  │  • drop forbidden top-level fields  │
  │  • mask outcome phrases in text     │
  │  • drop decision/RPC annotations    │
  └─────────────────────────────────────┘
        │
        ▼ src/graph/case_star_builder.py
  ┌─────────────────────────────────────┐
  │  Local Case Star Graph              │
  │  • 1 case node + text nodes         │
  │  • entity nodes from annotations    │
  └─────────────────────────────────────┘
        │
        ▼ src/graph/global_graph_builder.py
  ┌─────────────────────────────────────┐
  │  Global Authority Graph Merge       │
  │  • authority nodes shared globally  │
  │  • party nodes stay case-local      │
  └─────────────────────────────────────┘
        │
        ▼ src/graph/pyg_builder.py
  ┌─────────────────────────────────────┐
  │  Build PyG HeteroData               │
  │  • compute embeddings (cached)      │
  │  • concat scalars → node.x          │
  │  • attach masks & labels to case    │
  │  • ToUndirected() → bidirectional   │
  └─────────────────────────────────────┘
        │
        ▼ src/training/train.py
  ┌─────────────────────────────────────┐
  │  HGT 3-layer forward pass           │
  │  cross_entropy(train mask)          │
  │  AdamW + early stopping             │
  │  evaluate on val/test masks         │
  └─────────────────────────────────────┘
        │
        ▼  outputs/models/<run_name>/
  model.pt · metrics.json · predictions.csv
```

---

## 2. Full Graph Schema — Nodes & Edges

![Full Graph Schema](file:///scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/1_full_graph_schema.png)

### 2.1 Node Types

| Node Type | Scope | Description |
|---|---|---|
| `case` | Case-local | Root node. One per legal case. Carries all case-level metadata and combined text. |
| `preamble` | Case-local | Raw preamble text, leakage-masked. |
| `facts` | Case-local | Facts-section text, leakage-masked. |
| `arguments` | Case-local | Arguments-section text, leakage-masked. Central hub for legal citation edges. |
| `petitioner` | Case-local | Party who filed the petition/appeal. NOT shared across cases by default. |
| `respondent` | Case-local | Opposing party. NOT shared across cases. |
| `court` | **Shared globally** | Normalised court name. Merges across all cases. |
| `judge` | **Shared globally** | Normalised judge name. One node shared across all cases they appear in. |
| `lawyer` | **Shared globally** | Generic lawyer (side unknown). |
| `petitioner_lawyer` | **Shared globally** | Lawyer identified via context as representing the petitioner. |
| `defence_lawyer` | **Shared globally** | Lawyer identified as representing the respondent/state. |
| `statute` | **Shared globally** | Normalised statute name (e.g., "Indian Penal Code"). |
| `provision` | **Shared globally** | Specific provision citation (e.g., "Section 302"). |
| `precedent` | **Shared globally** | Cited precedent case name. |
| `org` | **Shared globally** | Organisation entity from NER. |
| `gpe` | **Shared globally** | Geo-political entity from NER. |
| `date` | **Shared globally** | Date entity from NER. |
| `case_number` | **Shared globally** | Case number string. |

> **"Shared globally"** means the same node object is reused across all cases that mention that entity. This is how the global authority graph creates cross-case information paths.

### 2.2 All Edge Types

| Edge (src → rel → dst) | Mandatory | Created by | Purpose |
|---|---|---|---|
| `case → has_preamble → preamble` | Yes | `case_star_builder.py` | Links case to its preamble text node |
| `case → has_facts → facts` | Yes | `case_star_builder.py` | Links case to its facts text node |
| `case → has_arguments → arguments` | Yes | `case_star_builder.py` | Links case to its arguments text node |
| `case → has_petitioner → petitioner` | Yes | `case_star_builder.py` | Links case to petitioner party |
| `case → has_respondent → respondent` | Yes | `case_star_builder.py` | Links case to respondent party |
| `case → heard_in → court` | Yes | `case_star_builder.py` | Links case to court authority |
| `case → decided_by_bench → judge` | Yes | `case_star_builder.py` | Links case to judge(s) |
| `case → has_lawyer → lawyer` | Yes | `case_star_builder.py` | Links case to generic lawyer |
| `case → has_petitioner_lawyer → petitioner_lawyer` | Yes | `case_star_builder.py` | Side-specific lawyer link |
| `case → has_defence_lawyer → defence_lawyer` | Yes | `case_star_builder.py` | Side-specific lawyer link |
| `case → mentions_org → org` | Optional | `case_star_builder.py` | NER organisation link |
| `case → mentions_gpe → gpe` | Optional | `case_star_builder.py` | NER geo-entity link |
| `case → has_date → date` | Optional | `case_star_builder.py` | Date entity link |
| `case → has_case_number → case_number` | Optional | `case_star_builder.py` | Case number link |
| `arguments → cites_statute → statute` | Optional | `case_star_builder.py` | Statute cited in arguments section |
| `arguments → cites_provision → provision` | Optional | `case_star_builder.py` | Provision cited in arguments section |
| `arguments → cites_precedent → precedent` | Optional | `case_star_builder.py` | Precedent cited in arguments section |
| `provision → belongs_to_statute → statute` | Optional | `case_star_builder.py` | Hierarchical legal link |
| `petitioner_lawyer → citation → arguments` | Optional | `case_star_builder.py` | Lawyer's argument citation |
| `defence_lawyer → citation → arguments` | Optional | `case_star_builder.py` | Lawyer's argument citation |
| `provision → used_in_arguments → arguments` | Bridging | `case_star_builder.py` | Shortcut: reduces provision→case to 2 hops |
| `statute → used_in_arguments → arguments` | Bridging | `case_star_builder.py` | Shortcut: reduces statute→case to 2 hops |
| `petitioner → is_party_in_arguments → arguments` | Bridging | `case_star_builder.py` | Party stance visible at arguments node |
| `respondent → is_party_in_arguments → arguments` | Bridging | `case_star_builder.py` | Party stance visible at arguments node |
| `judge → presided_arguments → arguments` | Bridging | `case_star_builder.py` | Judge context at arguments node |

> **After `ToUndirected()`** the graph adds reverse edges for all of the above, so messages flow in both directions during HGT convolution.

---

## 3. Exactly What is Stored in Each Node

![Node Feature Storage](file:///scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/2_node_feature_storage.png)

All nodes have the same tensor structure:

```
data[node_type].x  shape = (N_nodes, feature_dim)

feature_dim = embedding_dim + scalar_dim
            = 384          + 12
            = 396
```

The tensor is built in `pyg_builder.py` as:
```python
node_features[idx] = np.concatenate([embedding, scalars], axis=0)
```

### 3.1 `case` Node

**Text fed to encoder:** Concatenation of `preamble + facts + arguments` (all leakage-masked).

| Slot | Feature | Value range |
|---|---|---|
| 0–383 | Text embedding (all-MiniLM-L6-v2, 384-d) | ℝ |
| 384 | `respondent_count` / 100 | [0, 1] |
| 385 | `judge_count` / 100 | [0, 1] |
| 386 | `lawyer_count` / 100 | [0, 1] |
| 387 | `statute_count` / 100 | [0, 1] |
| 388 | `provision_count` / 100 | [0, 1] |
| 389 | `precedent_count` / 100 | [0, 1] |
| 390 | `preamble_length` / 5000 | [0, 1] |
| 391 | `facts_length` / 5000 | [0, 1] |
| 392 | `arguments_length` / 5000 | [0, 1] |
| 393 | `case_year` normalised: `(year − 1900) / 200` | [0, 1] |
| 394 | `petition_type_known` (binary) | {0, 1} |
| 395 | `petition_type_hash` (stable SHA256 → [0,1]) | [0, 1] |

**Extra attributes attached only to `case` nodes:**

| Attribute | Type | Contents |
|---|---|---|
| `data["case"].y` | `LongTensor (N,)` | class index: 0=lose, 1=win |
| `data["case"].train_mask` | `BoolTensor (N,)` | True for training cases |
| `data["case"].val_mask` | `BoolTensor (N,)` | True for validation cases |
| `data["case"].test_mask` | `BoolTensor (N,)` | True for test cases |
| `data["case"].case_id` | `list[str]` | Stem of the source JSON filename |
| `data["case"].file_name` | `list[str]` | Full source filename |
| `data["case"].raw_label` | `list[str]` | Original string label before mapping |

### 3.2 Text Nodes (`preamble`, `facts`, `arguments`)

**Text:** The actual section text.

| Slot | Feature |
|---|---|
| 0–383 | Text embedding of section content |
| 384 | `text_length` / 5000 |
| 385 | `is_preamble` (1-hot) |
| 386 | `is_facts` (1-hot) |
| 387 | `is_arguments` (1-hot) |
| 388 | `cited_statute_count` / 100 (args only) |
| 389 | `cited_provision_count` / 100 (args only) |
| 390 | `cited_precedent_count` / 100 (args only) |
| 391 | `petitioner_lawyer_count` / 100 |
| 392 | `defence_lawyer_count` / 100 |
| 393 | `petitioner_count` / 100 |
| 394 | `respondent_count` / 100 |
| 395 | `judge_count` / 100 |

### 3.3 Entity Nodes (`court`, `judge`, `lawyer`, `petitioner_lawyer`, `defence_lawyer`, `petitioner`, `respondent`, `statute`, `provision`, `precedent`, `org`, `gpe`, `date`, `case_number`)

**Text:** The canonical entity name string.

| Slot | Feature |
|---|---|
| 0–383 | Text embedding of canonical name |
| 384 | `mention_count` / 100 |
| 385 | `first_seen_preamble` (1-hot) |
| 386 | `first_seen_facts` (1-hot) |
| 387 | `first_seen_arguments` (1-hot) |
| 388 | `seen_in_arguments` (binary) |
| 389 | `seen_in_preamble` (binary) |
| 390 | `local_case_frequency` / 100 |
| 391 | `global_case_frequency` / 100 *(populated during global graph merge)* |
| 392 | `degree` / 100 *(edge-degree of this node across all case connections)* |
| 393 | `is_shared_node` (1 = appears in >1 case) |
| 394–395 | (zero-padded to match scalar_dim) |

### 3.4 Node Metadata (not in `.x`, stored in `GraphNode.metadata`)

Beyond `.x`, each `GraphNode` carries a `metadata` dict saved in the JSON graph dump:
- For entity nodes: `raw_name`, `canonical_name`, `mention_count`, `first_seen_section`, `seen_in_arguments`, `seen_in_preamble`, `linked_statute_canonical`, `is_shared_node`
- For text nodes: `case_id`, `section`, `text_length`, citation counts
- For case nodes: all metadata from `_build_case_metadata()` + `case_id`, `file_name`, `raw_label`

---

## 4. What Every Edge Does in the GNN

Edges in this HGT model are **typed**. Each `(src_type, relation, dst_type)` triple gets its own learned projection matrices for key, query, and value in the attention mechanism.

### 4.1 How HGT Uses Edges

For each target node `t`, the HGT layer does:
```
α(s→t) = softmax over s ∈ N(t) of  [Q(h_t) · K(h_s)] / √d_head
h_t_new = Σ_s  α(s→t) · V(h_s)
h_t ← LayerNorm( h_t_old + Dropout(h_t_new) ) → ReLU
```
where Q, K, V projections are specific to `(src_type, relation, dst_type)`.

### 4.2 Semantic Role of Each Edge Type

| Edge | What the GNN learns via this edge |
|---|---|
| `case → has_preamble/facts/arguments` | Case node absorbs the semantic content of all three text sections. |
| `case → heard_in → court` | Case learns from which court this case was adjudicated (shared court allows cross-case generalisation). |
| `case → decided_by_bench → judge` | Case absorbs judge identity/behaviour. Shared `judge` node accumulates signal from all cases that judge presided over. |
| `case → has_petitioner/respondent` | Case learns party identity. Party nodes remain local, preventing cross-case party confusion. |
| `case → has_*_lawyer` | Case learns advocate experience. Shared lawyer nodes learn from their entire caseload. |
| `arguments → cites_statute/provision` | Arguments node gains legal citation context, allowing case to reason about which laws were invoked. |
| `arguments → cites_precedent` | Arguments node incorporates cited precedents. |
| `provision → belongs_to_statute → statute` | Provisions link up to their parent statute, giving statute nodes richer legal context. |
| `petitioner_lawyer → citation → arguments` | Lawyer explicitly tied to specific arguments, allowing per-side legal argument attribution. |
| `defence_lawyer → citation → arguments` | Same as above for the defending side. |
| `provision → used_in_arguments (bridging)` | **Shortcut edge.** Reduces the hop distance from provision to case from 3 hops to 2. Without this, Layer 2 cannot see provision signal at the case node in a 2-layer model. |
| `statute → used_in_arguments (bridging)` | Same shortcut for statute. |
| `petitioner/respondent → is_party_in_arguments (bridging)` | Party signals flow directly to arguments, allowing the model to learn which party's arguments dominated. |
| `judge → presided_arguments (bridging)` | Judge context directly available to the arguments node in fewer hops. |

---

## 5. Layer-by-Layer Tensor State

![Layer by Layer State](file:///scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/3_layer_by_layer.png)

### Stage 0 — Raw Input

```python
data[node_type].x   # shape: (N_nodes, 396)
```
Each node type has a different number of nodes (`N_nodes`) but the same feature width of **396** (`384 embed + 12 scalars`).

### Projection Stage — Before Layer 1

Defined in `hetero_gnn.py`:
```python
self.input_projections = nn.ModuleDict({
    node_type: nn.Linear(input_dims[node_type], hidden_dim=128)
    for node_type in metadata[0]
})
self.type_embeddings = nn.ParameterDict({
    node_type: nn.Parameter(torch.zeros(1, 128))
    for node_type in metadata[0]
})
```

All node type tensors are projected to a uniform shape:
```
hidden[node_type] shape: (N_nodes, 128)
= Linear(396→128)(x) + type_embedding(1,128)  # broadcast
```

### Layer 1 (HGTConv #0) — 1-Hop Aggregation

- Each node type receives messages from its direct neighbours.
- **`case`** absorbs: preamble, facts, arguments, court, judge, petitioner, respondent, lawyers, org, gpe, date, case_number.
- **`arguments`** absorbs: case, statute, provision (via `cites_*`), lawyers (via `citation`), petitioner/respondent/judge (via bridging edges).

```python
# Forward pass internals:
conv_out = HGTConv_L1(hidden, edge_index_dict)
updated[node_type] = LayerNorm(hidden[node_type] + Dropout(conv_out[node_type]))
updated[node_type] = ReLU(updated[node_type])
```

After Layer 1: `hidden[node_type]` shape still `(N_nodes, 128)` but now contains **1-hop neighbourhood context**.

### Layer 2 (HGTConv #1) — 2-Hop Aggregation

- **`case`** now indirectly aggregates the neighbours of its Layer-1 neighbours:
  - Via `arguments`: `statute`, `provision`, `precedent` content (from `cites_*` edges).
  - Via `judge`/`lawyer`: information about all other cases those nodes connect to.
- **`arguments`** now aggregates 2nd-order neighbours: the statute that a provision belongs to (via `provision → belongs_to_statute`).

### Layer 3 (HGTConv #2) — 3-Hop Aggregation

- **`case`** can now see 3-hop paths such as:
  - `case → arguments → provision → statute` (statute signal at case node level 3)
  - Shared judge/court nodes accumulate signal from all cases in Layers 1 and 2, and now contribute that multi-case experience back to this case at Layer 3.
- This is the furthest reach without bridging edges. **Bridging edges shorten critical paths** (e.g., `statute → used_in_arguments` makes statute reachable in 2 hops instead of 3).

### Classification Head

After Layer 3:
```python
logits = self.classifier(hidden["case"])
# classifier = Linear(128→128) → ReLU → Dropout(0.25) → Linear(128→num_classes)
# logits shape: (N_case, 2)  [lose, win]
```
Only `case` node representations are passed to the MLP. All other node types are discarded.

---

## 6. Receptive Field — Multi-Hop Information Flow

![Receptive Field](file:///scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/4_receptive_field.png)

### Path Table — What `case` Can See at Layer 3

| Hops | Path | Via Edge |
|---|---|---|
| 1 | `case` ← preamble | `has_preamble` |
| 1 | `case` ← facts | `has_facts` |
| 1 | `case` ← arguments | `has_arguments` |
| 1 | `case` ← court | `heard_in` |
| 1 | `case` ← judge | `decided_by_bench` |
| 1 | `case` ← petitioner | `has_petitioner` |
| 1 | `case` ← respondent | `has_respondent` |
| 1 | `case` ← petitioner_lawyer | `has_petitioner_lawyer` |
| 1 | `case` ← defence_lawyer | `has_defence_lawyer` |
| 2 | `case` ← arguments ← statute | via `cites_statute` |
| 2 | `case` ← arguments ← provision | via `cites_provision` |
| 2 | `case` ← arguments ← petitioner_lawyer | via `citation` |
| 2 | `case` ← arguments ← petitioner | via `is_party_in_arguments` *(bridging)* |
| 2 | `case` ← arguments ← respondent | via `is_party_in_arguments` *(bridging)* |
| 2 | `case` ← arguments ← judge | via `presided_arguments` *(bridging)* |
| 2 | `case` ← arguments ← statute | via `used_in_arguments` *(bridging)* |
| 3 | `case` ← arguments ← provision ← statute | via `belongs_to_statute` |
| 3 | `case` ← judge ← [other cases] ← [their entities] | cross-case signal via shared judge |

---

## 7. Train / Val / Test — Exact Procedure

![Training Loop](file:///scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/document/5_training_loop.png)

### 7.1 Label Preparation

Labels come from `case_outcome_label` in the raw JSON (excluded from node features). The pipeline maps them:

```yaml
binary_map:
  appellant_won: win
  appellant_lost: lose
  postponed_or_procedural: null  # → dropped from dataset
```

`label_names = ["lose", "win"]` → integer indices 0, 1.

Cases with `null` mapping (procedural/postponed) are **dropped** before graph construction.

### 7.2 Split Assignment (`src/training/dataset.py`)

```yaml
splits:
  mode: random
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
  stratify: true
  random_state: 42
```

**How it works:**
1. `train_test_split(indices, y, train_size=0.70, stratify=y, random_state=42)` → `train_idx` + `temp_idx`
2. `train_test_split(temp_idx, test_size=0.50, stratify=y_temp, random_state=42)` → `val_idx` + `test_idx`
   - (test_size=0.50 of remaining 30% → 15% each)
3. Result: `{case_id: "train" | "val" | "test"}` dict
4. Saved to `data/graph_cache/split_assignments.json`

Three boolean masks are attached to the PyG data object:
```python
data["case"].train_mask  # shape (N_case,)
data["case"].val_mask
data["case"].test_mask
```

Also supported: `mode = "year"` (group by case year) or `mode = "court"` (group by court) using `GroupShuffleSplit` for temporal/spatial out-of-distribution evaluation.

### 7.3 Class Weights

```python
class_weights = compute_class_weight("balanced", classes=[0,1], y=y_train)
# e.g.  [1.42, 0.74]  if win is more frequent
class_weights_tensor = torch.tensor(class_weights).to(device)
```

Used in every training step to penalise the dominant class.

### 7.4 Optimizer

```yaml
training:
  lr: 0.001
  weight_decay: 0.00001
```

```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
```

### 7.5 Training Loop (Transductive)

**Transductive** means: the full graph (all nodes and edges) is always passed through the GNN. Only the **supervision signal** (loss computation) is restricted to `train_mask` nodes.

```
for epoch in 1..60:
    model.train()
    logits, _ = model(data.x_dict, data.edge_index_dict)   # FULL GRAPH
    loss = cross_entropy(logits[train_mask], y[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()

    model.eval()
    with no_grad:
        logits_eval, embeddings = model(data.x_dict, data.edge_index_dict)
        probs = softmax(logits_eval)
        preds = probs.argmax(-1)

    val_macro_f1 = f1_score(y[val_mask], preds[val_mask], average="macro")

    if val_macro_f1 >= best_val_macro_f1:
        save best_state_dict
        reset patience_counter
    else:
        patience_counter += 1

    if patience_counter >= 15:
        EARLY STOP
```

### 7.6 Early Stopping

```yaml
early_stopping_patience: 15
```

Monitored metric: **validation macro F1** (chosen because the dataset is imbalanced).

The best model weights are restored after training completes (`model.load_state_dict(best_state)`).

### 7.7 Final Evaluation

After restoring the best model:

```python
# One final full-graph forward pass
logits, hidden = model(data.x_dict, data.edge_index_dict)
probs = softmax(logits)
preds = probs.argmax(-1)

# Metrics computed separately per split
for split in ["train", "val", "test"]:
    mask = {train: train_mask, val: val_mask, test: test_mask}[split]
    metrics[split] = evaluate_split(y[mask], preds[mask], probs[mask], label_names)
```

### 7.8 Metrics Computed

| Metric | How |
|---|---|
| `accuracy` | `sklearn accuracy_score` |
| `macro_f1` | `f1_score(average="macro")` — primary monitored metric |
| `micro_f1` | `f1_score(average="micro")` |
| `roc_auc` | `roc_auc_score(y_true, probs[:, 1])` (binary only) |
| `pr_auc` | `average_precision_score(y_true, probs[:, 1])` (binary only) |
| `per_class` | Per-class precision, recall, F1, support |
| `confusion_matrix` | Raw confusion matrix (saved as PNG) |

---

## 8. Outputs

| File | Location | Contents |
|---|---|---|
| `model.pt` | `outputs/models/<run_name>/` | Best-epoch state dict |
| `metrics.json` | `outputs/models/<run_name>/` | Full metrics for all splits + training history |
| `predictions.csv` | `outputs/models/<run_name>/` | Per-case: case_id, split, true label, pred label, confidence |
| `confusion_matrix_test.png` | `outputs/models/<run_name>/` | Test set confusion matrix |
| `case_star_global_graph.pt` | `data/graph_cache/` | Serialised PyG HeteroData object |
| `graph_metadata.json` | `data/graph_cache/` | Node counts, edge types, feature dims |
| `node_mappings.json` | `data/graph_cache/` | `{node_type: {node_key: index}}` |
| `split_assignments.json` | `data/graph_cache/` | `{case_id: "train"/"val"/"test"}` |
| `all_nodes_*.npz` | `data/embeddings_cache/` | Cached embedding matrix (avoids recomputing) |
| `*.json` (audit) | `data/audits/` | Per-case leakage audit: dropped fields, matched phrases |

---

## Key Hyperparameters Summary

| Parameter | Value | Location |
|---|---|---|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | `configs/gnn_case_star.yaml` |
| Embedding dim | 384 | Model output |
| Scalar dim | 12 | Config `case_scalar_names` / `entity_scalar_names` |
| Total feature dim | 396 | `embedding_dim + scalar_dim` |
| GNN architecture | HGT | `model.architecture` |
| Hidden dim | 128 | `model.hidden_dim` |
| Num layers | 3 | `model.num_layers` |
| Attention heads | 4 | `model.num_heads` |
| Dropout | 0.25 | `model.dropout` |
| MLP hidden dim | 128 | `model.mlp_hidden_dim` |
| Epochs max | 60 | `training.epochs` |
| Learning rate | 1e-3 | `training.lr` |
| Weight decay | 1e-5 | `training.weight_decay` |
| Early stop patience | 15 | `training.early_stopping_patience` |
| Class weighting | balanced | `training.class_weight` |
| Split mode | random + stratified | `splits.mode` |
| Train / Val / Test | 70 / 15 / 15 % | `splits.*_size` |
| Task mode | binary (lose / win) | `labels.task_mode` |
