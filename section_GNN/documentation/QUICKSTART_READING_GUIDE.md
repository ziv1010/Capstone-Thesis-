# QUICKSTART_READING_GUIDE

If you are a new researcher starting with this massive legal GNN codebase, the worst thing you can do is start reading randomly. The pipeline is heavily decoupled, meaning files like `extract.py` have zero knowledge of PyTorch Geometric (PyG), and `models/hetero_gnn.py` has no idea what `appellant_won` means in strings.

Here is the exact order you should read the files to understand the flow without getting lost in the weeds.

## Recommended Reading Order

### 1. `configs/gnn_case_star...yaml`
- **Why:** Everything builds off the config. Before you look at the Python code, understand *what* it is building.
- **Look for:** `labels.task_mode`, `training.repeat_runs`, `preprocessing.leakage_phrases`, and `graph.include_node_types`.

### 2. `scripts/train_gnn.py`
- **Why:** This is the entry point for the core ML loop.
- **Look for:** How it loads a cached PyG bundle (`bundle["data"]`). Understand that by this stage, the node features and structural topology are entirely frozen.

### 3. `src/graph/schema.py`
- **Why:** The blueprint. The ontology.
- **Look for:** `ENTITY_NODE_TYPES`, `RELATION_DEFINITIONS`, and the `CleanedCase` dataclass. Notice the strict definitions of `preamble`, `petitioner_arguments`, and bridging edges like `cites_statute`.

### 4. `src/models/hetero_gnn.py`
- **Why:** The actual neural network.
- **Look for:** `HGTConv`. Notice the `_encode_inputs` function projecting raw structural data down to `128d`. See how the `mlp_head.py` takes only the `hidden["case"]` representation to emit classification probabilities.

### 5. `src/training/train.py`
- **Why:** Puts the pieces of PyTorch and Scikit-Learn together.
- **Look for:** `evaluate_split()`. The backpropagation algorithm. The fact that `early_stopping` logic relies exclusively on `Macro F1`.

### 6. `src/graph/pyg_builder.py`
- **Why:** To understand how string lists turn into `torch.Tensor` blocks.
- **Look for:** The `_case_scalar_vector` generation scaling everything manually (`/ 5000.0`, `/ 100.0`). The `_build_embeddings` caching the heavy textual `sentence-transformers`.

### 7. (Optional) `src/preprocessing/extract.py` and `leakage.py`
- **Why:** Only read this deeply if you are modifying what text is analyzed or diagnosing 100% false-positive accuracies pointing to data leakage.
- **Look for:** `_infer_lawyer_side` (the context-window matching logic), and `remove_or_mask_leakage`.

---

## Easiest way to grok it

Run the pipeline **step-by-step** and drop `print()` / breakpoints. The pipeline cleanly writes to `.pt` files.
- Run Step 1 (`preprocess`).
- Run Step 2 (`build_graph`). Read the `graph_metadata.json` explicitly.
- Run Step 3 (`train`). Watch the epochs print.
