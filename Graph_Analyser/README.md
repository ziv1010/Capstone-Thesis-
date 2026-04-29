# Graph_Analyser — GNN Explainability Pipeline

Five-phase pipeline that takes a **frozen, pre-trained Heterogeneous Graph
Transformer (HGT)** from `section_GNN/…/kfold/fold_XX/model.pt` and produces
plain-English explanations for its case-outcome predictions, grounded in
retrieved statutes/provisions/precedents and similar past cases.

## Pipeline

| Phase | Script | Outputs |
|-------|--------|---------|
| 1 & 2: Foundation + Retrieval | `scripts/phase1_2_inference_and_index.py` | `outputs/phase1_2_inference/` — predictions, 64-d case embeddings, FAISS index over train split |
| 3: Explainer Training | `scripts/phase3_train_explainer.py` | `outputs/phase3_explainer/explainer.pt` — trained PGExplainer-style edge masker |
| 4: Extraction | `scripts/phase4_extract_importance.py` | `outputs/phase4_explanations/cases/case_<idx>.json` — top statutes/provisions/precedents/arguments + top-k similar training cases per test case |
| 5: LLM Translation | `scripts/phase5_llm_translate.py` | `outputs/phase5_llm_reasoning/explanation_<idx>.json` — Mistral-Small 24B explanation |

## Environments

Phases 1–4 run in the `graph_explainer` micromamba env (torch 2.6, PyG 2.7,
FAISS). Phase 5 runs in the `llm` env (vLLM 0.11). Both already exist on the
host. No new env needs to be created.

## Running

```bash
cd /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Graph_Analyser

# Full pipeline, default config:
bash scripts/run_all.sh

# Phase 1–4 only (skip LLM):
bash scripts/run_all.sh --skip-llm

# Only explain the first 20 test cases with the LLM:
bash scripts/run_all.sh --limit 20
```

## Configuration

See `configs/default.yaml`. Important fields:

- `graph_cache`: path to the `.pt` HeteroData graph from `section_GNN`.
- `checkpoint_dir`: the `fold_XX` directory containing `model.pt`.
- `model`: HGT architecture hyperparameters (inferred automatically from the
  checkpoint if left partial — `hidden_dim` / `num_layers` are read off the
  state-dict shapes).
- `explainer.*`: training knobs for the heterogeneous PGExplainer.
- `extraction.*`: how many top nodes per category, and how many test cases to
  explain. Setting `top_n_test_cases: 0` explains every test case.
- `llm.model_path`: Mistral-Small 24B snapshot path.

## Heterogeneous PGExplainer — design note

PyG's stock `PGExplainer` is homogeneous and requires a model that accepts
`edge_weight`; HGTConv does not. `src/hetero_pg_explainer.py` implements the
same core idea — a shared MLP scores every edge from its endpoint embeddings
and a relation-specific embedding, trained via Gumbel-sigmoid straight-through
estimation against the frozen HGT's *predicted* label plus a sparsity /
entropy regulariser — but performs edge selection by subset-masking
`edge_index_dict` rather than weighting messages, which keeps the explainer
compatible with HGTConv as-is.

Node importance is derived in Phase 4 as the maximum score across incident
edges, restricted to nodes reachable from the target case node within
`num_layers` hops (the HGT's message-passing radius).

## Outputs layout

```
outputs/
├── phase1_2_inference/
│   ├── case_embeddings.npy
│   ├── predictions.{npy,csv}
│   ├── probabilities.npy
│   ├── train_faiss.index
│   ├── train_indices.npy
│   ├── effective_model_cfg.json
│   └── summary.json
├── phase3_explainer/
│   ├── explainer.pt
│   ├── explainer_cfg.json
│   └── training_history.json
├── phase4_explanations/
│   ├── manifest.json
│   └── cases/case_<node_idx>.json
└── phase5_llm_reasoning/
    ├── index.json
    └── explanation_<node_idx>.json
```
