# Fixed Open Pipeline

This folder converts sentence-level JSON files from:

- `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/fin_fraud_labelled/labelled_jsons`

into `cleaned_cases` that are compatible with the reasoning-focused graph builder in:

- `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/build_graph.py`

## What It Does

- reads sentence-level OpenNyAI output with `sentences`, `preamble_end_char_offset`, and `case_outcome_score`
- builds leakage-safe section texts:
  - `preamble`
  - `facts`
  - `arguments`
  - `petitioner_arguments`
  - `respondent_arguments`
  - `other_lawyer_arguments`
- converts sentence entities into the `CleanedCase` entity schema expected by `section_GNN`
- preserves binary labels with `case_outcome_score` values `"0"` and `"-1"` collapsed into `"-1"`, and `"1"` kept as `"1"`
- skips files that do not contain `case_outcome_score` and records them in `preprocess_summary.fixed_open.json`

## Default Role Mapping

- `PREAMBLE` -> `preamble`
- `FAC` -> `facts`
- `ARG_PETITIONER` -> `petitioner_arguments` and `arguments`
- `ARG_RESPONDENT` -> `respondent_arguments` and `arguments`
- `PRE_RELIED`, `PRE_NOT_RELIED`, `STA` -> `other_lawyer_arguments` and `arguments`
- `ANALYSIS`, `ISSUE`, `NONE`, `RATIO`, `RLC`, `RPC` are dropped conservatively

## Commands

Use the existing micromamba environment at:

- `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star`

Preprocess:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/preprocess_fixed_open.py" \
  --config "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/fixed_open_reasoning_config.yaml"
```

Build the reasoning-focused graph:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/build_graph.py" \
  --config "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/fixed_open_reasoning_config.yaml"
```

Train the GNN on the built graph:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/scripts/train_gnn.py" \
  --config "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/fixed_open_reasoning_config.yaml" \
  --run-name fin_fraud_labelled_reasoning
```
