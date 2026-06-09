# Graph_Analyser - Bucket-Locked GNN Diagnostics

This pipeline explains predictions from a frozen HGT model trained in
`section_GNN`. A config names the model scope, and the loader resolves and
validates the matching graph cache, checkpoint, cleaned-case directory, fold
split file, Phase 4 bundles, and Phase 1-2 prediction artifacts.

For single-bucket models, use a single bucket such as `fin_fraud`; cross-bucket
artifacts are rejected. For a model trained on the mixed dataset, use the
explicit `configs/cross_bucket.yaml` config.

There is no LLM stage. The analyzer now produces structured graph evidence and
quantitative diagnostics only, so explanations stay tied to the bucket/model
you selected.

## Pipeline

| Phase | Script | Outputs |
|-------|--------|---------|
| 1 & 2: Frozen Inference | `scripts/phase1_2_inference_and_index.py` | predictions, probabilities, case embeddings, split metadata |
| 3: Explainer Training | `scripts/phase3_train_explainer.py` | trained heterogeneous PGExplainer edge masker |
| 4: Evidence Extraction | `scripts/phase4_extract_importance.py` | bucket-local top graph/legal nodes per selected test case |
| 6: Evidence Diagnostic | `scripts/phase6_misclass_diagnostic.py` | training-label distribution and top-k support sweeps for surfaced evidence |
| 7: Embedding Neighbours | `scripts/phase7_topk_embedding.py` | nearest training cases in frozen GNN embedding space |

## Running

```bash
cd Graph_Analyser_OLD

# Full bucket-locked graph diagnostic pipeline.
bash scripts/run_all.sh

# Smoke config: three test cases, short explainer training.
bash scripts/run_all.sh --config configs/smoke.yaml --p4-gpus 1

# Cross-bucket model: uses the cross_bucket_total_dataset graph and checkpoint.
bash scripts/run_all.sh --config configs/cross_bucket.yaml

# Add embedding nearest-neighbour reports for untraceable cases.
bash scripts/run_all.sh --phase7 --phase7-only-untraceable --phase7-limit 25

# Run Phase 6 only after Phase 4 exists.
micromamba run -n graph_explainer python scripts/phase6_misclass_diagnostic.py \
  --config configs/default.yaml --all --skip-plots

# Include every connected training case for each surfaced statute/provision/precedent.
# Omit this flag to keep only the first 50 connected cases per legal node.
micromamba run -n graph_explainer python scripts/phase6_misclass_diagnostic.py \
  --config configs/cross_bucket.yaml --all --skip-plots --connected-case-limit 0
```

## Configuration

Use `configs/default.yaml` or copy it for a new run. The key fields are:

- `bucket`: one of `family_matrimonial`, `fin_fraud`, `land_property`,
  `motor_accidents`, `sexual_offences`, or `cross_bucket_total_dataset`.
- `graph_variant`: graph-cache suffix for the selected bucket, for example
  `party_args_preamble`.
- `model_variant`: model directory suffix before `_kfold`, for example
  `party_args_preamble_lr_decay`.
- `fold`: checkpoint fold directory, for example `fold_00`.
- `allow_cross_bucket`: defaults to `false`; set it to `true` only with
  `bucket: cross_bucket_total_dataset`.

If `graph_cache`, `checkpoint_dir`, or `cleaned_case_dir` are omitted, they are
resolved from the selected bucket and variant. If they are provided manually,
their paths are checked against `bucket`.

Phase 4 emits target-case text, top legal nodes, broader graph nodes, and raw
argument-role nodes. It does not emit similar-case text snippets.

## Outputs Layout

The default output root is resolved from bucket, model variant, and fold:

```text
outputs/<bucket>_<model_variant>_<fold>/
├── phase1_2_inference/
├── phase3_explainer/
├── phase4_explanations/
├── phase6_misclass_diagnostic/
└── phase7_topk_embedding/
```
