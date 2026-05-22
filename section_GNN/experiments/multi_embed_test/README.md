# Multi Embed Test

This subfolder runs a controlled encoder-only comparison for the reasoning-focused GNN pipeline.

It reuses the exact reasoning and graph-building code already in:
- `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/preprocess_fixed_open.py`
- `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/build_graph.py`

Training still uses:
- `/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/scripts/train_gnn.py`

What stays fixed:
- the cleaned FIXED_OPEN dataset
- the reasoning-focused graph topology
- the HGT model architecture and training config
- the split seed from `fixed_open_pipeline/fixed_open_reasoning_config.yaml`
- the per-encoder graph cache and split assignments once they are built

What changes:
- only `features.text_encoder`
- only the training seed across repeated runs for the same encoder

Default encoder variants:
- `hashing`
- `bge_m3` with `BAAI/bge-m3`

Additional variants remain available through `--variants`:
- `inlegalbert` with `law-ai/InLegalBERT`
- `legal_longformer` with `joelniklaus/legal-xlm-longformer-base`

Training repeats:
- each encoder is trained `1` time by default with the same split and graph cache
- only the training seed changes across repeats
- the summary reports `mean ± std` for validation and test macro-F1
- the summary also reports the best run selected by validation macro-F1
- text embedding stages now show progress bars for hashing, sentence-transformers, and HuggingFace encoders
- graph debug artifacts now keep `1` reproducible random sample case per encoder config

If you want the fourth slot to use `nomic-embed-text-v1.5` instead, replace the `legal_longformer`
entry in `encoder_variants.yaml` with a `sentence_transformers` config for that model.

## Run

From:

```bash
cd "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph"
```

Run the full test:

```bash
bash "multi embed test/run_fixed_open_multi_embed_test.sh"
```

If you are inside:

```bash
/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/updated_graph
```

then use:

```bash
bash "../multi embed test/run_fixed_open_multi_embed_test.sh"
```

The wrapper is pinned to:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7
```

Useful options:

```bash
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --skip-preprocess
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --limit 200
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --repeat-runs 3
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --repeat-runs 5
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --cuda-visible-devices 4,5,6,7
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --sequential-variants
bash "multi embed test/run_fixed_open_multi_embed_test.sh" --artifact-root /custom/output/root
```

Important:
- this guarantees only GPUs `4,5,6,7` are visible to the pipeline
- preprocess still runs once before the per-encoder jobs
- after preprocess, encoder build/train pipelines run in parallel across the visible GPUs by default
- each subprocess is pinned to one visible GPU, so `torch.device("cuda")` resolves to that assigned GPU
- when variants run in parallel, console logs and progress bars from different variants may interleave

## Outputs

Default output root:

```text
/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/outputs/fin_fraud_labelled_reasoning/multi_embed_test
```

Main artifacts:
- `configs/`: generated per-encoder YAML configs used by the run
- `shared_preprocess/`: one shared preprocessing pass reused by all variants
- `variants/<encoder>/`: encoder-specific embedding cache, graph cache, logs, and train outputs
- `summary/encoder_macro_f1_comparison.csv`: compact comparison table
- `summary/encoder_macro_f1_comparison.md`: readable comparison table
- `summary/consistency_checks.json`: verifies same split assignments and same graph structure across variants
- `variants/<encoder>/outputs/models/<run_name>/multi_run_summary.json`: per-seed repeated-run summary and best-run selection

The comparison summary is centered on:
- `val_macro_f1_mean ± val_macro_f1_std`
- `test_macro_f1_mean ± test_macro_f1_std`
- `best_run_label`
- `best_seed`
- `best_val_macro_f1`
- `best_test_macro_f1`

Accuracy is not used as the primary comparison metric.
