# InLegalLlama Model Comparison

This folder runs `L-NLProc/InLegalLlama` on one filtered fold test split using Hugging Face Transformers plus PEFT.

The first run used the continued-pretraining adapter under:

`INLegalLlama/CPT/llama2_cpt_checkpoint_3000_seq_2048`

The adapter config points to this base model:

`NousResearch/Llama-2-7b-chat-hf`

For the closer published-style comparison, use the separate original-prompt scripts below. They use `0` for rejection/loss and `1` for acceptance/win in the prompt, then map `0 -> -1` before metrics are computed.

The included test set is copied from:

`section_GNN/data/timed_bucket_runs/motor_accidents_timed_mistral/processed/cleaned_cases`

using the `test` rows from:

`section_GNN/outputs/timed_bucket_runs/motor_accidents_timed_mistral/models/motor_accidents_timed_mistral_kfold/kfold/fold_00/predictions.csv`

The runner does not put the gold label in the prompt. Labels are used only after generation for summary metrics.

## Environment

The env has been created as:

```bash
micromamba run -n model_comparison_inlegalllama python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

To recreate it:

```bash
cd "model comparison"
micromamba env create -f environment.yml
```

If any model download is gated, set `HF_TOKEN` before running.

## Smoke Test

This validates paths and sharding without loading the model:

```bash
cd "model comparison"
micromamba run -n model_comparison_inlegalllama python run_inlegalllama.py \
  --input-dir data/motor_accidents_fold_00_test_cases \
  --dry-run \
  --limit 2
```

## Full 8-GPU Run

```bash
cd "model comparison"
bash run_inlegalllama_8gpu.sh
```

Outputs are written under:

`outputs/lnlproc_inlegalllama_motor_accidents_fold00`

Each GPU gets a separate shard process. The wrapper launches eight processes with `CUDA_VISIBLE_DEVICES=0` through `CUDA_VISIBLE_DEVICES=7`, then combines the shard JSONL files and writes `metrics.json`.

To override back to the older full-model repo for a one-off run:

```bash
MODEL_NAME="sudipto-ducs/InLegalLLaMA" MODEL_SUBFOLDER="" ADAPTER_MODE="full" bash run_inlegalllama_8gpu.sh
```

## FactLegalLlama / TathyaNyaya Run

The Hugging Face repo `L-NLProc/TathyaNyaya-and-FactLegalLlama-Large-Language-Models-Based-Models` stores checkpoints inside zip archives. The helper script extracts the selected PEFT adapter into `models/factlegalllama/` and then reuses the same inference and metrics flow.

Default checkpoint:

`nyaya_facts_single`

Available prediction-only checkpoints:

`nyaya_facts_single`, `nyaya_facts_multi`, `nyaya_scrape_single`, `nyaya_scrape_multi`, `nyaya_simplify`

Prepare only:

```bash
cd "model comparison"
micromamba run -n model_comparison_inlegalllama python run_factlegalllama.py \
  --checkpoint nyaya_facts_single \
  --prepare-only
```

Dry run:

```bash
cd "model comparison"
micromamba run -n model_comparison_inlegalllama python run_factlegalllama.py \
  --checkpoint nyaya_facts_single \
  --input-dir data/motor_accidents_fold_00_test_cases \
  --output-dir outputs/factlegalllama_smoke \
  --dry-run \
  --limit 2
```

Full 8-GPU run:

```bash
cd "model comparison"
bash run_factlegalllama_8gpu.sh
```

Outputs are written under:

`outputs/factlegalllama_nyaya_facts_single_motor_accidents_fold00`

To use another checkpoint:

```bash
CHECKPOINT=nyaya_scrape_single bash run_factlegalllama_8gpu.sh
```

These adapters use `meta-llama/Meta-Llama-3-8B` as the base model, so set `HF_TOKEN` if Hugging Face requires access approval for that base model.

## Original-Prompt Runs

These are separate from the earlier JSON `-1/1` scripts and write to separate output folders.

Run the InLegalLlama SFT prediction-only checkpoint:

```bash
cd "model comparison"
bash run_inlegalllama_sft_original_prompt_8gpu.sh
```

Outputs are written under:

`outputs/lnlproc_inlegalllama_sft_pred_only_original_prompt_motor_accidents_fold00`

Run FactLegalLlama with facts-only preprocessing:

```bash
cd "model comparison"
bash run_factlegalllama_facts_original_prompt_8gpu.sh
```

Outputs are written under:

`outputs/factlegalllama_nyaya_facts_single_facts_original_prompt_motor_accidents_fold00`

To use another FactLegalLlama checkpoint with the same facts-only prompt:

```bash
CHECKPOINT=nyaya_scrape_single bash run_factlegalllama_facts_original_prompt_8gpu.sh
```
