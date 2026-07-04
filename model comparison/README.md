# 🤖 model comparison — Legal-LLM Baselines vs the GNN

> Supporting experiment · benchmarks published legal LLMs (**InLegalLlama**,
> **FactLegalLlama / TathyaNyaya**) on the **same held-out fold test cases** used by the GNN,
> so the thesis can compare graph models against LLM baselines fairly.

## 🧾 Test Set

Cases are copied from
`section_GNN/data/timed_bucket_runs/motor_accidents_timed_mistral/processed/cleaned_cases`,
selecting the `test` rows of
`section_GNN/outputs/timed_bucket_runs/motor_accidents_timed_mistral/models/motor_accidents_timed_mistral_kfold/kfold/fold_00/predictions.csv`.

The runner **never puts the gold label in the prompt** — labels are used only after
generation to compute summary metrics.

## 🧪 Environment

```bash
cd "model comparison"
micromamba env create -f environment.yml     # env: model_comparison_inlegalllama
```

Set `HF_TOKEN` if any model/base download is gated.

---

## 🦙 InLegalLlama Runs

The first run used the continued-pretraining adapter
`INLegalLlama/CPT/llama2_cpt_checkpoint_3000_seq_2048` from `L-NLProc/InLegalLlama`
(base model: `NousResearch/Llama-2-7b-chat-hf`).

```bash
cd "model comparison"

# Smoke test (paths + sharding, no model load):
micromamba run -n model_comparison_inlegalllama python run_inlegalllama.py \
  --input-dir data/motor_accidents_fold_00_test_cases --dry-run --limit 2

# Full 8-GPU run (one shard per GPU, combined JSONL + metrics.json):
bash run_inlegalllama_8gpu.sh
# → outputs/lnlproc_inlegalllama_motor_accidents_fold00

# One-off with the older full-model repo:
MODEL_NAME="sudipto-ducs/InLegalLLaMA" MODEL_SUBFOLDER="" ADAPTER_MODE="full" bash run_inlegalllama_8gpu.sh
```

## ⚖️ FactLegalLlama / TathyaNyaya Runs

The HF repo `L-NLProc/TathyaNyaya-and-FactLegalLlama-Large-Language-Models-Based-Models`
stores checkpoints in zip archives; `prepare_factlegalllama_adapter.py` extracts the selected
PEFT adapter into `models/factlegalllama/` (base model: `meta-llama/Meta-Llama-3-8B`).

Prediction-only checkpoints: `nyaya_facts_single` (default), `nyaya_facts_multi`,
`nyaya_scrape_single`, `nyaya_scrape_multi`, `nyaya_simplify`.

```bash
cd "model comparison"

# Prepare only / dry run:
micromamba run -n model_comparison_inlegalllama python run_factlegalllama.py \
  --checkpoint nyaya_facts_single --prepare-only
micromamba run -n model_comparison_inlegalllama python run_factlegalllama.py \
  --checkpoint nyaya_facts_single --input-dir data/motor_accidents_fold_00_test_cases \
  --output-dir outputs/factlegalllama_smoke --dry-run --limit 2

# Full 8-GPU run:
bash run_factlegalllama_8gpu.sh
# → outputs/factlegalllama_nyaya_facts_single_motor_accidents_fold00

CHECKPOINT=nyaya_scrape_single bash run_factlegalllama_8gpu.sh   # other checkpoints
```

## 📝 Original-Prompt Runs

For the closest published-style comparison, separate scripts use the papers' original prompt
format (`0` = rejection/loss, `1` = acceptance/win; `0` is mapped to `-1` before metrics):

```bash
cd "model comparison"
bash run_inlegalllama_sft_original_prompt_8gpu.sh
# → outputs/lnlproc_inlegalllama_sft_pred_only_original_prompt_motor_accidents_fold00

bash run_factlegalllama_facts_original_prompt_8gpu.sh
# → outputs/factlegalllama_nyaya_facts_single_facts_original_prompt_motor_accidents_fold00

CHECKPOINT=nyaya_scrape_single bash run_factlegalllama_facts_original_prompt_8gpu.sh
```

## 🗂️ Subfolders

| Folder | Contents |
|--------|----------|
| [`data/`](data/README.md) | 📤 Copied held-out test cases and prompt/evaluation inputs. |
| [`models/`](models/README.md) | 📤 Downloaded model assets / extracted adapters. |
| [`outputs/`](outputs/README.md) | 📤 Predictions, metrics, and reports per run. |

---

⬆️ Back to the [repository root](../README.md)
