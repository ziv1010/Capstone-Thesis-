#!/usr/bin/env python3
"""
Binary classifier: is each case in the financial_fraud folder actually
a financial fraud case?

Uses Mistral-7B-Instruct-v0.3 via vLLM on GPUs 6 & 7.
Outputs a results CSV and prints the final count.
"""

import os
import csv
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN             = "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"
MODEL_ID             = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_TEXT_CHARS       = 2500   # truncate to keep prompts fast
BATCH_SIZE           = 128
TENSOR_PARALLEL_SIZE = 2      # GPUs 6 & 7
CHECKPOINT_EVERY     = 10     # flush results every N batches

DATA_DIR   = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/DATA_SET_BUILDER_AND_EXPLORER/Bucket_testing/ff_pipeline/txts"
)
OUTPUT_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/DATA_SET_BUILDER_AND_EXPLORER/Bucket_testing/ff_pipeline"
)
OUT_CSV    = OUTPUT_DIR / "financial_fraud_classification_results.csv"
CKPT_CSV   = OUTPUT_DIR / "financial_fraud_classification_results.ckpt.csv"

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a legal case classifier for Indian courts.\n"
    "Your task: decide whether the given case text belongs to the "
    "FINANCIAL FRAUD bucket.\n\n"
    "The financial_fraud bucket covers:\n"
    "  - cheating, fraud, misappropriation\n"
    "  - money laundering\n"
    "  - corruption, bribery\n"
    "  - bank fraud, Ponzi schemes, financial crimes\n\n"
    "Respond with ONLY one word: YES or NO.\n"
    "Do not add any explanation, punctuation, or extra text."
)


def make_prompt(text: str, tokenizer) -> str:
    truncated = text.strip()[:MAX_TEXT_CHARS]
    messages = [
        {
            "role": "user",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Case text:\n{truncated}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_answer(raw: str) -> bool:
    cleaned = raw.strip().upper()
    if cleaned.startswith("YES"):
        return True
    if cleaned.startswith("NO"):
        return False
    # Fallback: keyword scan
    lower = raw.lower()
    for kw in ("fraud", "cheating", "misappropriation", "laundering",
               "corruption", "financial crime", "bribery"):
        if kw in lower:
            return True
    return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.environ["HF_TOKEN"]               = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["TRANSFORMERS_CACHE"]     = str(OUTPUT_DIR / ".hf_cache")
    os.environ["HF_HOME"]                = str(OUTPUT_DIR / ".hf_cache")
    os.environ["CUDA_VISIBLE_DEVICES"]   = "6,7"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # ── Collect .txt files ────────────────────────────────────────────────────
    txt_files = sorted(DATA_DIR.glob("*.txt"))
    print(f"Found {len(txt_files):,} .txt files in {DATA_DIR}")
    if not txt_files:
        print("No .txt files found — run pdf_to_txt.py first.")
        return

    # ── Resume from checkpoint ────────────────────────────────────────────────
    done_names: set[str] = set()
    results: list[dict]  = []
    if CKPT_CSV.exists():
        with open(CKPT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                done_names.add(row["filename"])
        print(f"Resuming from checkpoint: {len(results):,} already classified")

    remaining = [f for f in txt_files if f.name not in done_names]
    print(f"Remaining to classify: {len(remaining):,}")

    if not remaining:
        print("Nothing left to classify.")
    else:
        # ── Load model ────────────────────────────────────────────────────────
        print("Loading tokenizer …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        print(f"Loading {MODEL_ID} across {TENSOR_PARALLEL_SIZE} GPUs (6 & 7) …")
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=0.90,
            max_model_len=3072,
            trust_remote_code=True,
            dtype="bfloat16",
            disable_custom_all_reduce=True,
        )

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=8,
            stop=["</s>", "[INST"],
        )

        # ── Batch inference ───────────────────────────────────────────────────
        texts   = [f.read_text(encoding="utf-8", errors="replace") for f in remaining]
        prompts = [make_prompt(t, tokenizer) for t in texts]

        total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Running inference (batch_size={BATCH_SIZE}) …")

        for batch_idx in tqdm(range(total_batches), unit="batch"):
            start  = batch_idx * BATCH_SIZE
            batch  = prompts[start : start + BATCH_SIZE]
            files  = remaining[start : start + BATCH_SIZE]

            outputs = llm.generate(batch, sampling_params)
            for out, fpath in zip(outputs, files):
                raw   = out.outputs[0].text
                label = parse_answer(raw)
                results.append({
                    "filename":             fpath.name,
                    "is_financial_fraud":   label,
                    "raw_llm_output":       raw.strip(),
                })

            # Checkpoint flush
            if (batch_idx + 1) % CHECKPOINT_EVERY == 0 or (batch_idx + 1) == total_batches:
                with open(CKPT_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["filename", "is_financial_fraud", "raw_llm_output"]
                    )
                    writer.writeheader()
                    writer.writerows(results)

    # ── Write final CSV ───────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "is_financial_fraud", "raw_llm_output"]
        )
        writer.writeheader()
        writer.writerows(results)

    CKPT_CSV.unlink(missing_ok=True)
    print(f"\nResults saved → {OUT_CSV}")

    # ── Final counts ──────────────────────────────────────────────────────────
    true_count  = sum(1 for r in results if str(r["is_financial_fraud"]).lower() in ("true", "yes", "1"))
    false_count = len(results) - true_count

    print("\n" + "=" * 50)
    print(f"  Total cases classified  : {len(results):>6,}")
    print(f"  IS financial fraud      : {true_count:>6,}")
    print(f"  NOT financial fraud     : {false_count:>6,}")
    print(f"  Purity (%)              : {true_count / len(results) * 100:>6.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
