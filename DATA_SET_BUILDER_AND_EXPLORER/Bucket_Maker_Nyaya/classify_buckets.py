#!/usr/bin/env python3
"""
Bucket classifier for Indian legal case texts using Mistral-7B-Instruct.
Uses vLLM with tensor parallelism across all available GPUs for maximum throughput.

Buckets:
  1. criminal_violent      — murder, culpable homicide, robbery, dacoity
  2. sexual_offences       — rape, sexual assault, POCSO, molestation
  3. motor_accidents       — motor accident, MACT, compensation, rash driving
  4. financial_fraud       — cheating, fraud, misappropriation, money laundering, corruption
  5. family_matrimonial    — divorce, maintenance, custody, domestic violence, dowry, cruelty
  6. land_property         — land acquisition, possession, encroachment, tenancy
  7. food_safety           — food safety, FSSAI, adulteration, PFA Act, Essential Commodities
  8. narcotics_special_acts — NDPS, narcotic, contraband, recovery
  9. other                 — anything that does not fit the above
"""

import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN             = "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"
MODEL_ID             = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_TEXT_CHARS       = 2500   # truncate long texts — enough signal, keeps prompts fast
BATCH_SIZE           = 256    # vLLM queues these at once; tune up if VRAM allows
TENSOR_PARALLEL_SIZE = 8      # use all 8 GPUs (32 heads / 8 = 4, divisible)
CHECKPOINT_EVERY     = 10     # flush partial results to CSV every N batches

BASE_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/DATA_SET_BUILDER_AND_EXPLORER/Bucket_Maker"
)
INPUT_FILES = {
    "test": BASE_DIR / "current/CJPE_ext_SCI_HCs_Tribunals_daily_orders_test.csv",
    "dev":  BASE_DIR / "current/CJPE_ext_SCI_HCs_Tribunals_daily_orders_dev.csv",
}
OUTPUT_DIR = BASE_DIR / "bucketed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Bucket labels ─────────────────────────────────────────────────────────────
VALID_BUCKETS = [
    "criminal_violent",
    "sexual_offences",
    "motor_accidents",
    "financial_fraud",
    "family_matrimonial",
    "land_property",
    "food_safety",
    "narcotics_special_acts",
    "other",
]

# Keyword fallback map (used when the model output doesn't exactly match a label)
KEYWORD_MAP = {
    "criminal":     "criminal_violent",
    "violent":      "criminal_violent",
    "murder":       "criminal_violent",
    "homicide":     "criminal_violent",
    "robbery":      "criminal_violent",
    "dacoity":      "criminal_violent",
    "sexual":       "sexual_offences",
    "rape":         "sexual_offences",
    "pocso":        "sexual_offences",
    "molestation":  "sexual_offences",
    "assault":      "sexual_offences",
    "motor":        "motor_accidents",
    "mact":         "motor_accidents",
    "accident":     "motor_accidents",
    "rash":         "motor_accidents",
    "fraud":        "financial_fraud",
    "cheating":     "financial_fraud",
    "misappropriation": "financial_fraud",
    "laundering":   "financial_fraud",
    "corruption":   "financial_fraud",
    "financial":    "financial_fraud",
    "family":       "family_matrimonial",
    "matrimonial":  "family_matrimonial",
    "divorce":      "family_matrimonial",
    "maintenance":  "family_matrimonial",
    "custody":      "family_matrimonial",
    "dowry":        "family_matrimonial",
    "domestic":     "family_matrimonial",
    "land":         "land_property",
    "property":     "land_property",
    "tenancy":      "land_property",
    "encroachment": "land_property",
    "acquisition":  "land_property",
    "food":         "food_safety",
    "fssai":        "food_safety",
    "adulteration": "food_safety",
    "narcotic":     "narcotics_special_acts",
    "ndps":         "narcotics_special_acts",
    "contraband":   "narcotics_special_acts",
}

# ── Prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a legal case classifier for Indian courts. "
    "Classify the given case text into exactly ONE of these categories:\n\n"
    "criminal_violent      — murder, culpable homicide, robbery, dacoity\n"
    "sexual_offences       — rape, sexual assault, POCSO, molestation\n"
    "motor_accidents       — motor accident, MACT, compensation, rash driving\n"
    "financial_fraud       — cheating, fraud, misappropriation, money laundering, corruption\n"
    "family_matrimonial    — divorce, maintenance, custody, domestic violence, dowry, cruelty\n"
    "land_property         — land acquisition, possession, encroachment, tenancy\n"
    "food_safety           — food safety, FSSAI, adulteration, PFA Act, Essential Commodities\n"
    "narcotics_special_acts — NDPS, narcotic, contraband, recovery\n"
    "other                 — anything that does not fit any category above\n\n"
    "Respond with ONLY the category name exactly as written above. No explanation, no punctuation."
)


def make_prompt(text: str, tokenizer) -> str:
    truncated = text.strip()[:MAX_TEXT_CHARS]
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nCase text:\n{truncated}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_bucket(raw: str) -> str:
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    # Exact match
    for b in VALID_BUCKETS:
        if cleaned == b or cleaned.startswith(b):
            return b
    # Substring match against valid labels
    for b in VALID_BUCKETS:
        if b in cleaned:
            return b
    # Keyword fallback
    for kw, bucket in KEYWORD_MAP.items():
        if kw in cleaned:
            return bucket
    return "other"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.environ["HF_TOKEN"]                  = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"]    = HF_TOKEN
    os.environ["TRANSFORMERS_CACHE"]        = str(BASE_DIR / ".hf_cache")
    os.environ["HF_HOME"]                   = str(BASE_DIR / ".hf_cache")
    os.environ["CUDA_VISIBLE_DEVICES"]      = "0,1,2,3,4,5,6,7"

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading {MODEL_ID} across {TENSOR_PARALLEL_SIZE} GPUs (physical 4-7) …")
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.90,
        max_model_len=3072,          # keep memory footprint reasonable
        trust_remote_code=True,
        dtype="bfloat16",
        disable_custom_all_reduce=True,  # PCIe topology, suppress warning
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=24,               # category name is short; allow a little extra
        stop=["</s>", "[INST"],      # do NOT stop on \n — model emits newline before label
    )

    for split, input_path in INPUT_FILES.items():
        print(f"\n{'='*60}")
        print(f"Split: {split}  →  {input_path.name}")
        print(f"{'='*60}")

        df = pd.read_csv(input_path)
        print(f"Loaded {len(df):,} rows")

        texts = df["text"].fillna("").astype(str).tolist()
        prompts = [make_prompt(t, tokenizer) for t in texts]

        out_path  = OUTPUT_DIR / f"{input_path.stem}_bucketed.csv"
        ckpt_path = OUTPUT_DIR / f"{input_path.stem}_bucketed.ckpt.csv"

        # Resume from checkpoint if one exists
        start_batch = 0
        all_buckets: list[str] = []
        if ckpt_path.exists():
            ckpt_df = pd.read_csv(ckpt_path)
            all_buckets = ckpt_df["bucket"].tolist()
            start_batch = len(all_buckets) // BATCH_SIZE
            print(f"Resuming from checkpoint: {len(all_buckets):,} rows already done")

        total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Running inference (batch_size={BATCH_SIZE}, checkpoint every {CHECKPOINT_EVERY} batches) …")

        for batch_idx in tqdm(range(start_batch, total_batches), unit="batch",
                              initial=start_batch, total=total_batches):
            start   = batch_idx * BATCH_SIZE
            batch   = prompts[start : start + BATCH_SIZE]
            outputs = llm.generate(batch, sampling_params)
            for out in outputs:
                raw    = out.outputs[0].text
                bucket = parse_bucket(raw)
                all_buckets.append(bucket)

            # Flush checkpoint every N batches
            if (batch_idx + 1) % CHECKPOINT_EVERY == 0 or (batch_idx + 1) == total_batches:
                ckpt_df = df.iloc[:len(all_buckets)].copy()
                ckpt_df["bucket"] = all_buckets
                ckpt_df.to_csv(ckpt_path, index=False)

        # Write final output and remove checkpoint
        df["bucket"] = all_buckets
        df.to_csv(out_path, index=False)
        ckpt_path.unlink(missing_ok=True)
        print(f"\nSaved → {out_path}")

        print("\nBucket distribution:")
        counts = df["bucket"].value_counts()
        total  = len(df)
        for label, cnt in counts.items():
            bar = "#" * int(cnt / total * 40)
            print(f"  {label:<25} {cnt:>6,}  ({cnt/total*100:5.1f}%)  {bar}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
