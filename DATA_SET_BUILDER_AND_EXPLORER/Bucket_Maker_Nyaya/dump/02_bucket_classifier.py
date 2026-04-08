"""
02_bucket_classifier.py
=======================
Classifies each case JSON into one of 9 legal thematic buckets.

Strategy:
  Phase 1 — Fast keyword/regex heuristic  (~50% of cases, CPU, instant)
  Phase 2 — GPU NLI batched inference     (remaining ~50%, multi-GPU DataParallel)

Model: cross-encoder/nli-deberta-v3-large
  Zero-shot NLI classifier. For each case, runs 8 (text, hypothesis) pairs
  through the model in one large batched forward pass.
  DataParallel spreads work across all available GPUs automatically.

Resume-safe: skips any JSON that already has a "bucket" key.
"""

import json
import os
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "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")
NLI_MODEL      = "cross-encoder/nli-deberta-v3-large"
TEXT_LIMIT     = 512    # tokens (model max); truncated by tokenizer
BATCH_SIZE     = 256    # pairs per GPU forward pass
NUM_WORKERS    = 4      # DataLoader workers
MAX_GPUS       = 2      # cap GPU usage — set to None to use all available

CASE_JSONS_DIR = Path(__file__).parent / "output" / "case_jsons"
OUTPUT_DIR     = Path(__file__).parent / "output"
INDEX_PATH     = OUTPUT_DIR / "bucket_index.json"
STATS_PATH     = OUTPUT_DIR / "bucket_stats.json"

# ── Bucket Definitions ────────────────────────────────────────────────────────
BUCKETS = {
    1: {
        "name":  "Violent & Serious Crimes Against Persons",
        "emoji": "🔴",
        "hypothesis": "This case involves murder, assault, rape, kidnapping, robbery, or grievous hurt.",
        "keywords": [
            r"\bmurder\b", r"\bculpable homicide\b", r"\bgrievous hurt\b",
            r"\bassault\b", r"\battempt to murder\b", r"\brape\b",
            r"\bkidnapping\b", r"\babduction\b", r"\bdacoity\b", r"\brobbery\b",
            r"\bipc.{0,10}302\b", r"\bipc.{0,10}307\b", r"\bipc.{0,10}326\b",
            r"\bipc.{0,10}376\b", r"\bipc.{0,10}363\b",
            r"\bsection 302\b", r"\bsection 307\b", r"\bsection 376\b",
            r"\bhomicide\b",
        ],
    },
    2: {
        "name":  "Motor Accident & Negligence (Compensation)",
        "emoji": "🚗",
        "hypothesis": "This case involves a motor vehicle accident, compensation claim, rash driving, or insurance dispute.",
        "keywords": [
            r"\bmotor accident\b", r"\bhit and run\b", r"\brash driving\b",
            r"\bMAC[Tt]\b", r"\bmotor vehicles act\b",
            r"\baccident.{0,40}compensation\b", r"\bcompensation.{0,40}accident\b",
            r"\b304[Aa]\b",
        ],
    },
    3: {
        "name":  "Financial, Economic & Property Crimes",
        "emoji": "💰",
        "hypothesis": "This case involves cheating, fraud, forgery, corruption, bribery, embezzlement, or financial crime.",
        "keywords": [
            r"\bcheating\b", r"\bfraud\b", r"\bforgery\b", r"\bmisappropriation\b",
            r"\bcorruption\b", r"\bbribery\b", r"\bdishonest\b", r"\bembezzlement\b",
            r"\bextortion\b",
            r"\bipc.{0,10}420\b", r"\bipc.{0,10}406\b", r"\bipc.{0,10}409\b",
            r"\bipc.{0,10}468\b", r"\bsection 420\b",
            r"\bprevention of corruption\b", r"\bmoney laundering\b",
            r"\benforcement directorate\b",
        ],
    },
    4: {
        "name":  "Family, Matrimonial & Personal Law",
        "emoji": "👨‍👩‍👧",
        "hypothesis": "This case involves divorce, maintenance, child custody, dowry, domestic violence, or matrimonial dispute.",
        "keywords": [
            r"\bdivorce\b", r"\bmaintenance\b", r"\bcustody\b", r"\bdowry\b",
            r"\bdomestic violence\b", r"\bmatrimonial\b",
            r"\balimony\b", r"\bguardianship\b", r"\b498[Aa]\b",
            r"\bhindu marriage act\b", r"\bdv act\b",
            r"\bcrpc.{0,10}125\b", r"\bsection 125\b",
            r"\bwedlock\b", r"\bseparation\b",
        ],
    },
    5: {
        "name":  "Bail, Custody & Procedural Matters",
        "emoji": "🏛️",
        "hypothesis": "This case is primarily about bail, anticipatory bail, remand, habeas corpus, quashing an FIR, or criminal procedure.",
        "keywords": [
            r"\banticipatory bail\b", r"\bbail application\b", r"\bhabeas corpus\b",
            r"\bquashing.{0,20}FIR\b", r"\bFIR.{0,30}quash\b",
            r"\bremand\b", r"\bchargesheet\b", r"\bpre.arrest bail\b",
        ],
    },
    6: {
        "name":  "Land, Property & Revenue Disputes",
        "emoji": "🏗️",
        "hypothesis": "This case involves land acquisition, tenancy, eviction, property title, encroachment, or revenue disputes.",
        "keywords": [
            r"\bland acquisition\b", r"\btenancy\b",
            r"\bencroachment\b", r"\bmutation\b",
            r"\btitle.{0,20}land\b", r"\beviction\b",
            r"\bplotted land\b", r"\bagricultural land\b",
            r"\bpattas?\b", r"\bkhata\b", r"\bkhasra\b", r"\bsurvey number\b",
            r"\btransfer of property act\b",
        ],
    },
    7: {
        "name":  "Service, Employment & Constitutional Matters",
        "emoji": "🏢",
        "hypothesis": "This case involves government service, termination of employment, promotion, pension, writ petition, or constitutional rights.",
        "keywords": [
            r"\bservice.{0,20}termination\b", r"\btermination.{0,20}service\b",
            r"\bpension\b", r"\bgovernment employee\b", r"\bfundamental rights?\b",
            r"\breservation\b", r"\barticle 226\b", r"\barticle 32\b",
            r"\badministrative tribunal\b",
            r"\bdismissal.{0,20}service\b", r"\bretrenchment\b", r"\bseniority\b",
        ],
    },
    8: {
        "name":  "Narcotics, Arms & Special Statutes",
        "emoji": "💊",
        "hypothesis": "This case involves narcotics, NDPS, arms, explosives, POCSO, or the SC/ST Prevention of Atrocities Act.",
        "keywords": [
            r"\bNDPS\b", r"\bnarcotic\b", r"\bcontraband\b",
            r"\bPOCSO\b", r"\bsc/st\b", r"\batrocities act\b",
            r"\bndps act\b", r"\barms act\b",
            r"\bprevention of atrocities\b",
            r"\bsmuggling\b", r"\bUAPA\b", r"\bPOTA\b",
        ],
    },
    9: {
        "name":  "Other",
        "emoji": "📦",
        "hypothesis": "",
        "keywords": [],
    },
}

CLASSIFIABLE_BUCKETS = [bid for bid in sorted(BUCKETS) if bid != 9]
HYPOTHESES = [BUCKETS[bid]["hypothesis"] for bid in CLASSIFIABLE_BUCKETS]
N_LABELS   = len(CLASSIFIABLE_BUCKETS)  # 8

# Pre-compile keyword patterns
COMPILED_PATTERNS = {
    bid: [re.compile(kw, re.IGNORECASE) for kw in BUCKETS[bid]["keywords"]]
    for bid in CLASSIFIABLE_BUCKETS
}


# ── Keyword Classifier ────────────────────────────────────────────────────────
def keyword_classify(text: str) -> int | None:
    scores = defaultdict(int)
    for bid, patterns in COMPILED_PATTERNS.items():
        for pat in patterns:
            if pat.search(text):
                scores[bid] += 1
    if not scores:
        return None
    return max(scores, key=lambda k: (scores[k], -k))


# ── NLI Dataset (pairs: one text × all 8 hypotheses) ─────────────────────────
class NLIDataset(Dataset):
    """
    Each item = one (text_snippet, hypothesis) pair.
    For N texts and 8 hypotheses, len = N * 8.
    We retrieve results grouped by text after inference.
    """
    def __init__(self, texts: list[str], hypotheses: list[str]):
        self.pairs = [
            (text[:2000], hyp)
            for text in texts
            for hyp in hypotheses
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(tokenizer):
    def _collate(batch):
        premises, hypotheses = zip(*batch)
        return tokenizer(
            list(premises),
            list(hypotheses),
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
    return _collate


# ── Multi-GPU NLI Inference ───────────────────────────────────────────────────
def run_nli_gpu(texts: list[str]) -> list[int]:
    """
    Runs NLI inference on all texts using all available GPUs via DataParallel.
    Returns list of bucket IDs (1-8 or 9) for each text.
    """
    n_gpus_avail = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus = min(n_gpus_avail, MAX_GPUS) if MAX_GPUS else n_gpus_avail
    # Pin to first n_gpus devices
    gpu_ids = list(range(n_gpus))
    device  = torch.device(f"cuda:{gpu_ids[0]}" if n_gpus > 0 else "cpu")

    print(f"\n[02] Loading NLI model: {NLI_MODEL}")
    print(f"[02] GPUs available: {n_gpus_avail} — using: {n_gpus} {gpu_ids}")

    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL, token=HF_TOKEN
    )

    if n_gpus > 1:
        print(f"[02] Wrapping model with DataParallel on GPUs {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model = model.to(device)
    model.eval()
    print(f"[02] Model loaded on {device} ✓\n")

    dataset = NLIDataset(texts, HYPOTHESES)
    # With DataParallel, effective batch per GPU = BATCH_SIZE / n_gpus
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE * max(n_gpus, 1),
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn(tokenizer),
        pin_memory=True,
    )

    # NLI label mapping: ENTAILMENT index
    # DeBERTa NLI: labels are [contradiction, neutral, entailment] → index 2
    ENTAIL_IDX = 2

    all_entail_scores = []  # shape: [N_texts * N_hypotheses]

    with torch.no_grad():
        for batch in tqdm(loader, desc="  GPU NLI batches"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits          # (B, 3)
            probs  = torch.softmax(logits, dim=-1)  # (B, 3)
            entail = probs[:, ENTAIL_IDX]           # (B,)
            all_entail_scores.extend(entail.cpu().tolist())

    # Reshape: [N_texts, N_hypotheses]
    n_texts = len(texts)
    assert len(all_entail_scores) == n_texts * N_LABELS, \
        f"Score count mismatch: {len(all_entail_scores)} vs {n_texts * N_LABELS}"

    bucket_ids = []
    for i in range(n_texts):
        scores = all_entail_scores[i * N_LABELS : (i + 1) * N_LABELS]
        best_idx   = max(range(N_LABELS), key=lambda j: scores[j])
        best_score = scores[best_idx]
        if best_score < 0.20:
            bucket_ids.append(9)   # low confidence → Other
        else:
            bucket_ids.append(CLASSIFIABLE_BUCKETS[best_idx])

    return bucket_ids


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not CASE_JSONS_DIR.exists():
        print(f"ERROR: {CASE_JSONS_DIR} not found. Run 01_csv_to_json.py first.")
        sys.exit(1)

    json_files = sorted(CASE_JSONS_DIR.glob("*.json"))
    print(f"[02] Found {len(json_files):,} case JSON files")

    # Load existing index
    index: dict[str, int] = {}
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            index = json.load(f)
        print(f"[02] Resuming — {len(index):,} already in index")

    bucket_counter: Counter = Counter()
    method_counter: Counter = Counter()

    # ── Phase 1: Keyword pass ────────────────────────────────────────────────
    print(f"\n[02] Phase 1 — Keyword heuristic pass...")
    needs_llm_paths: list[Path] = []
    needs_llm_texts: list[str]  = []
    needs_llm_names: list[str]  = []

    for json_path in tqdm(json_files, desc="  Keyword pass", unit="case"):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"\n  WARN: {json_path.name}: {e}", file=sys.stderr)
            continue

        filename = data.get("filename", json_path.stem)

        if "bucket" in data:
            bid = data["bucket"]
            index[filename] = bid
            bucket_counter[bid] += 1
            method_counter["cached"] += 1
            continue

        text = data.get("text", "")
        bid  = keyword_classify(text)

        if bid is not None:
            data.update({
                "bucket": bid,
                "bucket_name": BUCKETS[bid]["name"],
                "bucket_emoji": BUCKETS[bid]["emoji"],
                "classification_method": "keyword",
            })
            with open(json_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            index[filename] = bid
            bucket_counter[bid] += 1
            method_counter["keyword"] += 1
        else:
            needs_llm_paths.append(json_path)
            needs_llm_texts.append(text)
            needs_llm_names.append(filename)

    print(f"  Keyword classified: {method_counter['keyword']:,}")
    print(f"  Cached (prior run): {method_counter['cached']:,}")
    print(f"  Needs GPU NLI:      {len(needs_llm_names):,}")

    # Checkpoint after Phase 1
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    # ── Phase 2: Batched GPU NLI ─────────────────────────────────────────────
    if needs_llm_names:
        print(f"\n[02] Phase 2 — Batched GPU NLI ({len(needs_llm_names):,} cases)...")

        bucket_ids = run_nli_gpu(needs_llm_texts)

        print(f"\n[02] Writing NLI results to JSONs...")
        for path, filename, bid in zip(
            tqdm(needs_llm_paths, desc="  Writing", unit="case"),
            needs_llm_names,
            bucket_ids,
        ):
            try:
                with open(path) as f:
                    data = json.load(f)
                data.update({
                    "bucket": bid,
                    "bucket_name":  BUCKETS[bid]["name"],
                    "bucket_emoji": BUCKETS[bid]["emoji"],
                    "classification_method": "nli_gpu",
                })
                with open(path, "w") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"\n  WARN: failed to write {path.name}: {e}", file=sys.stderr)

            index[filename] = bid
            bucket_counter[bid] += 1
            method_counter["nli_gpu"] += 1

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)

    stats = {
        "total_classified": len(index),
        "by_method": dict(method_counter),
        "by_bucket": {
            str(bid): {
                "count": bucket_counter[bid],
                "name":  BUCKETS[bid]["name"],
                "emoji": BUCKETS[bid]["emoji"],
            }
            for bid in sorted(BUCKETS)
        },
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[02] ═══ Complete! ═══")
    print(f"     Keyword:  {method_counter['keyword']:,}")
    print(f"     GPU NLI:  {method_counter.get('nli_gpu', 0):,}")
    print(f"     Cached:   {method_counter['cached']:,}")
    print(f"     Total:    {len(index):,}\n")
    print(f"  {'Bucket':<44} {'Count':>7}")
    print(f"  {'─'*52}")
    for bid in sorted(BUCKETS):
        b = BUCKETS[bid]
        print(f"  {b['emoji']} {b['name']:<40} {bucket_counter[bid]:>7,}")
    print(f"\n  Index → {INDEX_PATH}")
    print(f"  Stats → {STATS_PATH}")


if __name__ == "__main__":
    main()
