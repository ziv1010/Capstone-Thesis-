"""
NER Accuracy Evaluation for OpenNyAI's en_legal_ner_trf model.

Dataset: opennyaiorg/InLegalNER  (HuggingFace — requires HF login)
  - 4,501 test documents from Indian court judgments
  - The exact benchmark dataset used to evaluate en_legal_ner_trf
  - Reference: Kalamkar et al. (2022) "Named Entity Recognition in Indian Court Judgments"

The parquet files are loaded directly from the HuggingFace cache
(bypasses a datasets-library version incompatibility in this environment).

Entity labels:
  COURT, PETITIONER, RESPONDENT, JUDGE, LAWYER, DATE, ORG, GPE,
  STATUTE, PROVISION, PRECEDENT, CASE_NUMBER, WITNESS, OTHER_PERSON

Metric: Exact-span match  (start, end, label must all be identical)

Usage:
  # Login first (one-time):
  micromamba run -n fixed_gpu_opennyai_final huggingface-cli login --token <YOUR_TOKEN>

  micromamba run -n fixed_gpu_opennyai_final \\
      python Accuracy_Checking/check_ner_accuracy.py --use_gpu

  # Quick sanity check on 50 docs:
  micromamba run -n fixed_gpu_opennyai_final \\
      python Accuracy_Checking/check_ner_accuracy.py --use_gpu --max_docs 50

  # Full test set with saved results:
  micromamba run -n fixed_gpu_opennyai_final \\
      python Accuracy_Checking/check_ner_accuracy.py --use_gpu \\
      --output_file Accuracy_Checking/results/ner_accuracy_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ── Cache redirect ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_cache = PROJECT_ROOT / ".cache"
os.environ.setdefault("HF_HOME",               str(_cache / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",    str(_cache / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME",            str(_cache / "torch"))

# ── Parquet path (downloaded by HuggingFace hub after login) ─────────────────
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_SNAPSHOT  = "datasets--opennyaiorg--InLegalNER"
_PARQUET_GLOB = "data/test-*.parquet"


def _find_test_parquet() -> Path:
    """Locate the downloaded InLegalNER test parquet file."""
    candidates = list((_HF_CACHE / _SNAPSHOT).glob(f"snapshots/*/{_PARQUET_GLOB}"))
    if candidates:
        return candidates[0]
    # Try the project-local cache too
    local_hub = _cache / "huggingface" / "hub"
    candidates = list((local_hub / _SNAPSHOT).glob(f"snapshots/*/{_PARQUET_GLOB}"))
    if candidates:
        return candidates[0]
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gold_data(max_docs: int) -> List[Dict[str, Any]]:
    """Load the InLegalNER test split from the locally cached parquet file.

    Each row format (Label Studio export):
      id          : document identifier
      data        : {"text": <full document text>}
      annotations : [{"result": [{"value": {"start":int, "end":int,
                                            "text":str, "labels":[str]}}]}]
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not found. Install with: pip install pandas pyarrow")
        sys.exit(1)

    parquet_path = _find_test_parquet()
    if parquet_path is None:
        print("ERROR: InLegalNER parquet file not found in HuggingFace cache.")
        print("       Run the following to download it:")
        print("         micromamba run -n fixed_gpu_opennyai_final huggingface-cli login --token <TOKEN>")
        print("         micromamba run -n fixed_gpu_opennyai_final python -c \"")
        print("           from huggingface_hub import snapshot_download")
        print("           snapshot_download('opennyaiorg/InLegalNER', repo_type='dataset')\"")
        sys.exit(1)

    print(f"\nLoading InLegalNER test split from:\n  {parquet_path}")
    df = pd.read_parquet(parquet_path)
    total = len(df)
    if max_docs and max_docs < total:
        df = df.head(max_docs)
        print(f"Using {max_docs} of {total} test documents (--max_docs limit).")
    else:
        print(f"Using all {total} test documents.")

    docs = []
    skipped = 0
    for _, row in df.iterrows():
        text = row.get("data", {}).get("text", "")
        if not text or not text.strip():
            skipped += 1
            continue

        gold_entities: List[Tuple[int, int, str]] = []
        for ann_group in (row.get("annotations") or []):
            result = ann_group.get("result", []) if isinstance(ann_group, dict) else []
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = []
            for item in result:
                value = item.get("value", {}) if isinstance(item, dict) else {}
                start  = value.get("start")
                end    = value.get("end")
                labels = value.get("labels", [])
                if start is None or end is None or not labels:
                    continue
                gold_entities.append((int(start), int(end), str(labels[0])))

        docs.append({
            "id":            row.get("id", ""),
            "text":          text,
            "gold_entities": gold_entities,
        })

    if skipped:
        print(f"Skipped {skipped} documents with empty text.")
    print(f"Loaded {len(docs)} documents with {sum(len(d['gold_entities']) for d in docs):,} gold entities.\n")
    return docs


# ── NER model ─────────────────────────────────────────────────────────────────

def load_ner_model(use_gpu: bool):
    """Load en_legal_ner_trf directly via spaCy.

    WHY raw spaCy rather than the full OpenNyAI Pipeline:
      The production pipeline internally splits the document into a preamble
      section and a judgment section before running NER, then re-combines them.
      This shifting of text changes character offsets.  The gold labels in
      opennyaiorg/InLegalNER are annotated against the ORIGINAL raw text, so
      exact-span comparison (start, end, label) only works if we run NER on
      that same raw text — which is what spacy.load() does.

    What IS shared with production:
      - Exact same model weights  (en_legal_ner_trf, from opennyaiorg/en_legal_ner_trf)
      - Exact same entity label vocabulary
      - The production pipeline's sentence-level chunking and post-processing
        (coreference resolution) add incremental improvements ON TOP of these
        weights, so real production quality >= the F1 reported here.
    """
    try:
        import spacy
    except ImportError:
        print("ERROR: spaCy not found. Run inside the fixed_gpu_opennyai_final environment.")
        sys.exit(1)

    if use_gpu:
        activated = spacy.prefer_gpu()
        print(f"spaCy GPU: {'activated' if activated else 'not available, using CPU'}")

    model_name = "en_legal_ner_trf"
    if model_name not in spacy.util.get_installed_models():
        print(f"ERROR: spaCy model '{model_name}' is not installed.")
        print("       Run the main pipeline on any file first to auto-install it.")
        sys.exit(1)

    print(f"Loading {model_name} ...")
    nlp = spacy.load(model_name)
    print("Model ready.\n")
    return nlp


def predict_entities(nlp, text: str) -> List[Tuple[int, int, str]]:
    doc = nlp(text)
    return [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    gold_sets: List[Set[Tuple[int, int, str]]],
    pred_sets: List[Set[Tuple[int, int, str]]],
) -> Dict[str, Any]:
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for gold, pred in zip(gold_sets, pred_sets):
        for span in pred:
            (tp if span in gold else fp)[span[2]] += 1
        for span in gold:
            if span not in pred:
                fn[span[2]] += 1

    all_labels = sorted(set(list(tp) + list(fp) + list(fn)))
    by_label: Dict[str, Any] = {}
    macro_p = macro_r = macro_f1 = 0.0
    micro_tp = micro_fp = micro_fn = 0

    for label in all_labels:
        p  = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        by_label[label] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "tp": tp[label], "fp": fp[label], "fn": fn[label],
            "gold_count": tp[label] + fn[label],
            "pred_count": tp[label] + fp[label],
        }
        macro_p  += p;  macro_r  += r;  macro_f1 += f1
        micro_tp += tp[label]; micro_fp += fp[label]; micro_fn += fn[label]

    n = max(len(all_labels), 1)
    mp  = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    mr  = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    mf1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    return {
        "by_label": by_label,
        "macro": {
            "precision": round(macro_p  / n, 4),
            "recall":    round(macro_r  / n, 4),
            "f1":        round(macro_f1 / n, 4),
        },
        "micro": {
            "precision": round(mp,  4),
            "recall":    round(mr,  4),
            "f1":        round(mf1, 4),
            "total_tp":  micro_tp,
            "total_fp":  micro_fp,
            "total_fn":  micro_fn,
        },
    }


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(results: Dict[str, Any], n_docs: int, n_entities: int) -> None:
    print("\n" + "=" * 72)
    print("  OpenNyAI NER ACCURACY  —  en_legal_ner_trf")
    print("  Dataset  : opennyaiorg/InLegalNER  (test split, 4501 documents)")
    print("  Metric   : Exact-span match  (start + end + label)")
    print(f"  Evaluated: {n_docs} documents  |  {n_entities:,} gold entities")
    print("=" * 72)
    print(f"  {'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Gold':>8} {'Pred':>8}")
    print("  " + "-" * 70)
    for label, m in sorted(results["by_label"].items()):
        print(f"  {label:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['gold_count']:>8} {m['pred_count']:>8}")
    print("  " + "-" * 70)
    mi = results["micro"]
    ma = results["macro"]
    print(f"  {'MICRO OVERALL':<22} {mi['precision']:>10.4f} {mi['recall']:>10.4f} "
          f"{mi['f1']:>10.4f} {mi['total_tp']+mi['total_fn']:>8} {mi['total_tp']+mi['total_fp']:>8}")
    print(f"  {'MACRO OVERALL':<22} {ma['precision']:>10.4f} {ma['recall']:>10.4f} {ma['f1']:>10.4f}")
    print("=" * 72 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate en_legal_ner_trf on the full opennyaiorg/InLegalNER test set (4501 docs)."
    )
    p.add_argument("--use_gpu",     action="store_true", help="Use GPU for inference.")
    p.add_argument("--max_docs",    type=int, default=0,
                   help="Limit number of documents (0 = all 4501). Use 50-100 for a quick check.")
    p.add_argument("--output_file", default="",          help="Path to save JSON results.")
    p.add_argument("--verbose",     action="store_true", help="Print per-document details.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    docs = load_gold_data(args.max_docs)
    nlp  = load_ner_model(args.use_gpu)

    gold_sets: List[Set[Tuple[int, int, str]]] = []
    pred_sets: List[Set[Tuple[int, int, str]]] = []
    total_gold = sum(len(d["gold_entities"]) for d in docs)

    print(f"Running NER on {len(docs)} documents ({total_gold:,} gold entities) ...")
    for idx, doc in enumerate(docs, start=1):
        gold = set(doc["gold_entities"])
        pred = set(predict_entities(nlp, doc["text"]))
        gold_sets.append(gold)
        pred_sets.append(pred)

        if args.verbose:
            tp = gold & pred
            fp = pred - gold
            fn = gold - pred
            print(f"  [{idx}/{len(docs)}] {doc['id']}  tp={len(tp)} fp={len(fp)} fn={len(fn)}")
        elif idx % 100 == 0 or idx == len(docs):
            print(f"  {idx}/{len(docs)} documents processed ...")

    results = compute_metrics(gold_sets, pred_sets)
    results["metadata"] = {
        "model":                   "en_legal_ner_trf",
        "note":                    "Raw spaCy inference on original text — same model weights as production. "
                                   "Production adds sentence-level chunking + post-processing on top, so "
                                   "real production NER quality >= these numbers.",
        "dataset":                 "opennyaiorg/InLegalNER",
        "split":                   "test",
        "total_in_split":          4501,
        "documents_evaluated":     len(docs),
        "gold_entities_evaluated": total_gold,
        "use_gpu":                 args.use_gpu,
    }

    print_results(results, len(docs), total_gold)

    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"Results saved to: {out}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
