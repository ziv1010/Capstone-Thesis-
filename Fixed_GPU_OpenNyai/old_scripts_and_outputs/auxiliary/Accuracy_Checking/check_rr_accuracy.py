"""
Rhetorical Role (RR) Accuracy Evaluation for OpenNyAI's InRhetoricalRoles model.

Dataset: opennyaiorg/InRhetoricalRoles  (HuggingFace — requires HF login)
  - 50 test documents from Indian court judgments (train: 247, dev: 30)
  - The exact benchmark dataset used to evaluate OpenNyAI's RR model
  - Reference: Bhatt et al. (2019) + OpenNyAI extensions

The parquet file is loaded directly from the HuggingFace cache
(bypasses a datasets-library version incompatibility in this environment).

Rhetorical Role labels:
  PREAMBLE, FAC, ISSUE, ARG_PETITIONER, ARG_RESPONDENT,
  ANALYSIS, PRE_RELIED, PRE_NOT_RELIED, STA, RLC, RPC, RATIO, NONE

Metrics: sentence-level accuracy, per-label Precision/Recall/F1,
         weighted F1, macro F1, confusion matrix.

Usage:
  # Login first (one-time):
  micromamba run -n fixed_gpu_opennyai_final huggingface-cli login --token <YOUR_TOKEN>

  micromamba run -n fixed_gpu_opennyai_final \\
      python Accuracy_Checking/check_rr_accuracy.py --use_gpu

  micromamba run -n fixed_gpu_opennyai_final \\
      python Accuracy_Checking/check_rr_accuracy.py --use_gpu \\
      --output_file Accuracy_Checking/results/rr_accuracy_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Cache redirect ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_cache   = PROJECT_ROOT / ".cache"
_runtime = PROJECT_ROOT / ".runtime_home"
os.environ.setdefault("HOME",                  str(_runtime))
os.environ.setdefault("HF_HOME",               str(_cache / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",    str(_cache / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME",            str(_cache / "torch"))

# ── Parquet path ──────────────────────────────────────────────────────────────
_HF_CACHE    = Path.home() / ".cache" / "huggingface" / "hub"
_SNAPSHOT    = "datasets--opennyaiorg--InRhetoricalRoles"
_PARQUET_GLOB = "data/test-*.parquet"

RR_LABELS = {
    "PREAMBLE", "FAC", "ISSUE", "ARG_PETITIONER", "ARG_RESPONDENT",
    "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO", "NONE",
}


def _find_test_parquet() -> Path:
    candidates = list((_HF_CACHE / _SNAPSHOT).glob(f"snapshots/*/{_PARQUET_GLOB}"))
    if candidates:
        return candidates[0]
    local_hub = _cache / "huggingface" / "hub"
    candidates = list((local_hub / _SNAPSHOT).glob(f"snapshots/*/{_PARQUET_GLOB}"))
    if candidates:
        return candidates[0]
    return None


# ── Data loading ──────────────────────────────────────────────────────────────

def load_gold_data(max_docs: int) -> List[Dict[str, Any]]:
    """Load InRhetoricalRoles test split from the locally cached parquet file.

    Each row format (Label Studio export):
      data        : {"text": <full document text>}
      annotations : [{"result": [{"value": {"start":int, "end":int,
                                            "text":str, "labels":[role]}}]}]

    Each annotation result item is one sentence span with its rhetorical role.
    """
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas not found. Install with: pip install pandas pyarrow")
        sys.exit(1)

    parquet_path = _find_test_parquet()
    if parquet_path is None:
        print("ERROR: InRhetoricalRoles parquet file not found in HuggingFace cache.")
        print("       Run:  micromamba run -n fixed_gpu_opennyai_final huggingface-cli login --token <TOKEN>")
        print("       Then: micromamba run -n fixed_gpu_opennyai_final python -c \"")
        print("               from huggingface_hub import snapshot_download")
        print("               snapshot_download('opennyaiorg/InRhetoricalRoles', repo_type='dataset')\"")
        sys.exit(1)

    print(f"\nLoading InRhetoricalRoles test split from:\n  {parquet_path}")
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

        sentences:   List[str] = []
        gold_labels: List[str] = []

        for ann_group in (row.get("annotations") or []):
            result = ann_group.get("result", []) if isinstance(ann_group, dict) else []
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = []
            for item in result:
                value = item.get("value", {}) if isinstance(item, dict) else {}
                sent_text = str(value.get("text", "")).strip()
                labels    = list(value.get("labels", []))
                if not sent_text or not labels:
                    continue
                role = str(labels[0]).upper()
                if role not in RR_LABELS:
                    role = "NONE"
                sentences.append(sent_text)
                gold_labels.append(role)

        if not sentences:
            skipped += 1
            continue

        docs.append({
            "id":          str(row.get("meta", {}).get("group", len(docs))),
            "text":        text,
            "sentences":   sentences,
            "gold_labels": gold_labels,
        })

    if skipped:
        print(f"Skipped {skipped} documents with no annotations.")
    total_sents = sum(len(d["sentences"]) for d in docs)
    print(f"Loaded {len(docs)} documents with {total_sents:,} annotated sentences.\n")
    return docs


# ── RR pipeline ───────────────────────────────────────────────────────────────

def load_pipeline(use_gpu: bool):
    try:
        from opennyai.utils.preprocess import Data
    except ImportError:
        try:
            import opennyai
            Data = getattr(opennyai, "Data", None)
        except ImportError:
            Data = None
    try:
        from opennyai import Pipeline
    except ImportError:
        try:
            from opennyai.pipeline import Pipeline
        except ImportError:
            Pipeline = None

    if Data is None or Pipeline is None:
        print("ERROR: Cannot import OpenNyAI Data/Pipeline. Run inside fixed_gpu_opennyai_final.")
        sys.exit(1)

    print(f"Loading OpenNyAI Pipeline  (Rhetorical_Role, use_gpu={use_gpu}) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = Pipeline(components=["Rhetorical_Role"], use_gpu=use_gpu, verbose=False)
    print("Pipeline ready.\n")
    return Data, pipeline


def predict_roles(Data, pipeline, text: str, doc_id: str, use_gpu: bool) -> List[Tuple[str, str]]:
    """Run RR pipeline on full document text. Returns (sentence_text, label) pairs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_obj = Data(
            input_text=[text],
            preprocessing_nlp_model="en_core_web_trf",
            use_gpu=use_gpu,
            verbose=False,
            file_ids=[doc_id],
        )
        result = pipeline([data_obj[0]])

    raw_doc = result[0] if isinstance(result, list) else result
    predictions: List[Tuple[str, str]] = []
    for ann in raw_doc.get("annotations", []):
        labels = ann.get("labels", [])
        role   = str(labels[0]).upper() if labels else "NONE"
        if role not in RR_LABELS:
            role = "NONE"
        predictions.append((ann.get("text", "").strip(), role))
    return predictions


# ── Sentence matching ─────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words and not b_words:
        return 1.0
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def match_predictions_to_gold(
    pred_pairs: List[Tuple[str, str]],
    gold_sentences: List[str],
    gold_labels: List[str],
) -> Tuple[List[str], List[str]]:
    """Align predicted labels to gold labels by word-level Jaccard similarity.

    The pipeline may segment sentences slightly differently between runs, so
    we match by text similarity rather than position alone.
    """
    if len(pred_pairs) == len(gold_sentences):
        return gold_labels, [p[1] for p in pred_pairs]

    pred_labels = [p[1] for p in pred_pairs]
    pred_texts  = [p[0] for p in pred_pairs]
    matched_gold: List[str] = []
    matched_pred: List[str] = []
    used: set = set()

    for g_text, g_label in zip(gold_sentences, gold_labels):
        best_idx, best_score = -1, -1.0
        for i, p_text in enumerate(pred_texts):
            if i in used:
                continue
            score = _jaccard(g_text, p_text)
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx >= 0 and best_score >= 0.3:
            matched_gold.append(g_label)
            matched_pred.append(pred_labels[best_idx])
            used.add(best_idx)

    return matched_gold, matched_pred


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_gold: List[str], all_pred: List[str]) -> Dict[str, Any]:
    assert len(all_gold) == len(all_pred)
    total   = len(all_gold)
    correct = sum(g == p for g, p in zip(all_gold, all_pred))
    gold_counts = Counter(all_gold)
    labels_present = sorted(set(all_gold) | set(all_pred))

    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    for g, p in zip(all_gold, all_pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    by_label: Dict[str, Any] = {}
    macro_p = macro_r = macro_f1 = weighted_f1 = 0.0
    for label in labels_present:
        p  = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        r  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        by_label[label] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "support":   gold_counts[label],
            "tp": tp[label], "fp": fp[label], "fn": fn[label],
        }
        macro_p += p; macro_r += r; macro_f1 += f1
        weighted_f1 += f1 * gold_counts[label]

    n = max(len(labels_present), 1)
    return {
        "accuracy":    round(correct / total, 4) if total else 0.0,
        "correct":     correct,
        "total":       total,
        "by_label":    by_label,
        "macro":  {"precision": round(macro_p/n,4), "recall": round(macro_r/n,4), "f1": round(macro_f1/n,4)},
        "weighted_f1": round(weighted_f1 / total, 4) if total else 0.0,
    }


def build_confusion_matrix(all_gold: List[str], all_pred: List[str]) -> Dict[str, Any]:
    labels = sorted(set(all_gold) | set(all_pred))
    matrix: Dict[str, Dict[str, int]] = {l: defaultdict(int) for l in labels}
    for g, p in zip(all_gold, all_pred):
        matrix[g][p] += 1
    return {"labels": labels, "matrix": {k: dict(v) for k, v in matrix.items()}}


# ── Display ───────────────────────────────────────────────────────────────────

def print_results(results: Dict[str, Any], confusion: Dict[str, Any], n_docs: int) -> None:
    print("\n" + "=" * 72)
    print("  OpenNyAI RHETORICAL ROLE ACCURACY  —  InRhetoricalRoles")
    print("  Dataset  : opennyaiorg/InRhetoricalRoles  (test split, 50 documents)")
    print("  Metric   : Sentence-level exact label match")
    print(f"  Evaluated: {n_docs} documents  |  {results['total']:,} matched sentences")
    print("=" * 72)
    print(f"  Accuracy     : {results['accuracy']:.4f}  "
          f"({results['correct']}/{results['total']} sentences correct)")
    print(f"  Weighted F1  : {results['weighted_f1']:.4f}")
    print(f"  Macro F1     : {results['macro']['f1']:.4f}")
    print()
    print(f"  {'Label':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}")
    print("  " + "-" * 65)
    for label in sorted(results["by_label"]):
        m = results["by_label"][label]
        print(f"  {label:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['support']:>9}")
    print("  " + "-" * 65)
    ma = results["macro"]
    print(f"  {'MACRO AVERAGE':<22} {ma['precision']:>10.4f} {ma['recall']:>10.4f} "
          f"{ma['f1']:>10.4f} {results['total']:>9}")
    print("=" * 72)

    labels = confusion["labels"]
    if len(labels) <= 15:
        print("\n  Confusion Matrix  (rows = gold, cols = predicted)")
        col_w = 7
        print("  " + " " * 22 + "".join(f"{l[:col_w]:>{col_w}}" for l in labels))
        print("  " + " " * 22 + "-" * (col_w * len(labels)))
        for gold_label in labels:
            row = [confusion["matrix"].get(gold_label, {}).get(p, 0) for p in labels]
            print(f"  {gold_label:<22}" + "".join(f"{v:>{col_w}}" for v in row))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate InRhetoricalRoles on the full opennyaiorg/InRhetoricalRoles test set (50 docs)."
    )
    p.add_argument("--use_gpu",     action="store_true", help="Use GPU for inference.")
    p.add_argument("--max_docs",    type=int, default=0,
                   help="Limit number of documents (0 = all 50).")
    p.add_argument("--output_file", default="",          help="Path to save JSON results.")
    p.add_argument("--verbose",     action="store_true", help="Print per-sentence details.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    docs           = load_gold_data(args.max_docs)
    Data, pipeline = load_pipeline(args.use_gpu)

    all_gold: List[str] = []
    all_pred: List[str] = []

    for idx, doc in enumerate(docs, start=1):
        print(f"  [{idx}/{len(docs)}] doc={doc['id']}  ({len(doc['sentences'])} gold sentences) ...")
        pred_pairs = predict_roles(Data, pipeline, doc["text"], str(idx), args.use_gpu)
        matched_gold, matched_pred = match_predictions_to_gold(
            pred_pairs, doc["sentences"], doc["gold_labels"]
        )
        all_gold.extend(matched_gold)
        all_pred.extend(matched_pred)

        if args.verbose:
            for sent, gold, pred in zip(doc["sentences"], matched_gold, matched_pred):
                marker = "✓" if gold == pred else "✗"
                print(f"    {marker} gold={gold:<18} pred={pred:<18}  '{sent[:55]}'")

    if not all_gold:
        print("ERROR: No sentences to evaluate.")
        sys.exit(1)

    results   = compute_metrics(all_gold, all_pred)
    confusion = build_confusion_matrix(all_gold, all_pred)
    results["metadata"] = {
        "model":                  "InRhetoricalRoles (opennyaiorg/InRhetoricalRoles)",
        "dataset":                "opennyaiorg/InRhetoricalRoles",
        "split":                  "test",
        "total_docs_in_split":    50,
        "documents_evaluated":    len(docs),
        "sentences_evaluated":    len(all_gold),
        "use_gpu":                args.use_gpu,
    }

    print_results(results, confusion, len(docs))

    output = {"results": results, "confusion_matrix": confusion}
    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2))
        print(f"Results saved to: {out}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
