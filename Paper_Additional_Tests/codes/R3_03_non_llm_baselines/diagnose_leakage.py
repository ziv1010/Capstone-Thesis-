#!/usr/bin/env python3
"""Is the strong TF-IDF result genuine signal, or residual leakage?

Run this after `build_features.py`. On fold 0 it:

  1. fits the headline TF-IDF + Linear SVM and prints its highest-weight n-grams
     per class, so residual outcome vocabulary is visible;
  2. applies the canonical final sanitizer used by every reported TF-IDF model;
  3. asserts the sanitized corpus and fitted vocabulary contain no mask artifact
     or direct operative-outcome term.

The point is that the GNN reads *the same text*, so anything found here is a
property of the shared input rather than of the baseline. Output is written to
`outputs/diagnostic_tfidf_leakage.log` by `run.sh`.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_baselines import LABEL_NAMES, REFERENCE_KFOLD, TFIDF_KWARGS  # noqa: E402
from text_sanitization import (  # noqa: E402
    assert_vocabulary_clean,
    sanitize_document,
)

FEATURES = _HERE / "outputs" / "features"
MASK_TOKEN = "[LEAKAGE_MASK]"


def main() -> None:
    index = json.loads((FEATURES / "case_index.json").read_text())
    case_ids = index["case_ids"]
    y = np.array([LABEL_NAMES.index(v) for v in index["labels"]])
    with gzip.open(FEATURES / "text.jsonl.gz", "rt", encoding="utf-8") as handle:
        texts = [json.loads(line) for line in handle]

    position = {case_id: i for i, case_id in enumerate(case_ids)}
    frame = pd.read_csv(REFERENCE_KFOLD / "fold_00" / "predictions.csv")
    rows = frame["case_id"].map(position).to_numpy()
    split = frame["split"].to_numpy()
    train, test = rows[split == "train"], rows[split == "test"]

    def run(tag: str, documents=None, stop_words=None):
        documents = texts if documents is None else documents
        vectorizer = TfidfVectorizer(**TFIDF_KWARGS, **({"stop_words": stop_words} if stop_words else {}))
        X_train = vectorizer.fit_transform([documents[i] for i in train])
        X_test = vectorizer.transform([documents[i] for i in test])
        model = LinearSVC(C=0.25, class_weight="balanced", dual=True, max_iter=4000, random_state=42)
        model.fit(X_train, y[train])
        prediction = model.predict(X_test)
        print(
            f"{tag:38s} acc={accuracy_score(y[test], prediction):.4f} "
            f"macroF1={f1_score(y[test], prediction, average='macro'):.4f}",
            flush=True,
        )
        return vectorizer, model

    print("=== fold 0, TF-IDF + LinearSVC leakage audit ===\n")
    vectorizer, model = run("1. unsanitized HGT-source text")

    coefficients = np.asarray(model.coef_).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(coefficients)
    print(f"\n  top-25 toward NEGATIVE ({LABEL_NAMES[0]}, dismissed/procedural):")
    print("   ", ", ".join(terms[order[:25]]))
    print(f"\n  top-25 toward POSITIVE ({LABEL_NAMES[1]}, allowed):")
    print("   ", ", ".join(terms[order[-25:][::-1]]))

    mask_features = sorted(
        ((terms[i], float(coefficients[i])) for i in range(len(terms)) if "leakage" in terms[i] or "mask" in terms[i]),
        key=lambda kv: -abs(kv[1]),
    )
    print(f"\n  mask-derived features in vocabulary: {len(mask_features)}")
    print(f"  strongest: {mask_features[:8]}")

    sanitized = [sanitize_document(text) for text in texts]
    print("\nCanonical sanitization audit:")
    full_audit = json.loads((_HERE / "outputs/tfidf_input_audit.json").read_text())
    if full_audit.get("status") != "PASS":
        raise AssertionError("full-corpus TF-IDF input audit has not passed")
    print(json.dumps(full_audit["sanitization"], indent=2))
    print()
    clean_vectorizer, _ = run("2. canonical sanitized TF-IDF", documents=sanitized)
    assert_vocabulary_clean(clean_vectorizer.get_feature_names_out())
    print(
        "\nPASS: canonical TF-IDF uses the same retained HGT-source sections, with zero "
        "mask artifacts and zero direct operative-outcome terms in its fitted vocabulary."
    )


if __name__ == "__main__":
    main()
