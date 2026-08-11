#!/usr/bin/env python3
"""Which section carries the flat-text signal? (diagnostic for R3-03)

Re-reads the cleaned cases directly and rebuilds the TF-IDF document from
individual sections, so we can tell apart:
  * PREAMBLE  -- party names, court, case type: pre-decision base rates
  * FAC       -- the factual narrative
  * arguments -- what the parties argued
Same folds, same sanitizer, same LinearSVC as B3.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import LABEL_NAMES, REFERENCE_KFOLD, TFIDF_KWARGS
from text_sanitization import assert_vocabulary_clean, sanitize_document
from build_features import CLEANED_CASES

paths = sorted(glob.glob(str(CLEANED_CASES / "*.json")))
sections = {"preamble": [], "facts": [], "arguments": []}
ids, labels = [], []
for p in paths:
    c = json.load(open(p, encoding="utf-8")); t = c.get("texts") or {}
    for s in sections: sections[s].append(str(t.get(s) or ""))
    ids.append(c["case_id"]); labels.append(str(c["raw_label"]))
y = np.array([LABEL_NAMES.index(v) for v in labels])
print(f"{len(ids):,} cases loaded")

pos = {c: i for i, c in enumerate(ids)}
folds = []
for f in range(5):
    fr = pd.read_csv(REFERENCE_KFOLD / f"fold_{f:02d}" / "predictions.csv")
    r = fr["case_id"].map(pos).to_numpy(); sp = fr["split"].to_numpy()
    folds.append({k: r[sp == k] for k in ("train", "val", "test")})

VARIANTS = {
    "preamble only":            ["preamble"],
    "facts only":               ["facts"],
    "arguments only":           ["arguments"],
    "facts + arguments":        ["facts", "arguments"],
    "all three (fixed C)":         ["preamble", "facts", "arguments"],
}
FOLDS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0"])]
out = {}
for name, keys in VARIANTS.items():
    docs = [sanitize_document("\n".join(sections[k][i] for k in keys)) for i in range(len(ids))]
    accs, f1s = [], []
    for f in FOLDS:
        tr, te = folds[f]["train"], folds[f]["test"]
        v = TfidfVectorizer(**TFIDF_KWARGS)
        Xtr = v.fit_transform([docs[i] for i in tr]); Xte = v.transform([docs[i] for i in te])
        assert_vocabulary_clean(v.get_feature_names_out())
        m = LinearSVC(C=0.25, class_weight="balanced", dual=True, max_iter=4000, random_state=42+f).fit(Xtr, y[tr])
        p = m.predict(Xte)
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, average="macro"))
    out[name] = {"sections": keys, "folds": FOLDS, "accuracy_mean": float(np.mean(accs)),
                 "accuracy_std": float(np.std(accs)), "macro_f1_mean": float(np.mean(f1s)),
                 "n_features": int(Xtr.shape[1])}
    print(f"  {name:<20} acc={np.mean(accs):.4f}  macroF1={np.mean(f1s):.4f}  ({Xtr.shape[1]:,} feats)", flush=True)
Path("outputs/section_ablation.json").write_text(json.dumps(out, indent=2))
print("\nwrote outputs/section_ablation.json")
