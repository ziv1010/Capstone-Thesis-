#!/usr/bin/env python3
"""Non-LLM baselines on the HGT's own five folds (Reviewer 3, R3-03).

The splits are not re-derived. They are read out of the HGT run's own
``fold_XX/predictions.csv``, which records ``split in {train,val,test}`` for all
71,813 cases in graph-node order. The five test folds are asserted to be
pairwise disjoint and to cover the corpus exactly, so every number here is
directly comparable to the paper's 80.63% / 0.8002.

Protocol, identical for every model and every fold:

  1. Vectorisers, scalers, SVD and authority vocabularies are fit on
     ``split == "train"`` rows ONLY, then applied to val and test.
  2. The single regularisation hyperparameter is chosen on ``split == "val"``
     macro-F1.
  3. ``split == "test"`` is scored once.
  4. Class weighting is on everywhere, matching the HGT's balanced
     cross-entropy.
  5. Metrics come from ``section_GNN/src/training/metrics.py::compute_metrics``
     -- the same function that produced the HGT numbers.

All reported TF-IDF models apply a final lexical guard to the HGT-source text:
``[LEAKAGE_MASK]`` artifacts and direct operative-outcome words are removed and
the fitted vocabulary is asserted clean. The optional B7 diagnostic is invalid
as a baseline and is never run by default.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_SECTION_GNN = _REPO / "section_GNN"
if str(_SECTION_GNN) not in sys.path:
    sys.path.insert(0, str(_SECTION_GNN))

from src.training.metrics import compute_metrics  # noqa: E402  -- the paper's own metrics
from text_sanitization import (  # noqa: E402
    assert_vocabulary_clean,
    audit_documents,
    sanitize_document,
)

REFERENCE_KFOLD = (
    _SECTION_GNN
    / "outputs/ablations/entity_resolved_data/cross_bucket_total_dataset/models"
    / "ablation_entity_resolved_section_sep_lr_decay_cross_bucket_kfold/kfold"
)
FEATURES_DIR = _HERE / "outputs" / "features"
OUTPUTS_DIR = _HERE / "outputs"
LABEL_NAMES = ["-1", "1"]  # config.yaml labels.class_order_binary
K = 5
BASE_SEED = 42

TFIDF_KWARGS = dict(
    ngram_range=(1, 2),
    min_df=5,
    max_features=300_000,
    sublinear_tf=True,
    strip_accents="unicode",
    lowercase=True,
    dtype=np.float32,
)

BASELINES = {
    "B0_majority": {"label": "Majority class", "features": "--", "valid": True},
    "B1_stratified": {"label": "Stratified random", "features": "--", "valid": True},
    "B2_logreg_entity": {"label": "Logistic Regression", "features": "entity counts + case scalars", "valid": True},
    "B3_svm_tfidf": {"label": "Linear SVM", "features": "sanitized TF-IDF (1--2 gram)", "valid": True},
    "B4_logreg_tfidf": {"label": "Logistic Regression", "features": "sanitized TF-IDF (1--2 gram)", "valid": True},
    "B5_xgboost_entity": {"label": "XGBoost", "features": "entity counts + authority counts", "valid": True},
    "B6_xgboost_tfidf_svd": {"label": "XGBoost", "features": "sanitized TF-IDF $\\to$ SVD-256", "valid": True},
    "B7_oracle_rawtext": {"label": "Oracle: Linear SVM", "features": "unfiltered raw judgment text", "valid": False},
}

# ------------------------------------------------------------------ loading


def _read_jsonl_gz(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_folds(case_ids: list[str]) -> list[dict[str, np.ndarray]]:
    """Fold membership taken verbatim from the HGT run's predictions.csv."""
    position = {case_id: i for i, case_id in enumerate(case_ids)}
    folds, test_sets = [], []
    for fold in range(K):
        frame = pd.read_csv(REFERENCE_KFOLD / f"fold_{fold:02d}" / "predictions.csv")
        rows = frame["case_id"].map(position)
        if rows.isna().any():
            missing = int(rows.isna().sum())
            raise SystemExit(f"fold {fold}: {missing} case_ids in predictions.csv are absent from the feature cache")
        rows = rows.to_numpy(dtype=np.int64)
        split = frame["split"].to_numpy()
        indices = {name: rows[split == name] for name in ("train", "val", "test")}
        folds.append(indices)
        test_sets.append(set(indices["test"].tolist()))
    overlaps = sum(len(test_sets[a] & test_sets[b]) for a in range(K) for b in range(a + 1, K))
    union = set().union(*test_sets)
    if overlaps or len(union) != len(case_ids):
        raise SystemExit(f"folds do not partition the corpus: overlap={overlaps}, union={len(union)}/{len(case_ids)}")
    print(f"folds verified: pairwise test overlap=0, union={len(union):,}/{len(case_ids):,}")
    return folds


def check_labels(case_ids: list[str], y: np.ndarray) -> None:
    """The cached labels must equal the labels the HGT trained against."""
    frame = pd.read_csv(REFERENCE_KFOLD / "fold_00" / "predictions.csv")
    position = {case_id: i for i, case_id in enumerate(case_ids)}
    rows = frame["case_id"].map(position).to_numpy(dtype=np.int64)
    recorded = np.array([LABEL_NAMES.index(str(v)) for v in frame["raw_label"]])
    mismatches = int((y[rows] != recorded).sum())
    if mismatches:
        raise SystemExit(f"{mismatches} cached labels disagree with predictions.csv")
    print(f"labels verified against predictions.csv: 0 mismatches over {len(rows):,} cases")


# ----------------------------------------------------------- model fitting


def _fit_score(model, X_train, y_train, X_val, X_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        proba_val, proba_test = model.predict_proba(X_val), model.predict_proba(X_test)
    else:  # LinearSVC -- decision_function is rank-equivalent for ROC/PR AUC
        d_val, d_test = model.decision_function(X_val), model.decision_function(X_test)
        proba_val = np.column_stack([-d_val, d_val])
        proba_test = np.column_stack([-d_test, d_test])
    return proba_val.argmax(1), proba_val, proba_test.argmax(1), proba_test


def _select_on_val(candidates, X_train, y_train, X_val, y_val, X_test):
    """Pick the hyperparameter with the best validation macro-F1. Test is untouched."""
    best = None
    for name, model in candidates:
        pred_val, _, pred_test, proba_test = _fit_score(model, X_train, y_train, X_val, X_test)
        score = f1_score(y_val, pred_val, average="macro", zero_division=0)
        if best is None or score > best["val_macro_f1"]:
            best = {
                "hyperparameter": name,
                "val_macro_f1": float(score),
                "pred_test": pred_test,
                "proba_test": proba_test,
                "model": model,
            }
    return best


def _authority_matrix(auth_counts: list[dict], rows: np.ndarray, vocabulary: dict[str, int]) -> sparse.csr_matrix:
    indptr, indices, data = [0], [], []
    for row in rows:
        entry = auth_counts[row]
        for entity_type, counts in entry.items():
            for name, count in counts.items():
                column = vocabulary.get(f"{entity_type}:{name}")
                if column is not None:
                    indices.append(column)
                    data.append(float(count))
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices), np.array(indptr)),
        shape=(len(rows), len(vocabulary)),
    )


def _authority_vocabulary(auth_counts: list[dict], train_rows: np.ndarray, top_k: int) -> dict[str, int]:
    """Top-K authorities by TRAIN document frequency only."""
    document_frequency: dict[str, int] = {}
    for row in train_rows:
        for entity_type, counts in auth_counts[row].items():
            for name in counts:
                key = f"{entity_type}:{name}"
                document_frequency[key] = document_frequency.get(key, 0) + 1
    ranked = sorted(document_frequency.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return {key: i for i, (key, _) in enumerate(ranked)}


def _xgb(seed: int, n_estimators: int, max_depth: int, pos_weight: float, n_jobs: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )


# ------------------------------------------------------------------- folds


def run_fold(fold: int, data: dict, indices: dict, args) -> dict[str, dict]:
    seed = BASE_SEED + fold
    train, val, test = indices["train"], indices["val"], indices["test"]
    y = data["y"]
    y_train, y_val, y_test = y[train], y[val], y[test]
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    results: dict[str, dict] = {}

    def record(name: str, best: dict, seconds: float, extra: dict | None = None) -> None:
        metrics = compute_metrics(
            y_true=y_test, y_pred=best["pred_test"], label_names=LABEL_NAMES, y_proba=best["proba_test"]
        )
        # Per-case test predictions, so make_tables.py can run a paired McNemar
        # test against the HGT's own predictions over the pooled folds.
        fold_dir = args.out / "per_fold" / name / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "case_id": [data["case_ids"][i] for i in test],
                "target_index": y_test,
                "pred_index": best["pred_test"],
            }
        ).to_csv(fold_dir / "predictions.csv", index=False)
        metrics["hyperparameter"] = best["hyperparameter"]
        metrics["val_macro_f1"] = best["val_macro_f1"]
        metrics["fit_seconds"] = round(seconds, 1)
        metrics["valid_baseline"] = BASELINES[name]["valid"]
        if extra:
            metrics.update(extra)
        results[name] = metrics
        print(
            f"    {name:24s} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"({best['hyperparameter']}, {seconds:.0f}s)",
            flush=True,
        )

    def dummy(name: str, strategy: str) -> None:
        start = time.time()
        model = DummyClassifier(strategy=strategy, random_state=seed)
        pred_val, _, pred_test, proba_test = _fit_score(
            model, np.zeros((len(train), 1)), y_train, np.zeros((len(val), 1)), np.zeros((len(test), 1))
        )
        best = {
            "hyperparameter": strategy,
            "val_macro_f1": float(f1_score(y_val, pred_val, average="macro", zero_division=0)),
            "pred_test": pred_test,
            "proba_test": proba_test,
        }
        record(name, best, time.time() - start)

    if "B0_majority" in args.models:
        dummy("B0_majority", "most_frequent")
    if "B1_stratified" in args.models:
        dummy("B1_stratified", "stratified")

    # --- B2: Logistic Regression on entity counts (the reviewer's phrasing)
    if "B2_logreg_entity" in args.models:
        start = time.time()
        scaler = StandardScaler().fit(data["ent"][train])
        Xtr, Xva, Xte = (scaler.transform(data["ent"][idx]) for idx in (train, val, test))
        best = _select_on_val(
            [
                (f"C={c}", LogisticRegression(C=c, class_weight="balanced", max_iter=3000, n_jobs=args.n_jobs))
                for c in (0.1, 1.0, 10.0)
            ],
            Xtr, y_train, Xva, y_val, Xte,
        )
        record("B2_logreg_entity", best, time.time() - start, {"n_features": int(Xtr.shape[1])})

    # --- Sanitized TF-IDF is fit once per fold on train rows and shared by B3/B4/B6.
    needs_tfidf = {"B3_svm_tfidf", "B4_logreg_tfidf", "B6_xgboost_tfidf_svd"} & set(args.models)
    if needs_tfidf:
        start = time.time()
        texts = data["texts"]
        vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
        Ttr = vectorizer.fit_transform([texts[i] for i in train])
        Tva = vectorizer.transform([texts[i] for i in val])
        Tte = vectorizer.transform([texts[i] for i in test])
        assert_vocabulary_clean(vectorizer.get_feature_names_out())
        print(f"    tf-idf: {Ttr.shape[1]:,} features fit on train only ({time.time() - start:.0f}s)", flush=True)

        if "B3_svm_tfidf" in args.models:
            start = time.time()
            best = _select_on_val(
                [(f"C={c}", LinearSVC(C=c, class_weight="balanced", dual=True, max_iter=4000, random_state=seed))
                 for c in (0.05, 0.25, 1.0)],
                Ttr, y_train, Tva, y_val, Tte,
            )
            top = _top_terms(best["model"], vectorizer)
            record(
                "B3_svm_tfidf",
                best,
                time.time() - start,
                {
                    "n_features": int(Ttr.shape[1]),
                    "top_terms": top,
                    "text_sanitization": data["text_sanitization"],
                },
            )

        if "B4_logreg_tfidf" in args.models:
            start = time.time()
            best = _select_on_val(
                [(f"C={c}", LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=3000))
                 for c in (1.0, 4.0, 16.0)],
                Ttr, y_train, Tva, y_val, Tte,
            )
            record(
                "B4_logreg_tfidf",
                best,
                time.time() - start,
                {"n_features": int(Ttr.shape[1]), "text_sanitization": data["text_sanitization"]},
            )

        if "B6_xgboost_tfidf_svd" in args.models:
            start = time.time()
            svd = TruncatedSVD(n_components=256, algorithm="randomized", n_iter=4, random_state=seed).fit(Ttr)
            Str, Sva, Ste = svd.transform(Ttr), svd.transform(Tva), svd.transform(Tte)
            best = _select_on_val(
                [(f"n={n},d={d}", _xgb(seed, n, d, pos_weight, args.n_jobs)) for n, d in ((400, 6), (800, 8))],
                Str, y_train, Sva, y_val, Ste,
            )
            record(
                "B6_xgboost_tfidf_svd",
                best,
                time.time() - start,
                {"n_features": 256, "text_sanitization": data["text_sanitization"]},
            )

    # --- B5: XGBoost on entity counts + top-K canonical authority counts
    if "B5_xgboost_entity" in args.models:
        start = time.time()
        vocabulary = _authority_vocabulary(data["auth"], train, args.auth_top_k)
        Atr, Ava, Ate = (_authority_matrix(data["auth"], idx, vocabulary) for idx in (train, val, test))
        Xtr = sparse.hstack([sparse.csr_matrix(data["ent"][train]), Atr]).tocsr()
        Xva = sparse.hstack([sparse.csr_matrix(data["ent"][val]), Ava]).tocsr()
        Xte = sparse.hstack([sparse.csr_matrix(data["ent"][test]), Ate]).tocsr()
        best = _select_on_val(
            [(f"n={n},d={d}", _xgb(seed, n, d, pos_weight, args.n_jobs)) for n, d in ((400, 6), (800, 8))],
            Xtr, y_train, Xva, y_val, Xte,
        )
        record(
            "B5_xgboost_entity", best, time.time() - start,
            {"n_features": int(Xtr.shape[1]), "n_authority_features": len(vocabulary)},
        )

    # --- B7: ORACLE. Unfiltered raw text -- contains the operative decision.
    if "B7_oracle_rawtext" in args.models:
        start = time.time()
        raw = data["rawtext"]
        vectorizer = TfidfVectorizer(**TFIDF_KWARGS)
        Rtr = vectorizer.fit_transform([raw[i] for i in train])
        Rva = vectorizer.transform([raw[i] for i in val])
        Rte = vectorizer.transform([raw[i] for i in test])
        best = _select_on_val(
            [(f"C={c}", LinearSVC(C=c, class_weight="balanced", dual=True, max_iter=4000, random_state=seed))
             for c in (0.25, 1.0)],
            Rtr, y_train, Rva, y_val, Rte,
        )
        top = _top_terms(best["model"], vectorizer)
        record("B7_oracle_rawtext", best, time.time() - start, {"n_features": int(Rtr.shape[1]), "top_terms": top})

    return results


def _top_terms(model, vectorizer, k: int = 40) -> dict[str, list[str]]:
    """Highest-weight n-grams per class, so residual outcome phrases are visible."""
    coefficients = np.asarray(model.coef_).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(coefficients)
    return {
        LABEL_NAMES[0]: [str(t) for t in terms[order[:k]]],
        LABEL_NAMES[1]: [str(t) for t in terms[order[-k:][::-1]]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument(
        "--models",
        default=",".join(name for name, spec in BASELINES.items() if spec["valid"]),
        help="comma-separated model IDs; the raw-decision oracle is opt-in",
    )
    parser.add_argument("--auth-top-k", type=int, default=4000)
    parser.add_argument("--n-jobs", type=int, default=32)
    args = parser.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(args.models) - set(BASELINES)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    folds_to_run = [int(f) for f in args.folds.split(",")]

    print("loading feature caches ...", flush=True)
    index = json.loads((args.features / "case_index.json").read_text())
    case_ids = index["case_ids"]
    y = np.array([LABEL_NAMES.index(v) for v in index["labels"]], dtype=np.int64)
    payload = np.load(args.features / "ent.npz", allow_pickle=False)
    data = {"case_ids": case_ids, "y": y, "ent": payload["X"].astype(np.float64)}
    if {"B3_svm_tfidf", "B4_logreg_tfidf", "B6_xgboost_tfidf_svd"} & set(args.models):
        source_texts = _read_jsonl_gz(args.features / "text.jsonl.gz")
        data["texts"] = [sanitize_document(text) for text in source_texts]
        data["text_sanitization"] = audit_documents(source_texts, data["texts"])
        print(f"  text sanitization audit: {data['text_sanitization']}", flush=True)
    if "B5_xgboost_entity" in args.models:
        data["auth"] = _read_jsonl_gz(args.features / "auth.jsonl.gz")
    if "B7_oracle_rawtext" in args.models:
        data["rawtext"] = _read_jsonl_gz(args.features / "rawtext.jsonl.gz")
    print(f"  {len(case_ids):,} cases, ENT {data['ent'].shape}", flush=True)

    check_labels(case_ids, y)
    folds = load_folds(case_ids)

    per_fold: dict[str, dict[int, dict]] = {name: {} for name in args.models}
    for fold in folds_to_run:
        print(f"\n=== fold {fold} ===", flush=True)
        results = run_fold(fold, data, folds[fold], args)
        for name, metrics in results.items():
            per_fold[name][fold] = metrics
            fold_dir = args.out / "per_fold" / name / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Fold results already on disk from an earlier invocation are folded back in,
    # so models can be added incrementally without re-running the whole sweep.
    for model_dir in sorted((args.out / "per_fold").glob("*")):
        name = model_dir.name
        if name not in BASELINES or not BASELINES[name]["valid"]:
            continue
        per_fold.setdefault(name, {})
        for fold_dir in sorted(model_dir.glob("fold_*")):
            fold = int(fold_dir.name.removeprefix("fold_"))
            if fold not in per_fold[name] and (fold_dir / "metrics.json").exists():
                per_fold[name][fold] = json.loads((fold_dir / "metrics.json").read_text())
    report_models = [name for name, spec in BASELINES.items() if spec["valid"] and per_fold.get(name)]

    # --- aggregate exactly like kfold_cv_v2._aggregate (np.std, ddof=0)
    summary = {
        "reference_run": str(REFERENCE_KFOLD),
        "k": K,
        "n_cases": len(case_ids),
        "label_names": LABEL_NAMES,
        "note": "Splits read verbatim from the HGT run's predictions.csv. "
                "TF-IDF uses the same retained HGT-source sections after removing mask artifacts "
                "and direct operative-outcome vocabulary, with corpus/vocabulary assertions. "
                "Vectorisers/scalers/SVD/vocabularies fit on train rows only; "
                "hyperparameters selected on val; test scored once.",
        "baselines": {},
    }
    for name in report_models:
        folds_done = sorted(per_fold[name])
        if not folds_done:
            continue
        pick = lambda key: [per_fold[name][f][key] for f in folds_done]  # noqa: E731
        aggregate = {
            "accuracy_mean": float(np.mean(pick("accuracy"))), "accuracy_std": float(np.std(pick("accuracy"))),
            "macro_f1_mean": float(np.mean(pick("macro_f1"))), "macro_f1_std": float(np.std(pick("macro_f1"))),
            "micro_f1_mean": float(np.mean(pick("micro_f1"))), "micro_f1_std": float(np.std(pick("micro_f1"))),
        }
        aucs = [per_fold[name][f]["roc_auc"] for f in folds_done if "roc_auc" in per_fold[name][f]]
        if aucs:
            aggregate["roc_auc_mean"] = float(np.mean(aucs))
            aggregate["roc_auc_std"] = float(np.std(aucs))
        summary["baselines"][name] = {
            **{k: v for k, v in BASELINES[name].items()},
            "n_folds_completed": len(folds_done),
            "aggregate": aggregate,
            "folds": {
                str(f): {k: v for k, v in per_fold[name][f].items()
                         if k not in ("classification_report", "top_terms")}
                for f in folds_done
            },
        }
        top_terms = {str(f): per_fold[name][f]["top_terms"] for f in folds_done if "top_terms" in per_fold[name][f]}
        if top_terms:
            (args.out / f"top_features_{name}.json").write_text(json.dumps(top_terms, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baselines_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== summary ===")
    for name, entry in summary["baselines"].items():
        agg = entry["aggregate"]
        flag = "" if entry["valid"] else "   <-- ORACLE, not a baseline"
        print(f"  {name:24s} acc={agg['accuracy_mean']:.4f}±{agg['accuracy_std']:.4f}  "
              f"macro_f1={agg['macro_f1_mean']:.4f}±{agg['macro_f1_std']:.4f}{flag}")
    print(f"\nwrote {args.out / 'baselines_summary.json'}")


if __name__ == "__main__":
    main()
