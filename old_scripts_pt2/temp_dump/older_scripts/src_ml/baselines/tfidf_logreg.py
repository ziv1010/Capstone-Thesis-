from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src_ml.common.metrics import compute_classification_metrics
from src_ml.common.sklearn_compat import make_logistic_regression
from src_ml.common.text_utils import safe_text


@dataclass
class TfidfLogRegArtifacts:
    vectorizer: TfidfVectorizer
    classifier: Any


def _extract_confidence(proba: np.ndarray | None) -> np.ndarray | None:
    if proba is None:
        return None
    if proba.ndim == 1:
        return proba
    return np.max(proba, axis=1)


def train_tfidf_logreg(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    split_col: str,
    tfidf_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    label_names: list[str],
) -> tuple[TfidfLogRegArtifacts, dict[str, Any], pd.DataFrame]:
    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()

    vectorizer = TfidfVectorizer(
        max_features=int(tfidf_cfg.get("max_features", 100000)),
        ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
        min_df=int(tfidf_cfg.get("min_df", 1)),
        max_df=float(tfidf_cfg.get("max_df", 1.0)),
        sublinear_tf=bool(tfidf_cfg.get("sublinear_tf", True)),
    )

    X_train = vectorizer.fit_transform(train_df[text_col].fillna("").map(safe_text))
    y_train = train_df[label_col].astype(int).values

    clf = make_logistic_regression(
        max_iter=int(model_cfg.get("max_iter", 2000)),
        C=float(model_cfg.get("C", 1.0)),
        class_weight=model_cfg.get("class_weight"),
        n_jobs=int(model_cfg.get("n_jobs", 1)),
        solver=str(model_cfg.get("solver", "lbfgs")),
        multi_class=str(model_cfg.get("multi_class", "auto")),
        random_state=int(model_cfg.get("seed", 42)),
    )
    clf.fit(X_train, y_train)

    preds_frames: list[pd.DataFrame] = []
    metrics: dict[str, Any] = {
        "model": "tfidf_logreg",
        "text_col": text_col,
        "labels": label_names,
    }

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X_split = vectorizer.transform(split_df[text_col].fillna("").map(safe_text))
        y_true = split_df[label_col].astype(int).values
        y_pred = clf.predict(X_split)

        y_proba = clf.predict_proba(X_split) if hasattr(clf, "predict_proba") else None
        split_metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(range(len(label_names))),
            label_names=label_names,
            y_proba=y_proba,
        )
        metrics[f"{split_name}_metrics"] = split_metrics

        confidence = _extract_confidence(y_proba)
        frame = pd.DataFrame(
            {
                "case_id": split_df["case_id"].astype(str).values,
                "split": split_name,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_name": [label_names[int(i)] for i in y_pred],
                "prob_or_confidence": confidence if confidence is not None else np.nan,
            }
        )
        preds_frames.append(frame)

    preds_df = pd.concat(preds_frames, ignore_index=True)
    artifacts = TfidfLogRegArtifacts(vectorizer=vectorizer, classifier=clf)
    return artifacts, metrics, preds_df
