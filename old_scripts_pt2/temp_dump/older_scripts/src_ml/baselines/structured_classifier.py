from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from src_ml.common.metrics import compute_classification_metrics
from src_ml.common.sklearn_compat import make_logistic_regression
from src_ml.common.text_utils import safe_list, safe_text


@dataclass
class StructuredArtifacts:
    vectorizer: DictVectorizer
    classifier: Any


def _row_to_feature_dict(row: pd.Series) -> dict[str, float]:
    feats: dict[str, float] = {}

    court = safe_text(row.get("court"))
    if court:
        feats[f"court={court}"] = 1.0

    year = row.get("year")
    if year is not None and str(year) != "nan":
        feats[f"year={int(year)}"] = 1.0

    case_type = safe_text(row.get("case_type"))
    if case_type:
        feats[f"case_type={case_type}"] = 1.0

    for statute in safe_list(row.get("statutes")):
        feats[f"statute={statute}"] = 1.0
    for provision in safe_list(row.get("provisions")):
        feats[f"provision={provision}"] = 1.0
    for precedent in safe_list(row.get("precedents")):
        feats[f"precedent={precedent}"] = 1.0

    return feats


def _extract_confidence(proba: np.ndarray | None) -> np.ndarray | None:
    if proba is None:
        return None
    if proba.ndim == 1:
        return proba
    return np.max(proba, axis=1)


def train_structured_classifier(
    df: pd.DataFrame,
    label_col: str,
    split_col: str,
    model_cfg: dict[str, Any],
    label_names: list[str],
) -> tuple[StructuredArtifacts, dict[str, Any], pd.DataFrame]:
    train_df = df[df[split_col] == "train"].copy()
    val_df = df[df[split_col] == "val"].copy()
    test_df = df[df[split_col] == "test"].copy()

    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(train_df.apply(_row_to_feature_dict, axis=1).tolist())
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

    metrics: dict[str, Any] = {
        "model": "structured_logreg",
        "labels": label_names,
    }
    preds_frames: list[pd.DataFrame] = []

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X_split = vectorizer.transform(split_df.apply(_row_to_feature_dict, axis=1).tolist())
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
    return StructuredArtifacts(vectorizer=vectorizer, classifier=clf), metrics, preds_df
