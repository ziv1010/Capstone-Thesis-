from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    label_names: list[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if y_proba.ndim == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            elif y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    return metrics
