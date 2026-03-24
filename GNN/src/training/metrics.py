from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    labels = list(range(len(label_names)))
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)) if len(y_true) else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if len(y_true) else [],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
        if len(y_true)
        else {},
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    metrics["per_class"] = {
        label_names[idx]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(label_names))
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if y_proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba[:, 1]))
            elif y_proba.shape[1] > 2:
                metrics["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return metrics


def save_confusion_matrix_plot(
    confusion: list[list[int]],
    label_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    matrix = np.asarray(confusion, dtype=np.int64)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(label_names))
    plt.xticks(ticks, label_names, rotation=45, ha="right")
    plt.yticks(ticks, label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            plt.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
