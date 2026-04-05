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


def save_training_history_plot(
    history: list[dict[str, Any]],
    output_path: str | Path,
    best_epoch: int | None = None,
    title: str = "Training History",
) -> None:
    if not history:
        return

    epochs = [int(item.get("epoch", idx + 1)) for idx, item in enumerate(history)]
    train_loss = [float(item.get("train_loss", 0.0)) for item in history]
    train_macro_f1 = [float(item.get("train_macro_f1", 0.0)) for item in history]
    val_macro_f1 = [float(item.get("val_macro_f1", 0.0)) for item in history]

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=2, label="Train loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_macro_f1, color="#2ca02c", linewidth=2, label="Train macro F1")
    axes[1].plot(epochs, val_macro_f1, color="#d62728", linewidth=2, label="Val macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if best_epoch is not None and best_epoch > 0:
        for axis in axes:
            axis.axvline(best_epoch, color="#444444", linestyle="--", linewidth=1.5, alpha=0.8)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_split_metric_bar_plot(
    split_metrics: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str = "Split Metrics",
) -> None:
    metric_names = ["accuracy", "macro_f1", "micro_f1"]
    split_names = ["train", "val", "test"]
    x_positions = np.arange(len(metric_names), dtype=np.float32)
    width = 0.22

    plt.figure(figsize=(8, 5))
    for offset_idx, split_name in enumerate(split_names):
        metric_values = [float(split_metrics.get(split_name, {}).get(metric_name, 0.0)) for metric_name in metric_names]
        plt.bar(
            x_positions + (offset_idx - 1) * width,
            metric_values,
            width=width,
            label=split_name,
            alpha=0.9,
        )

    plt.xticks(x_positions, ["Accuracy", "Macro F1", "Micro F1"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
