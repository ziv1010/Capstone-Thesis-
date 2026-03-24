from __future__ import annotations

from typing import Any

import numpy as np

from src.training.metrics import compute_metrics


def evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n_samples": 0}
    return compute_metrics(y_true=y_true, y_pred=y_pred, y_proba=y_proba, label_names=label_names)
