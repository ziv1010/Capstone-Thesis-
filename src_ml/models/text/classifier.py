from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier

from src_ml.common.sklearn_compat import make_logistic_regression


def train_text_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> Any:
    classifier_type = str(cfg.get("type", "logreg"))

    if classifier_type == "logreg":
        clf = make_logistic_regression(
            max_iter=int(cfg.get("max_iter", 2000)),
            C=float(cfg.get("C", 1.0)),
            class_weight=cfg.get("class_weight"),
            n_jobs=int(cfg.get("n_jobs", 1)),
            solver=str(cfg.get("solver", "lbfgs")),
            multi_class=str(cfg.get("multi_class", "auto")),
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        return clf

    if classifier_type == "mlp":
        hidden = cfg.get("hidden_layer_sizes", [256])
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(int(x) for x in hidden),
            activation=str(cfg.get("activation", "relu")),
            alpha=float(cfg.get("alpha", 1e-4)),
            learning_rate_init=float(cfg.get("learning_rate_init", 1e-3)),
            max_iter=int(cfg.get("max_iter", 200)),
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        return clf

    raise ValueError(f"Unsupported classifier type: {classifier_type}")
