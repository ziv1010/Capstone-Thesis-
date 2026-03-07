from __future__ import annotations

import inspect
from typing import Any

import sklearn
from sklearn.linear_model import LogisticRegression


def _parse_version(version: str) -> tuple[int, int]:
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def make_logistic_regression(**kwargs: Any) -> LogisticRegression:
    """Create LogisticRegression while tolerating sklearn API drift.

    Newer sklearn releases can remove kwargs (e.g. multi_class in 1.8+).
    This helper only forwards parameters supported by the installed version.
    """
    sig = inspect.signature(LogisticRegression.__init__)
    supported = set(sig.parameters.keys()) - {"self"}

    filtered: dict[str, Any] = {
        key: value
        for key, value in kwargs.items()
        if key in supported and value is not None
    }

    major, minor = _parse_version(getattr(sklearn, "__version__", "0.0"))
    if (major, minor) >= (1, 8):
        # Deprecated/no-op in sklearn 1.8+
        filtered.pop("penalty", None)
        filtered.pop("n_jobs", None)

    return LogisticRegression(**filtered)
