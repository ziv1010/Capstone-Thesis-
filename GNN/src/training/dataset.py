from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.graph.schema import CleanedCase


@dataclass
class PreparedCases:
    cases: list[CleanedCase]
    label_names: list[str]
    y: np.ndarray
    dropped_case_ids: list[str]


def prepare_cases_for_task(
    cleaned_cases: list[CleanedCase],
    labels_cfg: dict[str, Any],
) -> PreparedCases:
    task_mode = str(labels_cfg.get("task_mode", "binary"))
    drop_procedural = bool(labels_cfg.get("drop_procedural", True))
    if task_mode == "binary":
        label_map = labels_cfg.get("binary_map", {})
        label_names = [str(label) for label in labels_cfg.get("class_order_binary", ["lose", "win"])]
    elif task_mode == "multiclass":
        label_map = labels_cfg.get("multiclass_map", {})
        label_names = [
            str(label) for label in labels_cfg.get("class_order_multiclass", ["lose", "procedural", "win"])
        ]
    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    kept_cases: list[CleanedCase] = []
    targets: list[int] = []
    dropped_case_ids: list[str] = []

    for case in cleaned_cases:
        mapped = label_map.get(case.raw_label)
        if mapped is None:
            if task_mode == "binary" and drop_procedural:
                dropped_case_ids.append(case.case_id)
                continue
            raise ValueError(
                "Encountered unmapped label "
                f"{case.raw_label!r} for case {case.case_id}. Adjust label config before building the graph."
            )
        targets.append(label_names.index(str(mapped)))
        kept_cases.append(case)

    return PreparedCases(
        cases=kept_cases,
        label_names=label_names,
        y=np.asarray(targets, dtype=np.int64),
        dropped_case_ids=dropped_case_ids,
    )


def _case_group(case: CleanedCase, mode: str) -> str:
    if mode == "year":
        year = case.metadata.get("case_year")
        return "unknown_year" if year is None else str(year)
    if mode == "court":
        courts = case.metadata.get("court_names", [])
        if courts:
            return str(courts[0])
        return "unknown_court"
    return case.case_id


def build_split_assignments(
    cleaned_cases: list[CleanedCase],
    y: np.ndarray,
    splits_cfg: dict[str, Any],
) -> dict[str, str]:
    mode = str(splits_cfg.get("mode", "random"))
    random_state = int(splits_cfg.get("random_state", 42))
    train_size = float(splits_cfg.get("train_size", 0.7))
    val_size = float(splits_cfg.get("val_size", 0.15))
    test_size = float(splits_cfg.get("test_size", 0.15))
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    case_ids = [case.case_id for case in cleaned_cases]
    indices = np.arange(len(cleaned_cases))
    stratify = y if bool(splits_cfg.get("stratify", True)) and len(np.unique(y)) > 1 else None

    if len(cleaned_cases) < 3:
        return {case_id: "train" for case_id in case_ids}

    if mode == "random":
        try:
            train_idx, temp_idx, y_train, y_temp = train_test_split(
                indices,
                y,
                train_size=train_size,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError:
            train_idx, temp_idx, y_train, y_temp = train_test_split(
                indices,
                y,
                train_size=train_size,
                random_state=random_state,
                stratify=None,
            )
        remaining_test_ratio = test_size / (val_size + test_size)
        temp_stratify = y_temp if len(np.unique(y_temp)) > 1 else None
        try:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=remaining_test_ratio,
                random_state=random_state,
                stratify=temp_stratify,
            )
        except ValueError:
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=remaining_test_ratio,
                random_state=random_state,
                stratify=None,
            )
    elif mode in {"year", "court"}:
        groups = np.asarray([_case_group(case, mode) for case in cleaned_cases], dtype=object)
        train_splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, temp_idx = next(train_splitter.split(indices, y, groups))
        remaining_test_ratio = test_size / (val_size + test_size)
        temp_groups = groups[temp_idx]
        temp_y = y[temp_idx]
        val_test_splitter = GroupShuffleSplit(
            n_splits=1,
            train_size=1.0 - remaining_test_ratio,
            random_state=random_state,
        )
        val_rel_idx, test_rel_idx = next(val_test_splitter.split(temp_idx, temp_y, temp_groups))
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]
    else:
        raise ValueError(f"Unsupported split mode: {mode}")

    assignments = {case_ids[idx]: "train" for idx in train_idx}
    assignments.update({case_ids[idx]: "val" for idx in val_idx})
    assignments.update({case_ids[idx]: "test" for idx in test_idx})
    return assignments
