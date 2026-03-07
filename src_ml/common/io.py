from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def iter_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= int(limit):
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def flatten_case_record(record: dict[str, Any]) -> dict[str, Any]:
    texts = record.get("texts") or {}
    ml = record.get("ml") or {}
    outcome = record.get("outcome") or {}

    date_value = record.get("date")
    year = None
    if isinstance(date_value, str) and len(date_value) >= 4 and date_value[:4].isdigit():
        year = int(date_value[:4])

    return {
        "case_id": record.get("case_id"),
        "file_name": record.get("file_name"),
        "court": record.get("court"),
        "bench": record.get("bench"),
        "case_type": record.get("case_type"),
        "case_title": record.get("case_title"),
        "date": date_value,
        "year": year,
        "judge_names": _to_list(record.get("judge_names")),
        "statutes": _to_list(record.get("statutes")),
        "provisions": _to_list(record.get("provisions")),
        "precedents": _to_list(record.get("precedents")),
        "facts_text": texts.get("facts_text"),
        "arguments_petitioner": texts.get("arguments_petitioner"),
        "arguments_respondent": texts.get("arguments_respondent"),
        "reasoning_text": texts.get("reasoning_text"),
        "decision_text": texts.get("decision_text"),
        "ml_input_text": ml.get("input_text"),
        "ml_leakage_flag": bool(ml.get("leakage_flag", False)),
        "outcome_label": outcome.get("label"),
        "outcome_winner": outcome.get("winner"),
        "decision": record.get("decision"),
    }


def iter_case_chunks(
    path: str | Path,
    chunk_size: int = 4096,
    limit: int | None = None,
) -> Iterator[pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for rec in iter_jsonl(path=path, limit=limit):
        rows.append(flatten_case_record(rec))
        if len(rows) >= chunk_size:
            yield pd.DataFrame(rows)
            rows.clear()
    if rows:
        yield pd.DataFrame(rows)


def load_cases_dataframe(
    path: str | Path,
    limit: int | None = None,
    chunk_size: int = 4096,
) -> pd.DataFrame:
    chunks = [chunk for chunk in iter_case_chunks(path=path, chunk_size=chunk_size, limit=limit)]
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def save_split_ids(split_ids: dict[str, list[str]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(split_ids, f, ensure_ascii=False, indent=2)


def load_split_ids(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Split IDs file must contain a dict")
    out: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        out[split] = [str(x) for x in loaded.get(split, [])]
    return out


def build_or_load_splits(
    df: pd.DataFrame,
    split_path: str | Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    force_rebuild: bool = False,
    logger: Any | None = None,
) -> dict[str, list[str]]:
    p = Path(split_path)
    if p.exists() and not force_rebuild:
        split_ids = load_split_ids(p)
        if logger is not None:
            logger.info(
                "Loaded existing splits from %s | train=%d val=%d test=%d",
                p,
                len(split_ids["train"]),
                len(split_ids["val"]),
                len(split_ids["test"]),
            )
        return split_ids

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    work = df[["case_id", "y"]].drop_duplicates("case_id")
    case_ids = work["case_id"].astype(str).values
    y = work["y"].astype(int).values

    test_plus_val = val_ratio + test_ratio
    stratify_arr = y if len(np.unique(y)) > 1 else None
    try:
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            case_ids,
            y,
            test_size=test_plus_val,
            random_state=seed,
            stratify=stratify_arr,
        )
    except ValueError:
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            case_ids,
            y,
            test_size=test_plus_val,
            random_state=seed,
            stratify=None,
        )

    val_fraction_of_temp = val_ratio / test_plus_val if test_plus_val > 0 else 0.5
    stratify_temp = temp_y if len(np.unique(temp_y)) > 1 else None
    try:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_fraction_of_temp),
            random_state=seed,
            stratify=stratify_temp,
        )
    except ValueError:
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1.0 - val_fraction_of_temp),
            random_state=seed,
            stratify=None,
        )

    split_ids = {
        "train": sorted([str(x) for x in train_ids]),
        "val": sorted([str(x) for x in val_ids]),
        "test": sorted([str(x) for x in test_ids]),
    }
    save_split_ids(split_ids, p)

    if logger is not None:
        logger.info(
            "Generated splits | train=%d val=%d test=%d",
            len(split_ids["train"]),
            len(split_ids["val"]),
            len(split_ids["test"]),
        )

    return split_ids


def apply_splits(df: pd.DataFrame, split_ids: dict[str, list[str]]) -> pd.DataFrame:
    split_map: dict[str, str] = {}
    for split_name, ids in split_ids.items():
        for cid in ids:
            split_map[str(cid)] = split_name

    out = df.copy()
    out["split"] = out["case_id"].astype(str).map(split_map)
    out = out[out["split"].notna()].copy()
    return out
