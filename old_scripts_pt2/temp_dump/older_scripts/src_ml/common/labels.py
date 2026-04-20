from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_label_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return NORM_RE.sub("_", text).strip("_")


@dataclass
class LabelMapper:
    canonical_order: list[str]
    source_priority: list[str]
    exact_map: dict[str, str]
    contains_map: dict[str, list[str]]
    default_label: str | None = None

    def _map_single_value(self, value: Any) -> str | None:
        normalized = normalize_label_value(value)
        if not normalized:
            return None

        if normalized in self.exact_map:
            return self.exact_map[normalized]

        for target, keywords in self.contains_map.items():
            for kw in keywords:
                kw_norm = normalize_label_value(kw)
                if kw_norm and kw_norm in normalized:
                    return target

        if normalized in set(self.canonical_order):
            return normalized
        return self.default_label

    def map_from_record(self, record: dict[str, Any]) -> str | None:
        for source_key in self.source_priority:
            value = record.get(source_key)
            mapped = self._map_single_value(value)
            if mapped is not None:
                return mapped
        return self.default_label


def build_label_mapper(label_cfg: dict[str, Any]) -> LabelMapper:
    canonical_order = [str(x) for x in label_cfg.get("canonical_order", [])]
    source_priority = [str(x) for x in label_cfg.get("source_priority", [])]
    default_label = label_cfg.get("default_label")
    default_label = str(default_label) if default_label is not None else None

    exact_map: dict[str, str] = {}
    for target, raw_values in (label_cfg.get("exact_map", {}) or {}).items():
        target_name = str(target)
        for raw in raw_values or []:
            exact_map[normalize_label_value(raw)] = target_name

    contains_map: dict[str, list[str]] = {}
    for target, raw_values in (label_cfg.get("contains_map", {}) or {}).items():
        contains_map[str(target)] = [str(v) for v in (raw_values or [])]

    return LabelMapper(
        canonical_order=canonical_order,
        source_priority=source_priority,
        exact_map=exact_map,
        contains_map=contains_map,
        default_label=default_label,
    )


def derive_model_labels(
    df: pd.DataFrame,
    label_cfg: dict[str, Any],
    logger: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    mapper = build_label_mapper(label_cfg)

    source_columns = {
        "decision": "decision",
        "outcome.winner": "outcome_winner",
        "outcome.label": "outcome_label",
    }

    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        mapped_record: dict[str, Any] = {}
        for source_name in mapper.source_priority:
            col = source_columns.get(source_name, source_name.replace(".", "_"))
            mapped_record[source_name] = row_dict.get(col)
        records.append(mapped_record)

    y_name = [mapper.map_from_record(rec) for rec in records]
    work_df = df.copy()
    work_df["y_name"] = y_name

    drop_unknown = bool(label_cfg.get("drop_unknown", True))
    dropped = int(work_df["y_name"].isna().sum())
    if drop_unknown:
        work_df = work_df[work_df["y_name"].notna()].copy()

    label_to_id = {name: idx for idx, name in enumerate(mapper.canonical_order)}
    work_df = work_df[work_df["y_name"].isin(label_to_id)].copy()
    dropped += int((~work_df["y_name"].isin(label_to_id)).sum())

    work_df["y"] = work_df["y_name"].map(label_to_id).astype(int)
    id_to_label = {idx: name for name, idx in label_to_id.items()}

    if logger is not None:
        logger.info(
            "Label mapping complete | kept=%d dropped=%d labels=%s",
            len(work_df),
            dropped,
            mapper.canonical_order,
        )

    return work_df, label_to_id, id_to_label
