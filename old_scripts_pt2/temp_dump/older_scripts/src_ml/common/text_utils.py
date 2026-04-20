from __future__ import annotations

import re
from typing import Any, Iterable

import pandas as pd

WS_RE = re.compile(r"\s+")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return WS_RE.sub(" ", value).strip()
    if isinstance(value, list):
        parts = [safe_text(v) for v in value]
        return " ".join([p for p in parts if p]).strip()
    if isinstance(value, dict):
        return " ".join([safe_text(v) for v in value.values() if v is not None]).strip()
    return WS_RE.sub(" ", str(value)).strip()


def safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            txt = safe_text(item)
            if txt:
                out.append(txt)
        return out
    txt = safe_text(value)
    return [txt] if txt else []


def join_text_fields(row: dict[str, Any], fields: Iterable[str], sep: str = "\n\n") -> str:
    parts: list[str] = []
    for field in fields:
        value = row.get(field)
        text = safe_text(value)
        if text:
            parts.append(text)
    return sep.join(parts).strip()


def word_count(text: Any) -> int:
    s = safe_text(text)
    if not s:
        return 0
    return len(s.split())


def char_count(text: Any) -> int:
    return len(safe_text(text))


def text_length_summary(series: pd.Series) -> pd.DataFrame:
    chars = series.fillna("").map(char_count)
    words = series.fillna("").map(word_count)
    summary = pd.DataFrame(
        {
            "chars": chars.describe(),
            "words": words.describe(),
        }
    )
    return summary


def truncate_text(text: Any, max_chars: int) -> str:
    s = safe_text(text)
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars]
