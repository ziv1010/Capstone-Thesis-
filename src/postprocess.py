from __future__ import annotations

import re
from typing import Any

from .utils import dedupe_str_list

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def build_ml_input_text(texts: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("facts_text", "arguments_petitioner", "arguments_respondent"):
        value = texts.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    return "\n\n".join(parts).strip()


def _paragraph_index_for_snippet(
    snippet: str,
    paragraphs: list[dict[str, Any]],
) -> int:
    needle = snippet.strip().casefold()
    if not needle:
        return -1

    for para in paragraphs:
        para_text = str(para.get("text", "")).casefold()
        para_idx = int(para.get("index", -1))
        if not para_text:
            continue
        if needle in para_text or para_text in needle:
            return para_idx
    return -1


def apply_leakage_firewall(
    case_record: dict[str, Any],
    paragraphs: list[dict[str, Any]],
    leakage_phrases: list[str],
) -> dict[str, Any]:
    ml = case_record.setdefault("ml", {})
    input_text = str(ml.get("input_text") or "")
    removed_spans = ml.get("removed_spans", [])
    if not isinstance(removed_spans, list):
        removed_spans = []

    phrases = [p.strip() for p in leakage_phrases if isinstance(p, str) and p.strip()]
    lowered_text = input_text.casefold()
    leaked = [phrase for phrase in phrases if phrase.casefold() in lowered_text]

    if not leaked:
        ml["leakage_flag"] = False
        ml["leaked_phrases_found"] = []
        ml["removed_spans"] = removed_spans
        ml["input_text"] = input_text.strip()
        case_record["ml"] = ml
        return case_record

    sentence_chunks = [chunk.strip() for chunk in SENTENCE_SPLIT_RE.split(input_text) if chunk.strip()]
    kept_chunks: list[str] = []
    actually_removed: bool = False

    for sentence in sentence_chunks:
        lowered_sentence = sentence.casefold()
        if any(phrase.casefold() in lowered_sentence for phrase in leaked):
            removed_spans.append(
                {
                    "paragraph_index": _paragraph_index_for_snippet(sentence, paragraphs),
                    "reason": "firewall",
                    "text": sentence,
                }
            )
            actually_removed = True
        else:
            kept_chunks.append(sentence)

    ml["input_text"] = "\n\n".join(kept_chunks).strip()
    ml["removed_spans"] = removed_spans
    ml["leakage_flag"] = actually_removed
    ml["leaked_phrases_found"] = dedupe_str_list(leaked)
    case_record["ml"] = ml

    return case_record
