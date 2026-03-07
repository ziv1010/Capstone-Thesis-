from __future__ import annotations

from typing import Any

from src_ml.common.text_utils import safe_text, truncate_text


def _query_text(record: dict[str, Any], max_chars: int) -> str:
    text = safe_text(record.get("ml_input_text"))
    if not text:
        parts = [
            safe_text(record.get("facts_text")),
            safe_text(record.get("arguments_petitioner")),
            safe_text(record.get("arguments_respondent")),
        ]
        text = "\n\n".join([p for p in parts if p]).strip()
    return truncate_text(text, max_chars=max_chars)


def build_rag_prompt(
    query_record: dict[str, Any],
    retrieved_items: list[dict[str, Any]],
    label_names: list[str],
    snippet_max_chars: int,
) -> str:
    query = _query_text(query_record, max_chars=snippet_max_chars)

    lines: list[str] = []
    lines.append("You are a legal outcome classifier.")
    lines.append("Use the query case text and retrieved similar TRAIN cases.")
    lines.append("Return STRICT JSON ONLY with keys:")
    lines.append('{"pred_label":"...","pred_winner":"...","confidence":0-100,"rationale":"...","cited_case_ids":["..."]}')
    lines.append(f"Allowed pred_label values: {label_names}")
    lines.append("")
    lines.append("QUERY_CASE_TEXT:")
    lines.append(query)
    lines.append("")
    lines.append("RETRIEVED_TRAIN_CASES:")

    for idx, item in enumerate(retrieved_items, start=1):
        snippet = safe_text(item.get("snippet"))
        if not snippet:
            snippet = safe_text(item.get("ml_input_text"))
        snippet = truncate_text(snippet, max_chars=snippet_max_chars)
        lines.append(f"[{idx}] case_id={item.get('case_id')}")
        lines.append(f"court={item.get('court')} year={item.get('year')}")
        lines.append(f"outcome_label={item.get('outcome_label')} outcome_winner={item.get('outcome_winner')}")
        lines.append(f"snippet={snippet}")
        lines.append("")

    return "\n".join(lines).strip()
