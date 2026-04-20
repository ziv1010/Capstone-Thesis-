"""Normalise OpenNyAI outputs into stable JSON payloads."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


def unwrap_raw_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Support both direct OpenNyAI payloads and wrapped combined JSON files."""
    raw_result = payload.get("raw_result")
    if isinstance(raw_result, dict):
        return raw_result
    return payload


def get_summary_mapping(raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch whichever summary key is present for the current OpenNyAI version."""
    summary = raw_doc.get("summary")
    if isinstance(summary, dict):
        return summary
    summary = raw_doc.get("summaries")
    if isinstance(summary, dict):
        return summary
    return {}


def entity_label(entity: Dict[str, Any]) -> str | None:
    labels = entity.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    label = entity.get("label")
    if label is None:
        return None
    return str(label)


def rhetorical_role(annotation: Dict[str, Any]) -> str | None:
    labels = annotation.get("labels")
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return None


def normalise_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": entity.get("text", ""),
        "label": entity_label(entity),
        "start": entity.get("start"),
        "end": entity.get("end"),
    }


def iter_sentence_entities(annotation: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    entities = annotation.get("entities") or []
    if isinstance(entities, list):
        return entities
    return []


def build_combined_output(file_id: str, internal_id: str, source_path: str, raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve the exact OpenNyAI payload while adding stable external identifiers."""
    return {
        "file_id": file_id,
        "internal_file_id": internal_id,
        "source_path": source_path,
        "raw_result": raw_doc,
    }


def build_sentence_annotations_output(file_id: str, raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return cleaned sentence-level annotations for downstream tasks."""
    sentences = []
    for sentence_index, annotation in enumerate(raw_doc.get("annotations", []), start=1):
        sentences.append(
            {
                "sentence_id": sentence_index,
                "text": annotation.get("text", ""),
                "rhetorical_role": rhetorical_role(annotation),
                "in_summary": bool(annotation.get("in_summary", False)),
                "entities": [normalise_entity(entity) for entity in iter_sentence_entities(annotation)],
            }
        )
    return {"file_id": file_id, "sentences": sentences}


def _update_unique_entity_sets(entity: Dict[str, Any], unique_sets: Dict[str, set]) -> None:
    label = entity_label(entity)
    normalized_name = entity.get("normalized_name")
    text = entity.get("text", "")

    if label == "PROVISION":
        if isinstance(normalized_name, str) and "&&&" in normalized_name:
            provision_name, statute_name = [part.strip() for part in normalized_name.split("&&&", 1)]
            if provision_name:
                unique_sets["unique_provisions"].add(provision_name)
            if statute_name:
                unique_sets["unique_statutes"].add(statute_name)
        elif text:
            unique_sets["unique_provisions"].add(text)
    elif label == "STATUTE":
        unique_sets["unique_statutes"].add(str(normalized_name or text))
    elif label == "PRECEDENT":
        unique_sets["unique_precedents"].add(str(normalized_name or text))


def build_ner_output(file_id: str, raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return sentence-grouped entities plus flat and deduplicated views."""
    entities_by_sentence = []
    all_entities_flat = []
    unique_sets = {
        "unique_statutes": set(),
        "unique_provisions": set(),
        "unique_precedents": set(),
    }

    for sentence_index, annotation in enumerate(raw_doc.get("annotations", []), start=1):
        sentence_entities = []
        for entity in iter_sentence_entities(annotation):
            flattened_entity = normalise_entity(entity)
            flattened_entity["sentence_id"] = sentence_index
            normalized_name = entity.get("normalized_name")
            if normalized_name:
                flattened_entity["normalized_name"] = normalized_name
            all_entities_flat.append(flattened_entity)
            sentence_entities.append(normalise_entity(entity))
            _update_unique_entity_sets(entity, unique_sets)

        entities_by_sentence.append(
            {
                "sentence_id": sentence_index,
                "text": annotation.get("text", ""),
                "entities": sentence_entities,
            }
        )

    return {
        "file_id": file_id,
        "entities_by_sentence": entities_by_sentence,
        "all_entities_flat": all_entities_flat,
        "unique_statutes": sorted(value for value in unique_sets["unique_statutes"] if value),
        "unique_provisions": sorted(value for value in unique_sets["unique_provisions"] if value),
        "unique_precedents": sorted(value for value in unique_sets["unique_precedents"] if value),
    }


def build_rhetorical_roles_output(file_id: str, raw_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return rhetorical roles plus aggregate counts."""
    role_counts = Counter()
    sentences = []

    for annotation in raw_doc.get("annotations", []):
        role = rhetorical_role(annotation)
        if role:
            role_counts[role] += 1
        sentences.append(
            {
                "text": annotation.get("text", ""),
                "rhetorical_role": role,
            }
        )

    return {
        "file_id": file_id,
        "sentences": sentences,
        "role_counts": dict(role_counts),
    }


def render_summary_text(summary: Dict[str, Any]) -> str:
    """Render a readable plain-text summary view."""
    if not summary:
        return ""

    chunks = []
    for section_name, section_value in summary.items():
        chunks.append(f"[{section_name}]")
        if isinstance(section_value, str):
            chunks.append(section_value.strip())
        else:
            chunks.append(json.dumps(section_value, ensure_ascii=False, indent=2))
        chunks.append("")

    return "\n".join(chunks).strip() + "\n"


def build_summary_output(file_id: str, raw_doc: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Return both JSON and plain-text summary payloads."""
    summary = get_summary_mapping(raw_doc)
    return {"file_id": file_id, "summary": summary}, render_summary_text(summary)


def summary_sections_present(raw_doc: Dict[str, Any]) -> List[str]:
    """Convenience helper for the inspector script."""
    return list(get_summary_mapping(raw_doc).keys())
