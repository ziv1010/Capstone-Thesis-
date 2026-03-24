from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.graph.schema import CleanedCase, EntityMention, EntityRecord
from src.preprocessing.leakage import (
    compile_leakage_patterns,
    detect_leakage_spans,
    is_forbidden_annotation,
    remove_or_mask_leakage,
)
from src.preprocessing.normalize import (
    infer_petition_type,
    normalize_entity_name,
    normalize_whitespace,
    parse_provision_citation,
    stable_hash_to_unit_interval,
)

ALLOWED_ENTITY_LABELS = {
    "PETITIONER": "petitioner",
    "RESPONDENT": "respondent",
    "COURT": "court",
    "JUDGE": "judge",
    "LAWYER": "lawyer",
    "STATUTE": "statute",
    "PROVISION": "provision",
    "CASE_NUMBER": "case_number",
    "DATE": "date",
    "ORG": "org",
    "GPE": "gpe",
    "PRECEDENT": "precedent",
}

SECTION_NAME_MAP = {
    "PREAMBLE": "preamble",
    "facts": "facts",
    "arguments": "arguments",
}


def _annotation_section(annotation: dict[str, Any]) -> str | None:
    value = annotation.get("summary_section")
    return None if value is None else str(value)


def _aggregate_annotation_text(
    annotations: list[dict[str, Any]],
    target_sections: set[str],
    patterns: list[Any],
    mask_token: str,
) -> tuple[str, list[dict[str, Any]]]:
    chunks: list[str] = []
    masked_matches: list[dict[str, Any]] = []
    for annotation in annotations:
        section = _annotation_section(annotation)
        if section not in target_sections:
            continue
        text = normalize_whitespace(str(annotation.get("text", "") or ""))
        if not text:
            continue
        cleaned_text, matches = remove_or_mask_leakage(text, patterns, mask_token=mask_token)
        if cleaned_text:
            chunks.append(cleaned_text)
        masked_matches.extend(matches)
    return "\n".join(dict.fromkeys(chunks)), masked_matches


def extract_prejudgment_text(
    case_json: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    raw_result = case_json.get("raw_result", {}) or {}
    summary = raw_result.get("summary", {}) or {}
    raw_text = str(raw_result.get("data", {}).get("text", "") or "")
    preamble_end_offset = raw_result.get("data", {}).get("preamble_end_char_offset")

    patterns = compile_leakage_patterns(list(cfg.get("leakage_phrases", [])))
    mask_token = str(cfg.get("mask_token", "[LEAKAGE_MASK]"))
    annotations = raw_result.get("annotations", []) or []

    retained_texts: dict[str, str] = {"preamble": "", "facts": "", "arguments": ""}
    text_audit: dict[str, Any] = {"masked_matches": defaultdict(list), "sources": {}}

    preamble_source = str(summary.get("PREAMBLE", "") or "")
    if not preamble_source and cfg.get("retain_raw_preamble_fallback", True) and preamble_end_offset:
        preamble_source = raw_text[: int(preamble_end_offset)]
        text_audit["sources"]["preamble"] = "raw_result.data.text[:preamble_end_char_offset]"
    else:
        text_audit["sources"]["preamble"] = "raw_result.summary.PREAMBLE"

    for source_key, target_key in SECTION_NAME_MAP.items():
        value = normalize_whitespace(str(summary.get(source_key, "") or ""))
        if not value and cfg.get("fallback_to_annotation_text", True):
            fallback_text, matches = _aggregate_annotation_text(
                annotations=annotations,
                target_sections={source_key},
                patterns=patterns,
                mask_token=mask_token,
            )
            value = fallback_text
            if value:
                text_audit["sources"][target_key] = f"annotation_fallback:{source_key}"
            text_audit["masked_matches"][target_key].extend(matches)
        elif value:
            text_audit["sources"][target_key] = f"raw_result.summary.{source_key}"

        cleaned_value, matches = remove_or_mask_leakage(value, patterns, mask_token=mask_token)
        retained_texts[target_key] = normalize_whitespace(cleaned_value)
        text_audit["masked_matches"][target_key].extend(matches)

    cleaned_preamble, preamble_matches = remove_or_mask_leakage(
        normalize_whitespace(preamble_source),
        patterns,
        mask_token=mask_token,
    )
    retained_texts["preamble"] = normalize_whitespace(cleaned_preamble)
    text_audit["masked_matches"]["preamble"].extend(preamble_matches)

    text_audit["masked_matches"] = {
        section: values for section, values in text_audit["masked_matches"].items() if values
    }
    text_audit["retained_lengths"] = {key: len(value) for key, value in retained_texts.items()}
    return retained_texts, text_audit


def extract_entities_from_annotations(
    case_json: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[list[EntityRecord], dict[str, Any]]:
    raw_result = case_json.get("raw_result", {}) or {}
    annotations = raw_result.get("annotations", []) or []
    patterns = compile_leakage_patterns(list(cfg.get("leakage_phrases", [])))
    allowed_sections = {
        None if section is None else str(section)
        for section in cfg.get("allowed_annotation_sections", ["PREAMBLE", "facts", "arguments", None])
    }
    forbidden_sections = {
        str(section) for section in cfg.get("forbidden_annotation_sections", ["decision", "ANALYSIS", "issue"])
    }
    forbidden_labels = {str(label).upper() for label in cfg.get("forbidden_annotation_labels", ["RPC", "RLC"])}

    entity_index: dict[tuple[str, str], EntityRecord] = {}
    audit: dict[str, Any] = {
        "retained_annotations": 0,
        "ignored_annotations": 0,
        "ignored_entities": [],
    }

    for annotation in annotations:
        section = _annotation_section(annotation)
        forbidden, reasons, matches = is_forbidden_annotation(
            annotation=annotation,
            forbidden_sections=forbidden_sections,
            forbidden_labels=forbidden_labels,
            patterns=patterns,
        )
        if forbidden:
            audit["ignored_annotations"] += 1
            continue
        if section not in allowed_sections:
            audit["ignored_annotations"] += 1
            continue

        audit["retained_annotations"] += 1
        for entity in annotation.get("entities", []) or []:
            labels = [str(label).upper() for label in entity.get("labels", []) or []]
            if not labels:
                continue
            label = labels[0]
            if label not in ALLOWED_ENTITY_LABELS:
                audit["ignored_entities"].append(
                    {
                        "annotation_id": annotation.get("id"),
                        "entity_text": entity.get("text"),
                        "entity_label": label,
                    }
                )
                continue

            node_type = ALLOWED_ENTITY_LABELS[label]
            raw_name = normalize_whitespace(str(entity.get("text", "") or ""))
            candidate_name = normalize_whitespace(str(entity.get("normalized_name", "") or raw_name))
            canonical_name = normalize_entity_name(candidate_name, node_type)
            if not canonical_name:
                continue

            linked_statute: str | None = None
            if node_type == "provision":
                canonical_name, linked_statute = parse_provision_citation(candidate_name)
                canonical_name = normalize_entity_name(canonical_name, node_type)
                if linked_statute:
                    linked_statute = normalize_entity_name(linked_statute, "statute")

            mention = EntityMention(
                entity_type=node_type,
                raw_text=raw_name,
                canonical_text=canonical_name,
                section=section,
                annotation_id=str(annotation.get("id", "")),
                start=entity.get("start"),
                end=entity.get("end"),
            )

            key = (node_type, canonical_name)
            if key not in entity_index:
                entity_index[key] = EntityRecord(
                    entity_type=node_type,
                    raw_name=raw_name,
                    canonical_name=canonical_name,
                    mentions=[],
                    local_case_frequency=0,
                    first_seen_section=section,
                    seen_in_arguments=section == "arguments",
                    seen_in_preamble=section == "PREAMBLE",
                    linked_statute_canonical=linked_statute,
                )
            record = entity_index[key]
            record.mentions.append(mention)
            record.local_case_frequency += 1
            record.seen_in_arguments = record.seen_in_arguments or section == "arguments"
            record.seen_in_preamble = record.seen_in_preamble or section == "PREAMBLE"
            if record.first_seen_section is None:
                record.first_seen_section = section
            if linked_statute and not record.linked_statute_canonical:
                record.linked_statute_canonical = linked_statute

    entities = sorted(entity_index.values(), key=lambda item: (item.entity_type, item.canonical_name))
    return entities, audit


def _build_case_metadata(
    case_json: dict[str, Any],
    texts: dict[str, str],
    entities: list[EntityRecord],
) -> dict[str, Any]:
    entity_counts = defaultdict(int)
    entity_names: dict[str, list[str]] = defaultdict(list)
    for entity in entities:
        entity_counts[entity.entity_type] += 1
        entity_names[entity.entity_type].append(entity.canonical_name)

    case_year = None
    for entity in entities:
        if entity.entity_type == "date":
            digits = "".join(ch for ch in entity.raw_name if ch.isdigit())
            if len(digits) >= 4:
                case_year = int(digits[-4:])
                break

    petition_type = infer_petition_type(texts.get("preamble", ""))
    return {
        "respondent_count": entity_counts["respondent"],
        "judge_count": entity_counts["judge"],
        "lawyer_count": entity_counts["lawyer"],
        "statute_count": entity_counts["statute"],
        "provision_count": entity_counts["provision"],
        "precedent_count": entity_counts["precedent"],
        "court_count": entity_counts["court"],
        "preamble_length": len(texts.get("preamble", "")),
        "facts_length": len(texts.get("facts", "")),
        "arguments_length": len(texts.get("arguments", "")),
        "case_year": case_year,
        "petition_type": petition_type,
        "petition_type_hash": stable_hash_to_unit_interval(petition_type or "unknown"),
        "court_names": sorted(set(entity_names["court"])),
        "judge_names": sorted(set(entity_names["judge"])),
        "statute_names": sorted(set(entity_names["statute"])),
    }


def clean_case_for_gnn(
    case_json: dict[str, Any],
    file_path: str | Path,
    cfg: dict[str, Any],
) -> CleanedCase:
    patterns = compile_leakage_patterns(list(cfg.get("leakage_phrases", [])))
    leakage_audit = detect_leakage_spans(
        case_json=case_json,
        forbidden_top_level_fields=list(cfg.get("forbidden_top_level_fields", [])),
        forbidden_sections=list(cfg.get("forbidden_annotation_sections", [])),
        forbidden_labels=list(cfg.get("forbidden_annotation_labels", [])),
        patterns=patterns,
    )
    texts, text_audit = extract_prejudgment_text(case_json, cfg)
    entities, entity_audit = extract_entities_from_annotations(case_json, cfg)
    metadata = _build_case_metadata(case_json, texts, entities)
    leakage_audit["retained_texts"] = text_audit
    leakage_audit["entity_extraction"] = entity_audit

    path = Path(file_path)
    case_id = path.stem
    return CleanedCase(
        case_id=case_id,
        file_name=path.name,
        file_id=case_json.get("file_id"),
        internal_file_id=case_json.get("internal_file_id"),
        source_path=case_json.get("source_path"),
        raw_label=case_json.get("case_outcome_label"),
        texts=texts,
        metadata=metadata,
        entities=entities,
        leakage_audit=leakage_audit,
    )
