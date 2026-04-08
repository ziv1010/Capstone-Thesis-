#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.schema import CleanedCase, EntityMention, EntityRecord
from src.preprocessing.leakage import compile_leakage_patterns, remove_or_mask_leakage
from src.preprocessing.normalize import (
    infer_petition_type,
    normalize_entity_name,
    normalize_whitespace,
    parse_provision_citation,
    stable_hash_to_unit_interval,
)
from src.utils.io import dump_json, ensure_dir, list_json_files, load_json, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed


DEFAULT_ENTITY_LABEL_MAP = {
    "PETITIONER": "petitioner",
    "RESPONDENT": "respondent",
    "COURT": "court",
    "JUDGE": "judge",
    "LAWYER": "lawyer",
    "STATUTE": "statute",
    "PROVISION": "provision",
    "PRECEDENT": "precedent",
    "ORG": "org",
    "GPE": "gpe",
    "DATE": "date",
    "CASE_NUMBER": "case_number",
}

DEFAULT_PREAMBLE_ROLES = ("PREAMBLE",)
DEFAULT_FACTS_ROLES = ("FAC",)
DEFAULT_PETITIONER_ARGUMENT_ROLES = ("ARG_PETITIONER",)
DEFAULT_RESPONDENT_ARGUMENT_ROLES = ("ARG_RESPONDENT",)
DEFAULT_OTHER_ARGUMENT_ROLES = ("PRE_RELIED", "PRE_NOT_RELIED", "STA")
DEFAULT_DROPPED_ROLES = ("ANALYSIS", "ISSUE", "NONE", "RATIO", "RLC", "RPC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FIXED_OPEN sentence-level JSON files into section_GNN cleaned cases."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "fixed_open_reasoning_config.yaml"),
    )
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _role_set(cfg: dict[str, Any], key: str, default: tuple[str, ...]) -> set[str]:
    return {
        normalize_whitespace(str(item)).upper()
        for item in cfg.get(key, default)
        if normalize_whitespace(str(item))
    }


def _load_role_config(cfg: dict[str, Any]) -> dict[str, set[str]]:
    return {
        "preamble": _role_set(cfg, "preamble_roles", DEFAULT_PREAMBLE_ROLES),
        "facts": _role_set(cfg, "facts_roles", DEFAULT_FACTS_ROLES),
        "petitioner_arguments": _role_set(
            cfg, "petitioner_argument_roles", DEFAULT_PETITIONER_ARGUMENT_ROLES
        ),
        "respondent_arguments": _role_set(
            cfg, "respondent_argument_roles", DEFAULT_RESPONDENT_ARGUMENT_ROLES
        ),
        "other_lawyer_arguments": _role_set(
            cfg, "other_argument_roles", DEFAULT_OTHER_ARGUMENT_ROLES
        ),
        "dropped": _role_set(cfg, "dropped_sentence_roles", DEFAULT_DROPPED_ROLES),
    }


def _normalized_role(value: Any) -> str:
    return normalize_whitespace(str(value or "")).upper()


def _joined_unique(chunks: list[str]) -> str:
    return "\n".join(dict.fromkeys(chunk for chunk in chunks if chunk))


def _mask_sentence_text(
    text: str,
    patterns: list[Any],
    mask_token: str,
) -> tuple[str, list[dict[str, Any]]]:
    cleaned, matches = remove_or_mask_leakage(normalize_whitespace(text), patterns, mask_token=mask_token)
    return normalize_whitespace(cleaned), matches


def _base_section_for_role(role: str, role_cfg: dict[str, set[str]]) -> str | None:
    if role in role_cfg["preamble"]:
        return "PREAMBLE"
    if role in role_cfg["facts"]:
        return "facts"
    if (
        role in role_cfg["petitioner_arguments"]
        or role in role_cfg["respondent_arguments"]
        or role in role_cfg["other_lawyer_arguments"]
    ):
        return "arguments"
    return None


def _lawyer_entity_type(role: str, mapped_type: str, role_cfg: dict[str, set[str]]) -> str:
    if mapped_type != "lawyer":
        return mapped_type
    if role in role_cfg["petitioner_arguments"]:
        return "petitioner_lawyer"
    if role in role_cfg["respondent_arguments"]:
        return "defence_lawyer"
    return "lawyer"


def _derive_raw_label(payload: dict[str, Any], cfg: dict[str, Any]) -> str:
    label_field = str(cfg.get("label_field", "decision_label"))
    if label_field not in payload:
        raise KeyError(f"Missing label field '{label_field}' in payload for {payload.get('file_id')!r}")

    raw_value = payload.get(label_field)
    value_key = str(raw_value)
    label_value_map = {
        str(key): str(value)
        for key, value in (cfg.get("label_value_map", {}) or {}).items()
    }
    if label_value_map:
        if value_key not in label_value_map:
            raise ValueError(
                f"Unmapped label value {raw_value!r} for {payload.get('file_id')!r}. "
                "Update preprocessing.label_value_map."
            )
        return label_value_map[value_key]
    return value_key


def _build_text_sections(
    payload: dict[str, Any],
    cfg: dict[str, Any],
    role_cfg: dict[str, set[str]],
) -> tuple[dict[str, str], dict[str, Any], Counter[str], Counter[str]]:
    patterns = compile_leakage_patterns(list(cfg.get("leakage_phrases", [])))
    mask_token = str(cfg.get("mask_token", "[LEAKAGE_MASK]"))
    use_preamble_end_offset = bool(cfg.get("use_preamble_end_offset", True))

    sections: dict[str, list[str]] = {
        "preamble": [],
        "facts": [],
        "arguments": [],
        "petitioner_arguments": [],
        "respondent_arguments": [],
        "other_lawyer_arguments": [],
    }
    sources: dict[str, str] = {}
    masked_matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kept_role_counts: Counter[str] = Counter()
    dropped_role_counts: Counter[str] = Counter()

    sentences = payload.get("sentences", []) or []
    preamble_end_offset = payload.get("preamble_end_char_offset")
    preamble_added = False
    if use_preamble_end_offset and isinstance(preamble_end_offset, int):
        for sentence in sentences:
            role = _normalized_role(sentence.get("rhetorical_role"))
            if role in role_cfg["dropped"]:
                continue
            start = sentence.get("start")
            end = sentence.get("end")
            if not isinstance(start, int) and not isinstance(end, int):
                continue
            if isinstance(end, int) and end <= preamble_end_offset:
                text, matches = _mask_sentence_text(sentence.get("text", ""), patterns, mask_token)
            elif isinstance(start, int) and start < preamble_end_offset:
                text, matches = _mask_sentence_text(sentence.get("text", ""), patterns, mask_token)
            else:
                continue
            if text:
                sections["preamble"].append(text)
                preamble_added = True
            masked_matches["preamble"].extend(matches)
        if preamble_added:
            sources["preamble"] = "sentence_prefix:preamble_end_char_offset"

    for sentence in sentences:
        role = _normalized_role(sentence.get("rhetorical_role"))
        if role in role_cfg["dropped"]:
            dropped_role_counts[role] += 1
            continue

        text = normalize_whitespace(str(sentence.get("text", "") or ""))
        if not text:
            continue

        target_sections: list[str] = []
        if role in role_cfg["facts"]:
            target_sections.append("facts")
        elif role in role_cfg["petitioner_arguments"]:
            target_sections.extend(["petitioner_arguments", "arguments"])
        elif role in role_cfg["respondent_arguments"]:
            target_sections.extend(["respondent_arguments", "arguments"])
        elif role in role_cfg["other_lawyer_arguments"]:
            target_sections.extend(["other_lawyer_arguments", "arguments"])
        elif role in role_cfg["preamble"] and not preamble_added:
            target_sections.append("preamble")
        elif role in role_cfg["preamble"]:
            continue
        else:
            dropped_role_counts[role] += 1
            continue

        cleaned_text, matches = _mask_sentence_text(text, patterns, mask_token)
        if not cleaned_text:
            continue

        kept_role_counts[role] += 1
        for section_name in target_sections:
            sections[section_name].append(cleaned_text)
            masked_matches[section_name].extend(matches)
            if section_name not in sources:
                if section_name == "preamble" and not preamble_added:
                    sources[section_name] = "sentence_roles:PREAMBLE"
                elif section_name == "facts":
                    sources[section_name] = "sentence_roles:FAC"
                elif section_name == "arguments":
                    sources[section_name] = "sentence_roles:ARG_* + PRE_* + STA"
                elif section_name == "petitioner_arguments":
                    sources[section_name] = "sentence_roles:ARG_PETITIONER"
                elif section_name == "respondent_arguments":
                    sources[section_name] = "sentence_roles:ARG_RESPONDENT"
                elif section_name == "other_lawyer_arguments":
                    sources[section_name] = "sentence_roles:PRE_RELIED + PRE_NOT_RELIED + STA"

    texts = {name: _joined_unique(chunks) for name, chunks in sections.items()}
    for section_name, section_text in list(texts.items()):
        cleaned_text, matches = _mask_sentence_text(section_text, patterns, mask_token)
        texts[section_name] = cleaned_text
        if matches:
            masked_matches[section_name].extend(matches)
    text_audit = {
        "sources": sources,
        "masked_matches": {name: values for name, values in masked_matches.items() if values},
        "retained_lengths": {name: len(text) for name, text in texts.items()},
        "kept_sentence_role_counts": dict(sorted(kept_role_counts.items())),
        "dropped_sentence_role_counts": dict(sorted(dropped_role_counts.items())),
    }
    return texts, text_audit, kept_role_counts, dropped_role_counts


def _build_entities(
    payload: dict[str, Any],
    case_id: str,
    role_cfg: dict[str, set[str]],
    cfg: dict[str, Any],
) -> tuple[list[EntityRecord], dict[str, Any]]:
    entity_label_map = {
        normalize_whitespace(str(key)).upper(): str(value)
        for key, value in (
            DEFAULT_ENTITY_LABEL_MAP
            | dict(cfg.get("entity_label_map", {}) or {})
            | dict(payload.get("entity_label_map", {}) or {})
        ).items()
    }
    entity_index: dict[tuple[str, str], EntityRecord] = {}
    audit: dict[str, Any] = {
        "retained_sentences": 0,
        "ignored_sentences": 0,
        "ignored_entities": [],
    }

    for sentence in payload.get("sentences", []) or []:
        role = _normalized_role(sentence.get("rhetorical_role"))
        base_section = _base_section_for_role(role, role_cfg)
        if base_section is None or role in role_cfg["dropped"]:
            audit["ignored_sentences"] += 1
            continue

        audit["retained_sentences"] += 1
        sentence_id = sentence.get("sentence_id")
        for entity_index_in_sentence, entity in enumerate(sentence.get("entities", []) or []):
            label = _normalized_role(entity.get("label"))
            mapped_type = entity_label_map.get(label)
            if mapped_type is None:
                audit["ignored_entities"].append(
                    {
                        "sentence_id": sentence_id,
                        "entity_text": entity.get("text"),
                        "entity_label": label,
                    }
                )
                continue

            node_type = _lawyer_entity_type(role, mapped_type, role_cfg)
            raw_name = normalize_whitespace(str(entity.get("text", "") or ""))
            if not raw_name:
                continue

            canonical_name = normalize_entity_name(raw_name, node_type)
            linked_statute: str | None = None
            if node_type == "provision":
                canonical_name, linked_statute = parse_provision_citation(raw_name)
                canonical_name = normalize_entity_name(canonical_name, node_type)
                if linked_statute:
                    linked_statute = normalize_entity_name(linked_statute, "statute")

            if not canonical_name:
                continue

            mention = EntityMention(
                entity_type=node_type,
                raw_text=raw_name,
                canonical_text=canonical_name,
                section=base_section,
                annotation_id=f"{case_id}:sentence_{sentence_id}:entity_{entity_index_in_sentence}",
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
                    first_seen_section=base_section,
                    seen_in_arguments=base_section == "arguments",
                    seen_in_preamble=base_section == "PREAMBLE",
                    linked_statute_canonical=linked_statute,
                )

            record = entity_index[key]
            record.mentions.append(mention)
            record.local_case_frequency += 1
            record.seen_in_arguments = record.seen_in_arguments or base_section == "arguments"
            record.seen_in_preamble = record.seen_in_preamble or base_section == "PREAMBLE"
            if record.first_seen_section is None:
                record.first_seen_section = base_section
            if linked_statute and not record.linked_statute_canonical:
                record.linked_statute_canonical = linked_statute

    entities = sorted(entity_index.values(), key=lambda item: (item.entity_type, item.canonical_name))
    return entities, audit


def _build_case_metadata(
    texts: dict[str, str],
    entities: list[EntityRecord],
    raw_label: str,
    label_field: str,
) -> dict[str, Any]:
    entity_counts: Counter[str] = Counter()
    entity_names: defaultdict[str, list[str]] = defaultdict(list)
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
        "lawyer_count": entity_counts["lawyer"]
        + entity_counts["petitioner_lawyer"]
        + entity_counts["defence_lawyer"],
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
        "source_label_field": label_field,
        "source_label_value": raw_label,
        "source_decision_label": raw_label,
    }


def _convert_payload_to_cleaned_case(
    payload: dict[str, Any],
    source_path: Path,
    preprocessing_cfg: dict[str, Any],
) -> CleanedCase:
    role_cfg = _load_role_config(preprocessing_cfg)
    label_field = str(preprocessing_cfg.get("label_field", "decision_label"))
    raw_label = _derive_raw_label(payload, preprocessing_cfg)
    texts, text_audit, kept_role_counts, dropped_role_counts = _build_text_sections(
        payload=payload,
        cfg=preprocessing_cfg,
        role_cfg=role_cfg,
    )
    entities, entity_audit = _build_entities(
        payload=payload,
        case_id=source_path.stem,
        role_cfg=role_cfg,
        cfg=preprocessing_cfg,
    )
    metadata = _build_case_metadata(
        texts=texts,
        entities=entities,
        raw_label=raw_label,
        label_field=label_field,
    )

    leakage_audit = {
        "source_format": "fixed_open_sentence_json_v1",
        "fields_dropped": [],
        "retained_texts": text_audit,
        "entity_extraction": entity_audit,
        "kept_sentence_role_counts": dict(sorted(kept_role_counts.items())),
        "dropped_sentence_role_counts": dict(sorted(dropped_role_counts.items())),
    }

    return CleanedCase(
        case_id=source_path.stem,
        file_name=source_path.name,
        file_id=str(payload.get("file_id") or source_path.stem),
        internal_file_id=None,
        source_path=str(source_path),
        raw_label=raw_label,
        texts=texts,
        metadata=metadata,
        entities=entities,
        leakage_audit=leakage_audit,
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths_cfg = cfg.get("paths", {})
    preprocessing_cfg = cfg.get("preprocessing", {})
    input_dir = Path(args.input_dir or paths_cfg.get("raw_json_dir"))
    cleaned_dir = ensure_dir(paths_cfg.get("cleaned_case_dir"))
    entity_dir = ensure_dir(paths_cfg.get("normalized_entity_dir"))
    audit_dir = ensure_dir(paths_cfg.get("audits_dir"))
    processed_dir = ensure_dir(paths_cfg.get("processed_dir"))
    output_dir = ensure_dir(paths_cfg.get("outputs_dir"))
    log_dir = ensure_dir(output_dir / "logs")
    logger = configure_logger("preprocess_fixed_open", log_dir=log_dir)

    files = list_json_files(input_dir, pattern=str(cfg.get("data", {}).get("file_glob", "*.json")))
    if args.limit is not None:
        files = files[: args.limit]
    logger.info("Preprocessing %d FIXED_OPEN files from %s", len(files), input_dir)

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "num_files": len(files),
        "num_processed": 0,
        "num_skipped": 0,
        "label_field": str(preprocessing_cfg.get("label_field", "decision_label")),
        "label_distribution": Counter(),
        "sentence_role_distribution": Counter(),
        "skipped_cases": [],
        "cases": [],
    }
    skip_invalid_labels = bool(preprocessing_cfg.get("skip_missing_or_unmapped_labels", False))

    for index, path in enumerate(files, start=1):
        payload = load_json(path)
        try:
            cleaned_case = _convert_payload_to_cleaned_case(
                payload=payload,
                source_path=path,
                preprocessing_cfg=preprocessing_cfg,
            )
        except (KeyError, ValueError) as exc:
            if not skip_invalid_labels:
                raise
            summary["num_skipped"] += 1
            summary["skipped_cases"].append(
                {
                    "case_id": path.stem,
                    "file_name": path.name,
                    "reason": str(exc),
                }
            )
            continue

        dump_json(cleaned_case.to_dict(), cleaned_dir / path.name)
        dump_json(
            {
                "case_id": cleaned_case.case_id,
                "entities": [entity.to_dict() for entity in cleaned_case.entities],
            },
            entity_dir / path.name,
        )
        dump_json(cleaned_case.leakage_audit, audit_dir / path.name)

        summary["num_processed"] += 1
        summary["label_distribution"][cleaned_case.raw_label] += 1
        for sentence in payload.get("sentences", []) or []:
            summary["sentence_role_distribution"][_normalized_role(sentence.get("rhetorical_role"))] += 1
        summary["cases"].append(
            {
                "case_id": cleaned_case.case_id,
                "raw_label": cleaned_case.raw_label,
                "retained_lengths": cleaned_case.leakage_audit.get("retained_texts", {}).get("retained_lengths", {}),
                "entity_count": len(cleaned_case.entities),
            }
        )

        if index % 500 == 0:
            logger.info("Processed %d/%d files", index, len(files))

    summary["label_distribution"] = dict(sorted(summary["label_distribution"].items()))
    summary["sentence_role_distribution"] = dict(sorted(summary["sentence_role_distribution"].items()))
    dump_json(summary, processed_dir / "preprocess_summary.fixed_open.json")
    logger.info(
        "Wrote %d cleaned cases to %s and skipped %d unlabeled/unmapped files",
        summary["num_processed"],
        cleaned_dir,
        summary["num_skipped"],
    )


if __name__ == "__main__":
    main()
