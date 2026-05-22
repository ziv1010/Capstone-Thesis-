#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fixed_open_pipeline.preprocess_fixed_open import (  # noqa: E402
    DEFAULT_ENTITY_LABEL_MAP,
    _build_case_metadata,
    _build_text_sections,
    _derive_raw_label,
    _joined_unique,
    _lawyer_entity_type,
    _load_role_config,
    _normalized_role,
    _base_section_for_role,
)
from src.graph.schema import CleanedCase, EntityMention, EntityRecord  # noqa: E402
from src.preprocessing.normalize import normalize_entity_name, normalize_whitespace, parse_provision_citation  # noqa: E402
from src.utils.io import dump_json, ensure_dir, list_json_files, load_json, load_yaml  # noqa: E402
from src.utils.logging_utils import configure_logger  # noqa: E402
from src.utils.seed import set_global_seed  # noqa: E402


AUTHORITY_TYPES = {"statute", "provision", "precedent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert FIXED_OPEN sentence-level JSON files into section_GNN cleaned cases, "
            "using entity-resolution canonical_name/canonical_id fields when present."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _canonical_source_text(entity: dict[str, Any], node_type: str) -> tuple[str, str]:
    raw_name = normalize_whitespace(str(entity.get("text", "") or ""))
    canonical_name = normalize_whitespace(str(entity.get("canonical_name", "") or ""))

    # The resolved files add canonical_name for authority nodes. Use it for all
    # mapped types when available, but authority nodes are the main ablation axis.
    if canonical_name:
        return canonical_name, "resolved_canonical_name"
    return raw_name, "raw_text"


def _build_entities_resolved(
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
        "canonical_source_counts": Counter(),
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

            canonical_source, canonical_source_kind = _canonical_source_text(entity, node_type)
            audit["canonical_source_counts"][canonical_source_kind] += 1

            canonical_name = normalize_entity_name(canonical_source, node_type)
            linked_statute: str | None = None
            if node_type == "provision":
                canonical_name, linked_statute = parse_provision_citation(canonical_source)
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

    audit["canonical_source_counts"] = dict(sorted(audit["canonical_source_counts"].items()))
    entities = sorted(entity_index.values(), key=lambda item: (item.entity_type, item.canonical_name))
    return entities, audit


def _convert_payload_to_cleaned_case_resolved(
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
    entities, entity_audit = _build_entities_resolved(
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
        "source_format": "fixed_open_sentence_json_v1_entity_resolved",
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
    logger = configure_logger("preprocess_fixed_open_resolved", log_dir=log_dir)

    files = list_json_files(input_dir, pattern=str(cfg.get("data", {}).get("file_glob", "*.json")))
    files = [
        path
        for path in files
        if path.name != "report.json" and not path.name.endswith("__report.json")
    ]
    if args.limit is not None:
        files = files[: args.limit]
    logger.info("Preprocessing %d resolved FIXED_OPEN files from %s", len(files), input_dir)

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "num_files": len(files),
        "num_processed": 0,
        "num_skipped": 0,
        "label_field": str(preprocessing_cfg.get("label_field", "decision_label")),
        "label_distribution": Counter(),
        "sentence_role_distribution": Counter(),
        "canonical_source_distribution": Counter(),
        "skipped_cases": [],
        "cases": [],
    }
    skip_invalid_labels = bool(preprocessing_cfg.get("skip_missing_or_unmapped_labels", False))

    for index, path in enumerate(files, start=1):
        payload = load_json(path)
        try:
            cleaned_case = _convert_payload_to_cleaned_case_resolved(
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
        source_counts = cleaned_case.leakage_audit.get("entity_extraction", {}).get("canonical_source_counts", {})
        summary["canonical_source_distribution"].update(source_counts)
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
    summary["canonical_source_distribution"] = dict(sorted(summary["canonical_source_distribution"].items()))
    dump_json(summary, processed_dir / "preprocess_summary.fixed_open_resolved.json")
    logger.info(
        "Wrote %d cleaned cases to %s and skipped %d unlabeled/unmapped files",
        summary["num_processed"],
        cleaned_dir,
        summary["num_skipped"],
    )


if __name__ == "__main__":
    main()
