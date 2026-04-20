from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.graph.case_star_builder import build_case_star_graph
from src.graph.global_graph_builder import merge_case_graphs_into_global_graph
from src.graph.pyg_builder import build_pyg_heterodata
from src.graph.schema import CleanedCase, EntityMention, EntityRecord
from src.preprocessing.leakage import compile_leakage_patterns, match_leakage_phrases
from src.training.dataset import PreparedCases, build_split_assignments, prepare_cases_for_task
from src.utils.io import list_json_files, load_json


def cleaned_case_from_dict(payload: dict[str, Any]) -> CleanedCase:
    entities = []
    for entity in payload.get("entities", []):
        mentions = [
            EntityMention(
                entity_type=str(mention.get("entity_type")),
                raw_text=str(mention.get("raw_text", "")),
                canonical_text=str(mention.get("canonical_text", "")),
                section=mention.get("section"),
                annotation_id=str(mention.get("annotation_id", "")),
                start=mention.get("start"),
                end=mention.get("end"),
            )
            for mention in entity.get("mentions", [])
        ]
        entities.append(
            EntityRecord(
                entity_type=str(entity.get("entity_type")),
                raw_name=str(entity.get("raw_name", "")),
                canonical_name=str(entity.get("canonical_name", "")),
                mentions=mentions,
                local_case_frequency=int(entity.get("local_case_frequency", len(mentions))),
                first_seen_section=entity.get("first_seen_section"),
                seen_in_arguments=bool(entity.get("seen_in_arguments", False)),
                seen_in_preamble=bool(entity.get("seen_in_preamble", False)),
                linked_statute_canonical=entity.get("linked_statute_canonical"),
            )
        )
    return CleanedCase(
        case_id=str(payload.get("case_id")),
        file_name=str(payload.get("file_name")),
        file_id=payload.get("file_id"),
        internal_file_id=payload.get("internal_file_id"),
        source_path=payload.get("source_path"),
        raw_label=payload.get("raw_label"),
        texts=dict(payload.get("texts", {})),
        metadata=dict(payload.get("metadata", {})),
        entities=entities,
        leakage_audit=dict(payload.get("leakage_audit", {})),
    )


def load_cleaned_cases(cleaned_case_dir: str | Path, limit: int | None = None) -> list[CleanedCase]:
    files = list_json_files(cleaned_case_dir)
    if limit is not None:
        files = files[:limit]
    return [cleaned_case_from_dict(load_json(path)) for path in files]


def assert_cleaned_case_integrity(cleaned_case: CleanedCase, preprocessing_cfg: dict[str, Any]) -> None:
    forbidden_text_keys = {"decision", "llm_case_outcome", "case_outcome_label"}
    if forbidden_text_keys.intersection(cleaned_case.texts.keys()):
        raise AssertionError(
            f"Forbidden text fields found in cleaned case {cleaned_case.case_id}: "
            f"{sorted(forbidden_text_keys.intersection(cleaned_case.texts.keys()))}"
        )

    patterns = compile_leakage_patterns(list(preprocessing_cfg.get("leakage_phrases", [])))
    for section_name, text in cleaned_case.texts.items():
        matches = match_leakage_phrases(text, patterns)
        if matches:
            raise AssertionError(
                f"Leakage phrase survived in case {cleaned_case.case_id} section {section_name}: {matches[:3]}"
            )


def build_graph_bundle(
    cleaned_cases: list[CleanedCase],
    cfg: dict[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    for cleaned_case in cleaned_cases:
        assert_cleaned_case_integrity(cleaned_case, cfg.get("preprocessing", {}))

    prepared: PreparedCases = prepare_cases_for_task(cleaned_cases, cfg.get("labels", {}))
    case_graphs = [build_case_star_graph(case, cfg.get("graph", {})) for case in prepared.cases]
    global_graph = merge_case_graphs_into_global_graph(case_graphs)
    split_assignments = build_split_assignments(prepared.cases, prepared.y, cfg.get("splits", {}))
    data, pyg_metadata = build_pyg_heterodata(
        global_graph=global_graph,
        cleaned_cases=prepared.cases,
        labels=prepared.y,
        label_names=prepared.label_names,
        split_assignments=split_assignments,
        cfg=cfg,
        logger=logger,
    )

    debug_sample_size = int(cfg.get("graph", {}).get("debug_sample_size", 3))
    debug_samples = []
    for case, case_graph in zip(prepared.cases[:debug_sample_size], case_graphs[:debug_sample_size]):
        debug_samples.append(
            {
                "case_id": case.case_id,
                "file_name": case.file_name,
                "retained_text": case.texts,
                "dropped_fields": case.leakage_audit.get("fields_dropped", []),
                "dropped_annotations": len(case.leakage_audit.get("annotations_dropped", [])),
                "node_counts": case_graph.metadata.get("node_count_by_type", {}),
                "edge_counts": case_graph.metadata.get("edge_count_by_type", {}),
            }
        )

    label_distribution = defaultdict(int)
    for case in prepared.cases:
        label_distribution[str(case.raw_label)] += 1

    if logger is not None:
        logger.info(
            "Prepared %d cases for task mode=%s and dropped %d cases",
            len(prepared.cases),
            cfg.get("labels", {}).get("task_mode", "binary"),
            len(prepared.dropped_case_ids),
        )

    return {
        "data": data,
        "metadata": {
            **pyg_metadata,
            "label_names": prepared.label_names,
            "dropped_case_ids": prepared.dropped_case_ids,
            "split_assignments": split_assignments,
            "case_graph_count": len(case_graphs),
            "raw_label_distribution_after_filtering": dict(sorted(label_distribution.items())),
            "global_node_stats": global_graph["node_stats"],
            "global_relation_stats": global_graph["relation_stats"],
            "debug_samples": debug_samples,
            "case_summaries": global_graph["case_summaries"],
        },
    }
