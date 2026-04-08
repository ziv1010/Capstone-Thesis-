from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.graph.global_graph_builder import merge_case_graphs_into_global_graph
from src.graph.pyg_builder import build_pyg_heterodata
from src.training.dataset import PreparedCases, build_split_assignments, prepare_cases_for_task
from src.utils.pipeline import assert_cleaned_case_integrity, load_cleaned_cases

from .case_star_builder import build_case_star_graph
from .reasoning_graph_policy import apply_reasoning_graph_policy


def build_graph_bundle(
    cleaned_cases: list[Any],
    cfg: dict[str, Any],
    logger: Any | None = None,
) -> dict[str, Any]:
    for cleaned_case in cleaned_cases:
        assert_cleaned_case_integrity(cleaned_case, cfg.get("preprocessing", {}))

    prepared: PreparedCases = prepare_cases_for_task(cleaned_cases, cfg.get("labels", {}))
    graph_cfg = apply_reasoning_graph_policy(cfg.get("graph", {}))
    case_graphs = [build_case_star_graph(case, graph_cfg) for case in prepared.cases]
    global_graph = merge_case_graphs_into_global_graph(case_graphs)
    split_assignments = build_split_assignments(prepared.cases, prepared.y, cfg.get("splits", {}))
    data, pyg_metadata = build_pyg_heterodata(
        global_graph=global_graph,
        cleaned_cases=prepared.cases,
        labels=prepared.y,
        label_names=prepared.label_names,
        split_assignments=split_assignments,
        cfg={**cfg, "graph": graph_cfg},
        logger=logger,
    )

    debug_sample_size = int(graph_cfg.get("debug_sample_size", 3))
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
            "Prepared %d cases for reasoning-focused graph mode=%s and dropped %d cases",
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
            "effective_graph_config": graph_cfg,
        },
    }


__all__ = ["build_graph_bundle", "load_cleaned_cases"]
