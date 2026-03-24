from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.graph.schema import CaseGraphSpec, CleanedCase, GraphEdge, GraphNode


def _should_include(node_type: str, cfg: dict[str, Any]) -> bool:
    include_node_types = set(cfg.get("include_node_types", []))
    return node_type in include_node_types


def _node_key(
    case_id: str,
    node_type: str,
    canonical_name: str,
    share_across_cases: bool,
) -> str:
    if share_across_cases:
        return f"{node_type}::{canonical_name}"
    return f"case::{case_id}::{node_type}::{canonical_name}"


def build_case_star_graph(cleaned_case: CleanedCase, cfg: dict[str, Any]) -> CaseGraphSpec:
    include_sections = set(cfg.get("include_sections", ["preamble", "facts", "arguments"]))
    include_node_types = set(cfg.get("include_node_types", []))
    shareable_node_types = set(cfg.get("shareable_node_types", []))
    share_party_nodes = bool(cfg.get("share_party_nodes", False))
    build_mode = str(cfg.get("build_mode", "full_star_global"))

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_keys: set[tuple[str, str]] = set()
    arguments_node_key: str | None = None

    case_key = f"case::{cleaned_case.case_id}"
    case_text = "\n\n".join(
        text for section, text in cleaned_case.texts.items() if section in include_sections and text
    )
    case_node = GraphNode(
        node_type="case",
        node_key=case_key,
        text=case_text,
        metadata={
            **cleaned_case.metadata,
            "case_id": cleaned_case.case_id,
            "file_name": cleaned_case.file_name,
            "raw_label": cleaned_case.raw_label,
            "text_sections": {key: cleaned_case.texts.get(key, "") for key in include_sections},
        },
        share_across_cases=False,
    )
    nodes.append(case_node)
    node_keys.add(("case", case_key))

    for section_name in ("preamble", "facts", "arguments"):
        if section_name not in include_sections or section_name not in include_node_types:
            continue
        text_value = cleaned_case.texts.get(section_name, "")
        if not text_value:
            continue
        section_key = f"case::{cleaned_case.case_id}::{section_name}"
        nodes.append(
            GraphNode(
                node_type=section_name,
                node_key=section_key,
                text=text_value,
                metadata={
                    "case_id": cleaned_case.case_id,
                    "section": section_name,
                    "text_length": len(text_value),
                },
                share_across_cases=False,
            )
        )
        node_keys.add((section_name, section_key))
        relation = f"has_{section_name}"
        edges.append(
            GraphEdge(
                src_type="case",
                relation=relation,
                dst_type=section_name,
                src_key=case_key,
                dst_key=section_key,
                metadata={"case_id": cleaned_case.case_id},
            )
        )
        if section_name == "arguments":
            arguments_node_key = section_key

    entity_lookup: dict[tuple[str, str], str] = {}
    deferred_statute_nodes: dict[str, dict[str, Any]] = {}

    if build_mode != "text_only":
        for entity in cleaned_case.entities:
            if entity.entity_type not in include_node_types:
                continue

            share_across_cases = entity.entity_type in shareable_node_types
            if entity.entity_type in {"petitioner", "respondent"} and share_party_nodes:
                share_across_cases = True
            if build_mode == "local_star_only":
                share_across_cases = False

            key = _node_key(
                case_id=cleaned_case.case_id,
                node_type=entity.entity_type,
                canonical_name=entity.canonical_name,
                share_across_cases=share_across_cases,
            )
            entity_lookup[(entity.entity_type, entity.canonical_name)] = key

            if (entity.entity_type, key) not in node_keys:
                nodes.append(
                    GraphNode(
                        node_type=entity.entity_type,
                        node_key=key,
                        text=entity.canonical_name,
                        metadata={
                            "case_id": cleaned_case.case_id,
                            "raw_name": entity.raw_name,
                            "canonical_name": entity.canonical_name,
                            "mention_count": entity.local_case_frequency,
                            "first_seen_section": entity.first_seen_section,
                            "seen_in_arguments": entity.seen_in_arguments,
                            "seen_in_preamble": entity.seen_in_preamble,
                            "linked_statute_canonical": entity.linked_statute_canonical,
                            "is_shared_node": share_across_cases,
                        },
                        share_across_cases=share_across_cases,
                    )
                )
                node_keys.add((entity.entity_type, key))

            relation = {
                "petitioner": "has_petitioner",
                "respondent": "has_respondent",
                "court": "heard_in",
                "judge": "decided_by_bench",
                "lawyer": "has_lawyer",
                "org": "mentions_org",
                "gpe": "mentions_gpe",
                "case_number": "has_case_number",
                "date": "has_date",
            }.get(entity.entity_type)
            if relation:
                edges.append(
                    GraphEdge(
                        src_type="case",
                        relation=relation,
                        dst_type=entity.entity_type,
                        src_key=case_key,
                        dst_key=key,
                        metadata={"case_id": cleaned_case.case_id},
                    )
                )

            if arguments_node_key and entity.seen_in_arguments:
                citation_relation = {
                    "statute": "cites_statute",
                    "provision": "cites_provision",
                    "precedent": "cites_precedent",
                }.get(entity.entity_type)
                if citation_relation:
                    edges.append(
                        GraphEdge(
                            src_type="arguments",
                            relation=citation_relation,
                            dst_type=entity.entity_type,
                            src_key=arguments_node_key,
                            dst_key=key,
                            metadata={"case_id": cleaned_case.case_id},
                        )
                    )

            if entity.entity_type == "provision" and entity.linked_statute_canonical:
                if "statute" not in include_node_types:
                    continue
                statute_key = entity_lookup.get(("statute", entity.linked_statute_canonical))
                if statute_key is None:
                    share_statute = "statute" in shareable_node_types and build_mode != "local_star_only"
                    statute_key = _node_key(
                        case_id=cleaned_case.case_id,
                        node_type="statute",
                        canonical_name=entity.linked_statute_canonical,
                        share_across_cases=share_statute,
                    )
                    entity_lookup[("statute", entity.linked_statute_canonical)] = statute_key
                    deferred_statute_nodes[statute_key] = {
                        "canonical_name": entity.linked_statute_canonical,
                        "share_across_cases": share_statute,
                    }
                edges.append(
                    GraphEdge(
                        src_type="provision",
                        relation="belongs_to_statute",
                        dst_type="statute",
                        src_key=key,
                        dst_key=statute_key,
                        metadata={"case_id": cleaned_case.case_id},
                    )
                )

    for statute_key, payload in deferred_statute_nodes.items():
        if ("statute", statute_key) in node_keys:
            continue
        nodes.append(
            GraphNode(
                node_type="statute",
                node_key=statute_key,
                text=payload["canonical_name"],
                metadata={
                    "case_id": cleaned_case.case_id,
                    "raw_name": payload["canonical_name"],
                    "canonical_name": payload["canonical_name"],
                    "mention_count": 0,
                    "first_seen_section": None,
                    "seen_in_arguments": True,
                    "seen_in_preamble": False,
                    "linked_statute_canonical": None,
                    "is_shared_node": payload["share_across_cases"],
                    "synthetic_from_provision": True,
                },
                share_across_cases=payload["share_across_cases"],
            )
        )
        node_keys.add(("statute", statute_key))

    relation_counts = defaultdict(int)
    for edge in edges:
        relation_counts[(edge.src_type, edge.relation, edge.dst_type)] += 1

    metadata = {
        **cleaned_case.metadata,
        "node_count_by_type": dict(sorted((node.node_type, 0) for node in nodes)),
        "edge_count_by_type": {
            "|".join(relation): count for relation, count in sorted(relation_counts.items())
        },
    }
    node_count_by_type = defaultdict(int)
    for node in nodes:
        node_count_by_type[node.node_type] += 1
    metadata["node_count_by_type"] = dict(sorted(node_count_by_type.items()))

    return CaseGraphSpec(
        case_id=cleaned_case.case_id,
        file_name=cleaned_case.file_name,
        raw_label=cleaned_case.raw_label,
        metadata=metadata,
        nodes=nodes,
        edges=edges,
    )
