from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

UPDATED_BASE_TEXT_NODE_TYPES = ("preamble", "facts", "arguments")
UPDATED_ARGUMENT_ROLE_NODE_TYPES = (
    "petitioner_arguments",
    "respondent_arguments",
    "other_lawyer_arguments",
)
UPDATED_TEXT_NODE_TYPES = UPDATED_BASE_TEXT_NODE_TYPES + UPDATED_ARGUMENT_ROLE_NODE_TYPES
UPDATED_ENTITY_NODE_TYPES = (
    "petitioner",
    "respondent",
    "court",
    "judge",
    "petitioner_lawyer",
    "defence_lawyer",
    "lawyer",
    "statute",
    "provision",
    "precedent",
)
UPDATED_DEFAULT_NODE_TYPES = ("case",) + UPDATED_TEXT_NODE_TYPES + UPDATED_ENTITY_NODE_TYPES

FORBIDDEN_NODE_TYPES = frozenset({"org", "gpe", "date", "case_number"})
FORBIDDEN_EDGE_TYPES = frozenset(
    {
        ("statute", "used_in_arguments", "arguments"),
        ("provision", "used_in_arguments", "arguments"),
        ("judge", "presided_arguments", "arguments"),
        ("petitioner", "is_party_in_arguments", "arguments"),
        ("respondent", "is_party_in_arguments", "arguments"),
        ("petitioner_lawyer", "citation", "arguments"),
        ("defence_lawyer", "citation", "arguments"),
        ("lawyer", "citation", "arguments"),
    }
)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def apply_reasoning_graph_policy(graph_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(graph_cfg)
    respect_explicit_includes = bool(cfg.get("respect_explicit_includes", False))

    include_sections = list(cfg.get("include_sections") or UPDATED_TEXT_NODE_TYPES)
    if not respect_explicit_includes:
        include_sections = _ordered_unique(include_sections + list(UPDATED_TEXT_NODE_TYPES))
    cfg["include_sections"] = [section for section in include_sections if section in UPDATED_TEXT_NODE_TYPES]

    include_node_types = list(cfg.get("include_node_types") or UPDATED_DEFAULT_NODE_TYPES)
    if not respect_explicit_includes:
        include_node_types = _ordered_unique(include_node_types + list(UPDATED_DEFAULT_NODE_TYPES))
    cfg["include_node_types"] = [
        node_type
        for node_type in include_node_types
        if node_type in UPDATED_DEFAULT_NODE_TYPES and node_type not in FORBIDDEN_NODE_TYPES
    ]

    # Statute, provision, and precedent nodes are shared across cases to enable
    # legally meaningful cross-case connections (common cited law / common cited
    # case-law). Judge, lawyer, and court nodes stay local — they are identity
    # proxies that would create outcome-correlated shortcuts.
    # Temporal gating (precedent_year < case_year, statute_year <= case_year) is
    # enforced in case_star_builder before any citation edge is added.
    if cfg.get("disable_cross_case_sharing", False):
        cfg["shareable_node_types"] = []
    elif respect_explicit_includes and "shareable_node_types" in cfg:
        cfg["shareable_node_types"] = [
            node_type
            for node_type in cfg.get("shareable_node_types", [])
            if node_type in cfg["include_node_types"]
        ]
    else:
        cfg["shareable_node_types"] = ["statute", "provision", "precedent"]

    cfg["share_party_nodes"] = False
    cfg.setdefault("build_mode", "full_star_global")
    cfg.setdefault("use_to_undirected", True)
    cfg.setdefault("debug_sample_size", 3)
    raw_cache_name = str(cfg.get("cache_name") or "reasoning_focused_case_star_graph.pt")
    cache_path = Path(raw_cache_name)
    suffix = cache_path.suffix or ".pt"
    stem = cache_path.stem if cache_path.suffix else cache_path.name
    if "reasoning_focused" not in stem:
        stem = f"{stem}.reasoning_focused"
    cfg["cache_name"] = f"{stem}{suffix}"
    cfg["reasoning_graph_version"] = "reasoning_focused_v1"
    cfg["forbidden_node_types"] = sorted(FORBIDDEN_NODE_TYPES)
    cfg["forbidden_edge_types"] = ["|".join(edge_type) for edge_type in sorted(FORBIDDEN_EDGE_TYPES)]
    return cfg
