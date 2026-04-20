from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TEXT_NODE_TYPES = ("preamble", "facts", "arguments")
ENTITY_NODE_TYPES = (
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
    "org",
    "gpe",
    "date",
    "case_number",
)
DEFAULT_NODE_TYPES = ("case",) + TEXT_NODE_TYPES + ENTITY_NODE_TYPES
ALLOWED_SUMMARY_FIELDS = ("PREAMBLE", "facts", "arguments")

FORBIDDEN_TOP_LEVEL_FIELDS = (
    "case_outcome_label",
    "case_outcome_score",
    "llm_case_outcome",
    "decision_text",
    "rpc_texts",
    "short_explanation",
    "raw_model_response",
)

FORBIDDEN_ANNOTATION_LABELS = ("RPC", "RLC")

DEFAULT_LEAKAGE_PHRASES = (
    "petition stands disposed of",
    "writ petition stands disposed of",
    "appeal is allowed",
    "appeal allowed",
    "petition is allowed",
    "petition allowed",
    "appeal dismissed",
    "petition dismissed",
    "writ petition dismissed",
    "matter remanded",
    "set aside",
    "quashed",
    "liberty granted",
    "tribunal shall entertain",
    "consequently, connected miscellaneous petition is closed",
    "without rejecting it on the ground of limitation",
)

RELATION_DEFINITIONS: dict[tuple[str, str, str], bool] = {
    ("case", "has_preamble", "preamble"): True,
    ("case", "has_facts", "facts"): True,
    ("case", "has_arguments", "arguments"): True,
    ("case", "has_petitioner", "petitioner"): True,
    ("case", "has_respondent", "respondent"): True,
    ("case", "heard_in", "court"): True,
    ("case", "decided_by_bench", "judge"): True,
    ("case", "has_lawyer", "lawyer"): True,
    ("case", "has_petitioner_lawyer", "petitioner_lawyer"): True,
    ("case", "has_defence_lawyer", "defence_lawyer"): True,
    ("case", "mentions_org", "org"): False,
    ("case", "mentions_gpe", "gpe"): False,
    ("case", "has_case_number", "case_number"): False,
    ("case", "has_date", "date"): False,
    ("arguments", "cites_statute", "statute"): True,
    ("arguments", "cites_provision", "provision"): True,
    ("arguments", "cites_precedent", "precedent"): False,
    ("provision", "belongs_to_statute", "statute"): True,
    # citation edges: lawyer -> arguments
    ("petitioner_lawyer", "citation", "arguments"): True,
    ("defence_lawyer",    "citation", "arguments"): True,
    # bridging edges: shorten hop distance
    ("provision",   "used_in_arguments", "arguments"): True,
    ("statute",     "used_in_arguments", "arguments"): True,
    ("petitioner",  "is_party_in_arguments", "arguments"): True,
    ("respondent",  "is_party_in_arguments", "arguments"): True,
    ("judge",       "presided_arguments", "arguments"): True,
}


@dataclass
class EntityMention:
    entity_type: str
    raw_text: str
    canonical_text: str
    section: str | None
    annotation_id: str
    start: int | None
    end: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EntityRecord:
    entity_type: str
    raw_name: str
    canonical_name: str
    mentions: list[EntityMention] = field(default_factory=list)
    local_case_frequency: int = 0
    first_seen_section: str | None = None
    seen_in_arguments: bool = False
    seen_in_preamble: bool = False
    linked_statute_canonical: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "raw_name": self.raw_name,
            "canonical_name": self.canonical_name,
            "mentions": [mention.to_dict() for mention in self.mentions],
            "local_case_frequency": self.local_case_frequency,
            "first_seen_section": self.first_seen_section,
            "seen_in_arguments": self.seen_in_arguments,
            "seen_in_preamble": self.seen_in_preamble,
            "linked_statute_canonical": self.linked_statute_canonical,
        }


@dataclass
class CleanedCase:
    case_id: str
    file_name: str
    file_id: str | None
    internal_file_id: str | None
    source_path: str | None
    raw_label: str | None
    texts: dict[str, str]
    metadata: dict[str, Any]
    entities: list[EntityRecord]
    leakage_audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "file_name": self.file_name,
            "file_id": self.file_id,
            "internal_file_id": self.internal_file_id,
            "source_path": self.source_path,
            "raw_label": self.raw_label,
            "texts": self.texts,
            "metadata": self.metadata,
            "entities": [entity.to_dict() for entity in self.entities],
            "leakage_audit": self.leakage_audit,
        }


@dataclass
class GraphNode:
    node_type: str
    node_key: str
    text: str
    metadata: dict[str, Any]
    share_across_cases: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    src_type: str
    relation: str
    dst_type: str
    src_key: str
    dst_key: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseGraphSpec:
    case_id: str
    file_name: str
    raw_label: str | None
    metadata: dict[str, Any]
    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "file_name": self.file_name,
            "raw_label": self.raw_label,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }
