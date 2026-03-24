from __future__ import annotations

import hashlib
import re
import string


WHITESPACE_RE = re.compile(r"\s+")
PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
HONORIFICS_RE = re.compile(
    r"\b(hon'?ble|justice|judge|chief justice|mr|mrs|ms|dr|shri|smt|advocate|sr\.?|jr\.?)\b",
    flags=re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def stable_hash_to_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def normalize_entity_name(text: str, entity_type: str) -> str:
    normalized = normalize_whitespace(text).lower()
    if entity_type in {"judge", "lawyer"}:
        normalized = HONORIFICS_RE.sub(" ", normalized)
        normalized = normalize_whitespace(normalized)
    if entity_type == "court":
        normalized = normalized.replace("high court of judicature at", "high court at")
        normalized = normalized.replace("high court of judicature for", "high court for")
        normalized = normalized.replace("the ", " ")
        normalized = normalize_whitespace(normalized)
    if entity_type in {"statute", "provision"}:
        normalized = normalized.replace("sec.", "section")
        normalized = normalized.replace("u/s", "under section")
        normalized = normalized.replace("cr.p.c.", "code of criminal procedure")
        normalized = normalize_whitespace(normalized)
    if entity_type in {"petitioner", "respondent", "org", "gpe", "precedent", "case_number", "date"}:
        normalized = normalize_whitespace(normalized)

    if entity_type not in {"case_number", "date", "provision"}:
        normalized = normalized.translate(PUNCT_TRANSLATION)
        normalized = normalize_whitespace(normalized)
    return normalized


def parse_provision_citation(normalized_name: str) -> tuple[str, str | None]:
    if "&&&" not in normalized_name:
        return normalize_whitespace(normalized_name), None
    provision_part, statute_part = normalized_name.split("&&&", maxsplit=1)
    return normalize_whitespace(provision_part), normalize_whitespace(statute_part)


def infer_petition_type(text: str) -> str | None:
    lowered = normalize_whitespace(text).lower()
    petition_markers = (
        "writ petition",
        "criminal petition",
        "anticipatory bail application",
        "bail application",
        "civil revision petition",
        "appeal",
        "special leave petition",
        "revision petition",
    )
    for marker in petition_markers:
        if marker in lowered:
            return marker
    return None
