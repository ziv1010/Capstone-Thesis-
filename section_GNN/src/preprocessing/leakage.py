from __future__ import annotations

import re
from typing import Any


def compile_leakage_patterns(phrases: list[str]) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for phrase in phrases:
        escaped = re.escape(phrase.strip())
        escaped = escaped.replace(r"\ ", r"\s+")
        patterns.append(re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE))
    return patterns


def match_leakage_phrases(
    text: str,
    patterns: list[re.Pattern[str]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            matches.append(
                {
                    "pattern": pattern.pattern,
                    "matched_text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return matches


def remove_or_mask_leakage(
    text: str,
    patterns: list[re.Pattern[str]],
    mask_token: str = "[LEAKAGE_MASK]",
) -> tuple[str, list[dict[str, Any]]]:
    if not text:
        return "", []

    matches = match_leakage_phrases(text, patterns)
    cleaned = text
    for pattern in patterns:
        cleaned = pattern.sub(mask_token, cleaned)
    return cleaned, matches


def is_forbidden_annotation(
    annotation: dict[str, Any],
    forbidden_sections: set[str],
    forbidden_labels: set[str],
    patterns: list[re.Pattern[str]],
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    reasons: list[str] = []
    section = annotation.get("summary_section")
    if section is not None and str(section) in forbidden_sections:
        reasons.append(f"forbidden_section:{section}")

    label_values = {str(label).upper() for label in annotation.get("labels", []) or []}
    forbidden_label_hits = sorted(label_values.intersection(forbidden_labels))
    if forbidden_label_hits:
        reasons.append(f"forbidden_labels:{','.join(forbidden_label_hits)}")

    lowered_labels = {label.lower() for label in label_values}
    for label in lowered_labels:
        if any(term in label for term in ("decision", "disposition", "operative", "order", "rpc")):
            reasons.append(f"suspicious_label:{label}")

    text = str(annotation.get("text", "") or "")
    leakage_matches = match_leakage_phrases(text, patterns)
    if leakage_matches:
        reasons.append("explicit_leakage_phrase")

    return bool(reasons), reasons, leakage_matches


def detect_leakage_spans(
    case_json: dict[str, Any],
    forbidden_top_level_fields: list[str],
    forbidden_sections: list[str],
    forbidden_labels: list[str],
    patterns: list[re.Pattern[str]],
) -> dict[str, Any]:
    dropped_fields = [field for field in forbidden_top_level_fields if field in case_json]
    summary = case_json.get("raw_result", {}).get("summary", {}) or {}
    if "decision" in summary:
        dropped_fields.append("raw_result.summary.decision")

    dropped_annotations: list[dict[str, Any]] = []
    suspicious_annotation_count = 0
    for annotation in case_json.get("raw_result", {}).get("annotations", []) or []:
        forbidden, reasons, matches = is_forbidden_annotation(
            annotation=annotation,
            forbidden_sections=set(forbidden_sections),
            forbidden_labels={label.upper() for label in forbidden_labels},
            patterns=patterns,
        )
        if forbidden:
            suspicious_annotation_count += 1
            dropped_annotations.append(
                {
                    "annotation_id": annotation.get("id"),
                    "summary_section": annotation.get("summary_section"),
                    "labels": annotation.get("labels", []),
                    "reasons": reasons,
                    "matched_phrases": matches,
                    "text_preview": str(annotation.get("text", ""))[:240],
                }
            )

    decision_text = str(summary.get("decision", "") or "")
    decision_matches = match_leakage_phrases(decision_text, patterns)
    return {
        "fields_dropped": sorted(set(dropped_fields)),
        "annotations_dropped": dropped_annotations,
        "decision_text_removed": bool(decision_text),
        "decision_leakage_matches": decision_matches,
        "suspicious_annotation_count": suspicious_annotation_count,
    }
