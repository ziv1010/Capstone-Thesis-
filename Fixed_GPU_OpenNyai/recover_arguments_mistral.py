#!/usr/bin/env python3
"""Recover missed party-argument sentences from RR-annotated court cases.

This script is a pre-stage for the existing side-refinement pipeline in
`extract_arguments_mistral.py`.

It does not replace `llm_arguments` and does not overwrite `rhetorical_role`.
Instead, it keeps the original RR `ARG_*` labels as the base, scans only
non-RR candidate sentences with local context, and asks whether each target is
a *missing* petitioner argument, a *missing* respondent argument, or neither.
It then writes a separate `llm_argument_recovery` block plus sentence-level
recovery labels.

Pipeline intent:

    RR -> contextual argument recovery -> side refinement -> final summaries

Usage example:

    python recover_arguments_mistral.py
    python recover_arguments_mistral.py --resume
    python recover_arguments_mistral.py --max_files 10 --dry_run
    python recover_arguments_mistral.py --overwrite
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ANNOTATIONS_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/"
    "outputs/family_matrimonial/annotations"
)

RECOVERY_KEY = "llm_argument_recovery"
RECOVERY_SCHEMA_VERSION = 2
MAX_PROMPT_CHARS = 18_000
HEADER_CHAR_RESERVE = 1_600
TARGET_SENTENCE_CHAR_LIMIT = 420
CONTEXT_SENTENCE_CHAR_LIMIT = 220
MIN_CANDIDATE_SENTENCES = 1

DEFAULT_CANDIDATE_SEED_ROLES = (
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "ANALYSIS",
    "ISSUE",
    "PRE_RELIED",
)
ARGUMENT_SEED_ROLES = {"ARG_PETITIONER", "ARG_RESPONDENT"}
SUBSTANTIVE_ROLES = {
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "ANALYSIS",
    "RATIO",
    "PRE_RELIED",
    "RPC",
}
RECOVERY_LABELS = (
    "PETITIONER_ARGUMENT",
    "RESPONDENT_ARGUMENT",
    "NOT_MISSING_ARGUMENT",
)

APPELLANT_SIDE_PATTERN = re.compile(
    r"\b(appellant|petitioner|plaintiff|claimant|applicant|revisionist)\b"
)
RESPONDENT_SIDE_PATTERN = re.compile(
    r"\b(respondent|defendant|opposite\s+part(?:y|ies)|state|prosecution|complainant)\b"
)
ARGUMENT_VERB_PATTERN = re.compile(
    r"\b(argue[sd]?|contend(?:ed|s)?|submit(?:ted|s)?|urge[sd]?|claim(?:ed|s)?|"
    r"assert(?:ed|s)?|state[sd]?|plead(?:ed|s)?|maintain(?:ed|s)?|disput(?:ed|es|ing)|"
    r"den(?:ied|ies|ying)|oppose[sd]?|counter\s+affidavit|reply\s+affidavit|"
    r"written\s+statement|rejoinder|stand|case)\b"
)
RESPONDENT_COUNSEL_PATTERN = re.compile(
    r"\b(learned\s+)?(panel\s+lawyer|government\s+advocate|public\s+prosecutor|"
    r"standing\s+counsel|state\s+counsel|aga|a\.g\.a\.)\b"
)

LABEL_ALIASES = {
    "MISSING_ARG_PETITIONER": "PETITIONER_ARGUMENT",
    "PETITIONER_ARGUMENT": "PETITIONER_ARGUMENT",
    "APPELLANT_ARGUMENT": "PETITIONER_ARGUMENT",
    "PETITIONER": "PETITIONER_ARGUMENT",
    "APPELLANT": "PETITIONER_ARGUMENT",
    "ARG_PETITIONER": "PETITIONER_ARGUMENT",
    "MISSING_ARG_RESPONDENT": "RESPONDENT_ARGUMENT",
    "RESPONDENT_ARGUMENT": "RESPONDENT_ARGUMENT",
    "RESPONDENT": "RESPONDENT_ARGUMENT",
    "ARG_RESPONDENT": "RESPONDENT_ARGUMENT",
    "NOT_MISSING_ARGUMENT": "NOT_MISSING_ARGUMENT",
    "NOT_MISSING": "NOT_MISSING_ARGUMENT",
    "NEITHER": "NOT_MISSING_ARGUMENT",
    "KEEP": "NOT_MISSING_ARGUMENT",
    "KEEP_AS_IS": "NOT_MISSING_ARGUMENT",
    "LEAVE_ALONE": "NOT_MISSING_ARGUMENT",
    "NOT_ARGUMENT": "NOT_MISSING_ARGUMENT",
    "NO_ARGUMENT": "NOT_MISSING_ARGUMENT",
    "OTHER": "NOT_MISSING_ARGUMENT",
    "NOT_USEFUL": "NOT_MISSING_ARGUMENT",
    "NONE": "NOT_MISSING_ARGUMENT",
    "UNCLEAR": "NOT_MISSING_ARGUMENT",
    "COURT_REASONING": "NOT_MISSING_ARGUMENT",
    "COURT_ANALYSIS": "NOT_MISSING_ARGUMENT",
    "ANALYSIS": "NOT_MISSING_ARGUMENT",
    "REASONING": "NOT_MISSING_ARGUMENT",
    "FACT": "NOT_MISSING_ARGUMENT",
    "FACTS": "NOT_MISSING_ARGUMENT",
    "BACKGROUND": "NOT_MISSING_ARGUMENT",
}

SYSTEM_PROMPT = """\
You are a legal missing-argument detector for Indian court judgments.

You will receive TARGET sentences and short local CONTEXT around each target.
Your job is to classify each TARGET sentence only. CONTEXT lines are support
and must not be classified.

Use exactly one label per TARGET sentence:
- MISSING_ARG_PETITIONER
- MISSING_ARG_RESPONDENT
- NOT_MISSING_ARGUMENT

Interpretation rules:
- MISSING_ARG_PETITIONER: the TARGET sentence is not already an RR
  ARG_PETITIONER sentence, but it should be added to the petitioner/appellant
  argument bucket.
- MISSING_ARG_RESPONDENT: the TARGET sentence is not already an RR
  ARG_RESPONDENT sentence, but it should be added to the respondent/State
  argument bucket.
- NOT_MISSING_ARGUMENT: the TARGET sentence should not be added to either RR
  argument bucket.

Important:
- The original RR ARG_* labels are the base system and should be preserved.
- TARGET sentences are only possible *missing* arguments beyond that RR base.
- Do not try to relabel facts, reasoning, or other rhetorical roles.
- If a sentence is ambiguous, choose NOT_MISSING_ARGUMENT.
- Use local context to detect when a sentence continues the same party's
  submission from nearby RR argument sentences.
- Do not write summaries. Do not merge sentences. Classify each shown TARGET
  sentence individually.

Return JSON only with this schema:
{
  "classifications": [
    {
      "sentence_id": 42,
      "label": "MISSING_ARG_RESPONDENT",
      "confidence": "high",
      "reason": "Short reason."
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def truncate_text(text: str, max_chars: int) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return normalized[: max_chars - 3].rstrip() + "..."


def count_substantive_sentences(sentences: list[dict[str, Any]]) -> int:
    return sum(
        1
        for sentence in sentences
        if str(sentence.get("rhetorical_role") or "NONE") in SUBSTANTIVE_ROLES
    )


def coerce_sentence_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe_preserve_order(items: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    ordered: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def detect_party_cue(text: str) -> str:
    normalized = normalize_whitespace(text).lower()
    has_argument_verb = bool(ARGUMENT_VERB_PATTERN.search(normalized))

    appellant_match = bool(APPELLANT_SIDE_PATTERN.search(normalized)) and has_argument_verb
    respondent_match = bool(RESPONDENT_SIDE_PATTERN.search(normalized)) and has_argument_verb

    if not respondent_match and RESPONDENT_COUNSEL_PATTERN.search(normalized) and has_argument_verb:
        respondent_match = True

    if appellant_match and respondent_match:
        return "BOTH"
    if appellant_match:
        return "APPELLANT"
    if respondent_match:
        return "RESPONDENT"
    return "NONE"


def parse_role_list(value: str) -> tuple[str, ...]:
    roles = [normalize_whitespace(part).upper() for part in value.split(",") if normalize_whitespace(part)]
    return tuple(dedupe_preserve_order(roles))


# ---------------------------------------------------------------------------
# Candidate expansion
# ---------------------------------------------------------------------------

def build_candidate_pool(
    *,
    sentences: list[dict[str, Any]],
    candidate_seed_roles: set[str],
    neighbor_window: int,
    include_cue_matched_sentences: bool,
) -> list[dict[str, Any]]:
    sentence_rows: list[dict[str, Any]] = []
    argument_seed_indices: set[int] = set()

    for index, sentence in enumerate(sentences):
        sentence_id = coerce_sentence_id(sentence.get("sentence_id"))
        text = normalize_whitespace(str(sentence.get("text", "") or ""))
        if sentence_id is None or not text:
            continue
        role = str(sentence.get("rhetorical_role") or "NONE").upper()
        cue = detect_party_cue(text)
        row = {
            "index": index,
            "sentence_id": sentence_id,
            "text": text,
            "original_rhetorical_role": role,
            "cue": cue,
        }
        sentence_rows.append(row)
        if role in ARGUMENT_SEED_ROLES:
            argument_seed_indices.add(index)

    neighbor_indices: set[int] = set()
    for index in argument_seed_indices:
        for nearby_index in range(
            max(0, index - neighbor_window),
            min(len(sentences), index + neighbor_window + 1),
        ):
            neighbor_indices.add(nearby_index)

    candidates_by_id: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for row in sentence_rows:
        sources: list[str] = []
        role = row["original_rhetorical_role"]
        if role in ARGUMENT_SEED_ROLES:
            # Original RR argument sentences remain the base system; recovery
            # only targets possible missing argument sentences beyond that base.
            continue
        if role in candidate_seed_roles:
            sources.append(f"seed_role:{role}")
        if row["index"] in neighbor_indices:
            sources.append("neighbor_of_rr_argument")
        if include_cue_matched_sentences and row["cue"] != "NONE":
            sources.append(f"cue:{row['cue'].lower()}")
        if not sources:
            continue

        existing = candidates_by_id.get(row["sentence_id"])
        if existing is None:
            candidates_by_id[row["sentence_id"]] = {
                **row,
                "candidate_sources": dedupe_preserve_order(sources),
            }
            continue

        existing["candidate_sources"] = dedupe_preserve_order(
            list(existing.get("candidate_sources", [])) + sources
        )

    return list(candidates_by_id.values())


def build_candidate_block(
    *,
    candidate: dict[str, Any],
    sentences: list[dict[str, Any]],
    local_context_window: int,
) -> str:
    target_index = int(candidate["index"])
    lines: list[str] = [
        f"TARGET [{candidate['sentence_id']}] "
        f"orig={candidate['original_rhetorical_role']} "
        f"cue={candidate['cue']} "
        f"sources={','.join(candidate['candidate_sources'])}",
    ]

    for offset in range(-local_context_window, local_context_window + 1):
        context_index = target_index + offset
        if context_index < 0 or context_index >= len(sentences):
            continue
        context_sentence = sentences[context_index]
        context_id = coerce_sentence_id(context_sentence.get("sentence_id"))
        context_text = normalize_whitespace(str(context_sentence.get("text", "") or ""))
        if context_id is None or not context_text:
            continue
        context_role = str(context_sentence.get("rhetorical_role") or "NONE").upper()
        context_cue = detect_party_cue(context_text)

        if offset == 0:
            lines.append(
                "TEXT "
                f"[{context_id}] role={context_role} cue={context_cue} "
                f"{truncate_text(context_text, TARGET_SENTENCE_CHAR_LIMIT)}"
            )
            continue

        direction = "PREV" if offset < 0 else "NEXT"
        lines.append(
            f"CONTEXT_{direction}{abs(offset)} "
            f"[{context_id}] role={context_role} cue={context_cue} "
            f"{truncate_text(context_text, CONTEXT_SENTENCE_CHAR_LIMIT)}"
        )

    return "\n".join(lines)


def build_prompt_chunks(
    *,
    file_id: str,
    sentences: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    local_context_window: int,
    max_prompt_chars: int,
) -> list[dict[str, Any]]:
    blocks = [
        {
            "sentence_id": candidate["sentence_id"],
            "block_text": build_candidate_block(
                candidate=candidate,
                sentences=sentences,
                local_context_window=local_context_window,
            ),
        }
        for candidate in candidates
    ]

    chunks: list[list[dict[str, Any]]] = []
    current_chunk: list[dict[str, Any]] = []
    current_chars = 0
    max_body_chars = max(2_000, max_prompt_chars - HEADER_CHAR_RESERVE)

    for block in blocks:
        block_chars = len(block["block_text"]) + 2
        if current_chunk and current_chars + block_chars > max_body_chars:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(block)
        current_chars += block_chars

    if current_chunk:
        chunks.append(current_chunk)

    rendered_chunks: list[dict[str, Any]] = []
    total_chunks = len(chunks)
    for chunk_index, chunk in enumerate(chunks, start=1):
        shown_candidate_ids = [block["sentence_id"] for block in chunk]
        lines = [
            f"Case ID: {file_id}",
            f"Chunk: {chunk_index} of {total_chunks}",
            "",
            "Contextual argument recovery over an expanded candidate pool.",
            "Original RR ARG_* sentences remain the base system.",
            "Classify every TARGET sentence only as missing petitioner argument, missing respondent argument, or not missing argument.",
            "CONTEXT lines are support only and must not be classified.",
            "",
            f"Total candidate sentences in case: {len(candidates)}",
            f"Candidate sentences in this chunk: {len(shown_candidate_ids)}",
            f"Target sentence IDs in this chunk: {shown_candidate_ids}",
            "",
            "Return exactly one label per TARGET sentence using:",
            "MISSING_ARG_PETITIONER, MISSING_ARG_RESPONDENT, NOT_MISSING_ARGUMENT",
            "",
            "Candidate blocks:",
            "",
        ]
        for block in chunk:
            lines.append(block["block_text"])
            lines.append("")
        rendered_chunks.append(
            {
                "prompt_text": "\n".join(lines).strip(),
                "shown_candidate_ids": shown_candidate_ids,
            }
        )

    return rendered_chunks


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start: end + 1])


def normalize_label(value: Any) -> Optional[str]:
    normalized = normalize_whitespace(str(value or "")).upper()
    return LABEL_ALIASES.get(normalized)


def normalize_confidence(value: Any) -> str:
    normalized = normalize_whitespace(str(value or "")).lower()
    if normalized in {"high", "medium", "low"}:
        return normalized
    return "medium"


def fallback_label_for_candidate(candidate: dict[str, Any]) -> str:
    role = str(candidate.get("original_rhetorical_role") or "NONE").upper()
    if role == "ARG_PETITIONER":
        return "PETITIONER_ARGUMENT"
    if role == "ARG_RESPONDENT":
        return "RESPONDENT_ARGUMENT"
    return "NOT_MISSING_ARGUMENT"


def build_fallback_classification(
    candidate: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "sentence_id": candidate["sentence_id"],
        "text": candidate["text"],
        "original_rhetorical_role": candidate["original_rhetorical_role"],
        "cue": candidate["cue"],
        "candidate_sources": list(candidate["candidate_sources"]),
        "predicted_label": fallback_label_for_candidate(candidate),
        "confidence": "low",
        "reason": reason,
    }


def iter_raw_classifications(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("classifications", "predictions", "results", "labels"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def normalize_output(
    payload: dict[str, Any],
    *,
    shown_candidate_ids: list[int],
    candidate_lookup: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_classifications = iter_raw_classifications(payload)
    shown_candidate_id_set = set(shown_candidate_ids)

    by_id: dict[int, dict[str, Any]] = {}
    for item in raw_classifications:
        sentence_id = coerce_sentence_id(item.get("sentence_id"))
        if sentence_id is None or sentence_id not in shown_candidate_id_set:
            continue
        candidate = candidate_lookup.get(sentence_id)
        if candidate is None:
            continue

        label = normalize_label(item.get("label"))
        if label is None:
            continue

        by_id[sentence_id] = {
            "sentence_id": sentence_id,
            "text": candidate["text"],
            "original_rhetorical_role": candidate["original_rhetorical_role"],
            "cue": candidate["cue"],
            "candidate_sources": list(candidate["candidate_sources"]),
            "predicted_label": label,
            "confidence": normalize_confidence(item.get("confidence")),
            "reason": truncate_text(str(item.get("reason", "") or "").strip() or "Model classification.", 160),
        }

    normalized: list[dict[str, Any]] = []
    for sentence_id in shown_candidate_ids:
        candidate = candidate_lookup[sentence_id]
        normalized.append(
            by_id.get(sentence_id)
            or build_fallback_classification(
                candidate,
                reason="Missing from model output; conservative fallback applied.",
            )
        )
    return normalized


def aggregate_chunk_results(
    *,
    sentences: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    chunk_results: list[dict[str, Any]],
) -> dict[str, Any]:
    sentence_order = {}
    sentence_lookup: dict[int, dict[str, Any]] = {}
    original_rr_appellant_ids: list[int] = []
    original_rr_respondent_ids: list[int] = []
    for index, sentence in enumerate(sentences, start=1):
        sentence_id = coerce_sentence_id(sentence.get("sentence_id"))
        if sentence_id is None:
            continue
        sentence_order[sentence_id] = index
        sentence_lookup[sentence_id] = sentence
        role = str(sentence.get("rhetorical_role") or "NONE").upper()
        if role == "ARG_PETITIONER":
            original_rr_appellant_ids.append(sentence_id)
        elif role == "ARG_RESPONDENT":
            original_rr_respondent_ids.append(sentence_id)

    candidate_lookup = {candidate["sentence_id"]: candidate for candidate in candidates}
    classification_by_id: dict[int, dict[str, Any]] = {}
    raw_model_responses: list[dict[str, Any]] = []
    errors: list[str] = []

    for chunk_index, chunk_result in enumerate(chunk_results, start=1):
        raw_model_responses.append(
            {
                "chunk_index": chunk_index,
                "shown_candidate_ids": chunk_result["shown_candidate_ids"],
                "raw_model_response": chunk_result.get("raw_model_response", ""),
            }
        )
        if chunk_result.get("error"):
            errors.append(str(chunk_result["error"]))
        for classification in chunk_result["classifications"]:
            classification_by_id[classification["sentence_id"]] = classification

    base_classifications: list[dict[str, Any]] = []
    for sentence_id in sorted(original_rr_appellant_ids, key=lambda sid: sentence_order.get(sid, sid)):
        sentence = sentence_lookup[sentence_id]
        base_classifications.append(
            {
                "sentence_id": sentence_id,
                "text": normalize_whitespace(str(sentence.get("text", "") or "")),
                "original_rhetorical_role": "ARG_PETITIONER",
                "cue": detect_party_cue(str(sentence.get("text", "") or "")),
                "candidate_sources": ["rr_base_argument"],
                "predicted_label": "PETITIONER_ARGUMENT",
                "confidence": "high",
                "reason": "Original RR petitioner argument kept as base.",
            }
        )
    for sentence_id in sorted(original_rr_respondent_ids, key=lambda sid: sentence_order.get(sid, sid)):
        sentence = sentence_lookup[sentence_id]
        base_classifications.append(
            {
                "sentence_id": sentence_id,
                "text": normalize_whitespace(str(sentence.get("text", "") or "")),
                "original_rhetorical_role": "ARG_RESPONDENT",
                "cue": detect_party_cue(str(sentence.get("text", "") or "")),
                "candidate_sources": ["rr_base_argument"],
                "predicted_label": "RESPONDENT_ARGUMENT",
                "confidence": "high",
                "reason": "Original RR respondent argument kept as base.",
            }
        )

    combined_classification_by_id = {
        item["sentence_id"]: item for item in base_classifications
    }
    combined_classification_by_id.update(classification_by_id)

    ordered_classifications = sorted(
        combined_classification_by_id.values(),
        key=lambda item: sentence_order.get(item["sentence_id"], item["sentence_id"]),
    )

    label_counts = Counter(item["predicted_label"] for item in ordered_classifications)
    petitioner_candidate_ids = [
        item["sentence_id"]
        for item in ordered_classifications
        if item["predicted_label"] == "PETITIONER_ARGUMENT"
    ]
    respondent_candidate_ids = [
        item["sentence_id"]
        for item in ordered_classifications
        if item["predicted_label"] == "RESPONDENT_ARGUMENT"
    ]
    not_missing_candidate_ids = [
        item["sentence_id"]
        for item in ordered_classifications
        if item["predicted_label"] == "NOT_MISSING_ARGUMENT"
    ]

    newly_recovered_petitioner_ids = [
        sentence_id
        for sentence_id in petitioner_candidate_ids
        if sentence_id in candidate_lookup
        and candidate_lookup[sentence_id]["original_rhetorical_role"] != "ARG_PETITIONER"
    ]
    newly_recovered_respondent_ids = [
        sentence_id
        for sentence_id in respondent_candidate_ids
        if sentence_id in candidate_lookup
        and candidate_lookup[sentence_id]["original_rhetorical_role"] != "ARG_RESPONDENT"
    ]

    return {
        "original_rr_appellant_ids": sorted(original_rr_appellant_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "original_rr_respondent_ids": sorted(original_rr_respondent_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "petitioner_candidate_ids": sorted(petitioner_candidate_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "respondent_candidate_ids": sorted(respondent_candidate_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "not_missing_candidate_ids": sorted(not_missing_candidate_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "newly_recovered_petitioner_ids": sorted(newly_recovered_petitioner_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "newly_recovered_respondent_ids": sorted(newly_recovered_respondent_ids, key=lambda sid: sentence_order.get(sid, sid)),
        "classifications": ordered_classifications,
        "label_counts": dict(sorted(label_counts.items())),
        "reclassified_original_argument_sentences": [],
        "raw_model_responses": raw_model_responses,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Annotation updates
# ---------------------------------------------------------------------------

def clear_llm_argument_recovery_fields(annotation: dict[str, Any]) -> None:
    for sentence in annotation.get("sentences", []):
        if not isinstance(sentence, dict):
            continue
        sentence.pop("llm_argument_recovery_label", None)
        sentence.pop("llm_argument_recovery_confidence", None)
        sentence.pop("llm_argument_recovery_reason", None)
        sentence.pop("llm_argument_recovery_sources", None)


def apply_recovery_to_annotation(
    *,
    annotation: dict[str, Any],
    recovery: dict[str, Any],
) -> None:
    clear_llm_argument_recovery_fields(annotation)
    classification_by_id = {
        item["sentence_id"]: item for item in recovery.get("classifications", [])
    }
    for sentence in annotation.get("sentences", []):
        sentence_id = coerce_sentence_id(sentence.get("sentence_id"))
        if sentence_id is None or sentence_id not in classification_by_id:
            continue
        classification = classification_by_id[sentence_id]
        sentence["llm_argument_recovery_label"] = classification["predicted_label"]
        sentence["llm_argument_recovery_confidence"] = classification["confidence"]
        sentence["llm_argument_recovery_reason"] = classification["reason"]
        sentence["llm_argument_recovery_sources"] = classification["candidate_sources"]


def store_llm_argument_recovery(
    *,
    annotation: dict[str, Any],
    recovery: dict[str, Any],
    args: argparse.Namespace,
    sentence_count: int,
    substantive_sentence_count: int,
    candidate_count: int,
    prompt_chunk_count: int,
    candidate_seed_roles: tuple[str, ...],
) -> None:
    apply_recovery_to_annotation(annotation=annotation, recovery=recovery)

    annotation[RECOVERY_KEY] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "mode": "missing_argument_only_recovery",
        "extraction_status": "recovered_with_fallbacks" if recovery.get("errors") else "recovered",
        "sentence_count": sentence_count,
        "substantive_sentence_count": substantive_sentence_count,
        "candidate_sentence_count": candidate_count,
        "prompt_chunk_count": prompt_chunk_count,
        "candidate_seed_roles": list(candidate_seed_roles),
        "candidate_neighbor_window": args.candidate_neighbor_window,
        "local_context_window": args.local_context_window,
        "include_cue_matched_sentences": args.include_cue_matched_sentences,
        "original_rr_appellant_ids": recovery["original_rr_appellant_ids"],
        "original_rr_respondent_ids": recovery["original_rr_respondent_ids"],
        "petitioner_candidate_ids": recovery["petitioner_candidate_ids"],
        "respondent_candidate_ids": recovery["respondent_candidate_ids"],
        "not_missing_candidate_ids": recovery["not_missing_candidate_ids"],
        "court_reasoning_ids": [],
        "fact_ids": [],
        "other_ids": [],
        "newly_recovered_petitioner_ids": recovery["newly_recovered_petitioner_ids"],
        "newly_recovered_respondent_ids": recovery["newly_recovered_respondent_ids"],
        "label_counts": recovery["label_counts"],
        "classifications": recovery["classifications"],
        "reclassified_original_argument_sentences": recovery["reclassified_original_argument_sentences"],
        "model_path": args.model_path,
        "backend": args.backend,
        "raw_model_responses": recovery["raw_model_responses"],
    }
    if recovery.get("errors"):
        annotation[RECOVERY_KEY]["errors"] = recovery["errors"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_output_path(input_path: Path, output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        return input_path
    return output_dir / input_path.name


# ---------------------------------------------------------------------------
# Extractor classes
# ---------------------------------------------------------------------------

class LocalVLLMRecoveryExtractor:
    def __init__(
        self,
        *,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        dtype: str,
        tokenizer_mode: str,
        trust_remote_code: bool,
        enforce_eager: bool,
        max_output_tokens: int,
    ):
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "transformers and vllm are required for --backend local_vllm."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode=tokenizer_mode,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_output_tokens,
        )

    def _render_prompt(self, *, prompt_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{SYSTEM_PROMPT}\n\n{prompt_text}"

    def extract_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompts = [
            self._render_prompt(prompt_text=item["prompt_text"])
            for item in items
        ]
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        if len(outputs) != len(items):
            raise RuntimeError(f"Expected {len(items)} vLLM outputs, got {len(outputs)}.")

        results: list[dict[str, Any]] = []
        for item, output in zip(items, outputs):
            if not output.outputs:
                raise RuntimeError(f"Empty generation for {item['file_id']} chunk.")
            content = output.outputs[0].text
            try:
                parsed = extract_json_object(content)
                classifications = normalize_output(
                    parsed,
                    shown_candidate_ids=item["shown_candidate_ids"],
                    candidate_lookup=item["candidate_lookup"],
                )
                results.append(
                    {
                        "shown_candidate_ids": item["shown_candidate_ids"],
                        "classifications": classifications,
                        "raw_model_response": content,
                    }
                )
            except Exception as exc:
                fallback_classifications = [
                    build_fallback_classification(
                        item["candidate_lookup"][sentence_id],
                        reason=f"Extraction error: {exc}",
                    )
                    for sentence_id in item["shown_candidate_ids"]
                ]
                results.append(
                    {
                        "shown_candidate_ids": item["shown_candidate_ids"],
                        "classifications": fallback_classifications,
                        "raw_model_response": content,
                        "error": str(exc),
                    }
                )
        return results

    def extract(self, *, item: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return self.extract_batch([item])[0]
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(f"Argument recovery failed for {item['file_id']}: {last_error}") from last_error


class RemoteHFRecoveryExtractor:
    def __init__(self, *, model_id: str, provider: str, hf_token: Optional[str], timeout: float):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for --backend remote_hf."
            ) from exc

        self.model_id = model_id
        self.client = InferenceClient(provider=provider, token=hf_token, timeout=timeout)

    def extract(self, *, item: dict[str, Any], max_retries: int = 3) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat_completion(
                    model=self.model_id,
                    temperature=0.0,
                    max_tokens=1_200,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": item["prompt_text"]},
                    ],
                )
                content = response.choices[0].message.content
                parsed = extract_json_object(content)
                classifications = normalize_output(
                    parsed,
                    shown_candidate_ids=item["shown_candidate_ids"],
                    candidate_lookup=item["candidate_lookup"],
                )
                return {
                    "shown_candidate_ids": item["shown_candidate_ids"],
                    "classifications": classifications,
                    "raw_model_response": content,
                }
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(min(2 * attempt, 10))
        fallback_classifications = [
            build_fallback_classification(
                item["candidate_lookup"][sentence_id],
                reason=f"Extraction error: {last_error}",
            )
            for sentence_id in item["shown_candidate_ids"]
        ]
        return {
            "shown_candidate_ids": item["shown_candidate_ids"],
            "classifications": fallback_classifications,
            "raw_model_response": "",
            "error": str(last_error),
        }


def build_error_chunk_result(item: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "shown_candidate_ids": item["shown_candidate_ids"],
        "classifications": [
            build_fallback_classification(
                item["candidate_lookup"][sentence_id],
                reason=f"Extraction error: {exc}",
            )
            for sentence_id in item["shown_candidate_ids"]
        ],
        "raw_model_response": "",
        "error": str(exc),
    }


def finalize_recovery_case(
    *,
    case: dict[str, Any],
    chunk_results: list[dict[str, Any]],
    args: argparse.Namespace,
    candidate_seed_roles: set[str],
) -> int:
    annotation = case["annotation"]
    try:
        recovery = aggregate_chunk_results(
            sentences=case["sentences"],
            candidates=case["candidates"],
            chunk_results=chunk_results,
        )
        store_llm_argument_recovery(
            annotation=annotation,
            recovery=recovery,
            args=args,
            sentence_count=case["sentence_count"],
            substantive_sentence_count=case["substantive_sentence_count"],
            candidate_count=len(case["candidates"]),
            prompt_chunk_count=case["prompt_chunk_count"],
            candidate_seed_roles=candidate_seed_roles,
        )
        write_json(case["path"], annotation)
        if recovery.get("errors"):
            print(f"  [OK-WITH-FALLBACKS] {case['file_id']}")
            return len(recovery["errors"])
        print(f"  [OK] {case['file_id']}")
        return 0
    except Exception as exc:
        clear_llm_argument_recovery_fields(annotation)
        annotation[RECOVERY_KEY] = {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "mode": "missing_argument_only_recovery",
            "extraction_status": "error",
            "sentence_count": case["sentence_count"],
            "substantive_sentence_count": case["substantive_sentence_count"],
            "candidate_sentence_count": len(case["candidates"]),
            "prompt_chunk_count": case["prompt_chunk_count"],
            "candidate_seed_roles": list(candidate_seed_roles),
            "candidate_neighbor_window": args.candidate_neighbor_window,
            "local_context_window": args.local_context_window,
            "include_cue_matched_sentences": args.include_cue_matched_sentences,
            "model_path": args.model_path,
            "backend": args.backend,
            "error": str(exc),
        }
        write_json(case["path"], annotation)
        print(f"  [ERROR] {case['file_id']}: {exc}")
        return 1


def flush_local_vllm_case_batch(
    *,
    pending_cases: list[dict[str, Any]],
    extractor: LocalVLLMRecoveryExtractor,
    args: argparse.Namespace,
    candidate_seed_roles: set[str],
) -> int:
    if not pending_cases:
        return 0

    flat_items: list[dict[str, Any]] = []
    item_counts: list[int] = []
    for case in pending_cases:
        item_counts.append(len(case["items"]))
        flat_items.extend(case["items"])

    total_errors = 0
    try:
        flat_results = extractor.extract_batch(flat_items)
        offset = 0
        for case, item_count in zip(pending_cases, item_counts):
            case_results = flat_results[offset: offset + item_count]
            total_errors += finalize_recovery_case(
                case=case,
                chunk_results=case_results,
                args=args,
                candidate_seed_roles=candidate_seed_roles,
            )
            offset += item_count
        return total_errors
    except Exception:
        for case in pending_cases:
            try:
                case_results = extractor.extract_batch(case["items"])
            except Exception:
                case_results = []
                for item in case["items"]:
                    try:
                        case_results.append(
                            extractor.extract(item=item, max_retries=args.max_retries)
                        )
                    except Exception as exc:
                        case_results.append(build_error_chunk_result(item, exc))
            total_errors += finalize_recovery_case(
                case=case,
                chunk_results=case_results,
                args=args,
                candidate_seed_roles=candidate_seed_roles,
            )
        return total_errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations_dir",
        default=str(DEFAULT_ANNOTATIONS_DIR),
        help="Directory containing annotation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        help="Optional directory to write recovered JSON files. Defaults to in-place updates in --annotations_dir.",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Local model directory or HF model ID.",
    )
    parser.add_argument(
        "--backend",
        choices=["local_vllm", "remote_hf"],
        default="local_vllm",
    )
    parser.add_argument("--provider", default="auto", help="HF Inference provider (remote_hf only).")
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    parser.add_argument("--max_files", type=int, help="Cap number of files processed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Re-run even if {RECOVERY_KEY} already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Skip files that already have {RECOVERY_KEY} (default behavior unless --overwrite).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Parse files but skip LLM calls.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=1_200,
        help="Max tokens for structured sentence-classification output.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=16,
        help="Approximate number of prompt chunks to send together in each local vLLM batch.",
    )
    parser.add_argument(
        "--candidate_seed_roles",
        default=",".join(DEFAULT_CANDIDATE_SEED_ROLES),
        help="Comma-separated RR roles to include directly in the recovery candidate pool.",
    )
    parser.add_argument(
        "--candidate_neighbor_window",
        type=int,
        default=2,
        help="Include +/- this many sentences around every RR ARG_* sentence.",
    )
    parser.add_argument(
        "--local_context_window",
        type=int,
        default=2,
        help="Show +/- this many local context sentences around each target sentence in the prompt.",
    )
    parser.add_argument(
        "--max_prompt_chars",
        type=int,
        default=MAX_PROMPT_CHARS,
        help="Approximate maximum characters per chunk prompt.",
    )
    parser.add_argument(
        "--include_cue_matched_sentences",
        action="store_true",
        default=False,
        help="Whether to add strong cue-matched sentences even if RR did not mark them as candidates.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max token context window for vLLM.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tokenizer_mode", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    args = parse_args()

    annotations_dir = Path(args.annotations_dir).resolve()
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    candidate_seed_roles = parse_role_list(args.candidate_seed_roles)
    paths = sorted(annotations_dir.glob("*.json"))

    if not args.overwrite:
        to_process = []
        for input_path in paths:
            target_path = resolve_output_path(input_path, output_dir)
            try:
                data = json.loads(target_path.read_text(encoding="utf-8"))
            except Exception:
                to_process.append(input_path)
                continue
            if RECOVERY_KEY not in data:
                to_process.append(input_path)
        skipped_existing = len(paths) - len(to_process)
        if skipped_existing:
            print(
                f"Skipping {skipped_existing} files that already have {RECOVERY_KEY} "
                f"(use --overwrite to re-run)."
            )
        paths = to_process

    if args.max_files is not None:
        paths = paths[: args.max_files]

    print(f"Files to process: {len(paths)}")
    if args.dry_run:
        print("[dry_run] Would process:")
        for path in paths:
            print(f"  {path.name}")
        return 0

    if args.backend == "local_vllm":
        extractor = LocalVLLMRecoveryExtractor(
            model_path=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            max_output_tokens=args.max_output_tokens,
        )
    else:
        extractor = RemoteHFRecoveryExtractor(
            model_id=args.model_path,
            provider=args.provider,
            hf_token=args.hf_token,
            timeout=120.0,
        )

    total_errors = 0
    pending_case_batch: list[dict[str, Any]] = []
    pending_prompt_count = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Recovering arguments")
    except ImportError:
        iterator = paths

    for input_path in iterator:
        try:
            annotation = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [SKIP] {input_path.name}: cannot parse JSON — {exc}")
            continue

        output_path = resolve_output_path(input_path, output_dir)
        file_id = str(annotation.get("file_id") or input_path.stem)
        sentences = annotation.get("sentences", []) or []
        sentence_count = len(sentences)
        substantive_count = count_substantive_sentences(sentences)

        candidates = build_candidate_pool(
            sentences=sentences,
            candidate_seed_roles=set(candidate_seed_roles),
            neighbor_window=args.candidate_neighbor_window,
            include_cue_matched_sentences=args.include_cue_matched_sentences,
        )

        if sentence_count == 0 or len(candidates) < MIN_CANDIDATE_SENTENCES:
            clear_llm_argument_recovery_fields(annotation)
            annotation[RECOVERY_KEY] = {
                "schema_version": RECOVERY_SCHEMA_VERSION,
                "mode": "missing_argument_only_recovery",
                "extraction_status": "skipped_short_case",
                "sentence_count": sentence_count,
                "substantive_sentence_count": substantive_count,
                "candidate_sentence_count": len(candidates),
                "prompt_chunk_count": 0,
                "candidate_seed_roles": list(candidate_seed_roles),
                "candidate_neighbor_window": args.candidate_neighbor_window,
                "local_context_window": args.local_context_window,
                "include_cue_matched_sentences": args.include_cue_matched_sentences,
                "model_path": args.model_path,
                "backend": args.backend,
                "note": (
                    f"Case has only {sentence_count} sentence(s) and "
                    f"{len(candidates)} possible missing-argument candidate(s) — skipped LLM call."
                ),
            }
            write_json(output_path, annotation)
            print(f"  [SHORT] {file_id} ({sentence_count} sents, {len(candidates)} candidates)")
            continue

        prompt_chunks = build_prompt_chunks(
            file_id=file_id,
            sentences=sentences,
            candidates=candidates,
            local_context_window=args.local_context_window,
            max_prompt_chars=args.max_prompt_chars,
        )
        candidate_lookup = {candidate["sentence_id"]: candidate for candidate in candidates}

        items = [
            {
                "file_id": file_id,
                "prompt_text": chunk["prompt_text"],
                "shown_candidate_ids": chunk["shown_candidate_ids"],
                "candidate_lookup": candidate_lookup,
            }
            for chunk in prompt_chunks
        ]

        if args.backend == "local_vllm":
            pending_case_batch.append(
                {
                    "path": output_path,
                    "annotation": annotation,
                    "file_id": file_id,
                    "sentences": sentences,
                    "candidates": candidates,
                    "items": items,
                    "sentence_count": sentence_count,
                    "substantive_sentence_count": substantive_count,
                    "prompt_chunk_count": len(prompt_chunks),
                }
            )
            pending_prompt_count += len(items)
            if pending_prompt_count >= args.generation_batch_size:
                total_errors += flush_local_vllm_case_batch(
                    pending_cases=pending_case_batch,
                    extractor=extractor,
                    args=args,
                    candidate_seed_roles=candidate_seed_roles,
                )
                pending_case_batch = []
                pending_prompt_count = 0
            continue

        try:
            if args.backend == "remote_hf":
                chunk_results = [
                    extractor.extract(item=item, max_retries=args.max_retries)
                    for item in items
                ]
            recovery = aggregate_chunk_results(
                sentences=sentences,
                candidates=candidates,
                chunk_results=chunk_results,
            )
            store_llm_argument_recovery(
                annotation=annotation,
                recovery=recovery,
                args=args,
                sentence_count=sentence_count,
                substantive_sentence_count=substantive_count,
                candidate_count=len(candidates),
                prompt_chunk_count=len(prompt_chunks),
                candidate_seed_roles=candidate_seed_roles,
            )
            write_json(output_path, annotation)
            if recovery.get("errors"):
                total_errors += len(recovery["errors"])
                print(f"  [OK-WITH-FALLBACKS] {file_id}")
            else:
                print(f"  [OK] {file_id}")
        except Exception as exc:
            clear_llm_argument_recovery_fields(annotation)
            annotation[RECOVERY_KEY] = {
                "schema_version": RECOVERY_SCHEMA_VERSION,
                "mode": "missing_argument_only_recovery",
                "extraction_status": "error",
                "sentence_count": sentence_count,
                "substantive_sentence_count": substantive_count,
                "candidate_sentence_count": len(candidates),
                "prompt_chunk_count": len(prompt_chunks),
                "candidate_seed_roles": list(candidate_seed_roles),
                "candidate_neighbor_window": args.candidate_neighbor_window,
                "local_context_window": args.local_context_window,
                "include_cue_matched_sentences": args.include_cue_matched_sentences,
                "model_path": args.model_path,
                "backend": args.backend,
                "error": str(exc),
            }
            write_json(output_path, annotation)
            print(f"  [ERROR] {file_id}: {exc}")
            total_errors += 1

    if pending_case_batch:
        total_errors += flush_local_vllm_case_batch(
            pending_cases=pending_case_batch,
            extractor=extractor,
            args=args,
            candidate_seed_roles=candidate_seed_roles,
        )

    print(f"\nDone. Errors: {total_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
