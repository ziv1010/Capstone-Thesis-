#!/usr/bin/env python3
"""Extract appellant and respondent arguments from annotated Indian court cases.

Reads annotation JSON files (output of NER + rhetorical-role pipeline) and uses
Mistral-Small-24B-Instruct-2501 to synthesise coherent argument summaries for
the appellant and the respondent.

The rhetorical-role labels in the annotation files are used as *hints* to the
LLM for side refinement. If a prior ``llm_argument_recovery`` stage exists,
this script refines the recovered petitioner/respondent argument buckets;
otherwise it falls back to the original RR argument labels.

Results are appended/updated in-place on the annotation JSON files under the
key ``llm_arguments``.

Usage example (local GPU, default paths)::

    python extract_arguments_mistral.py

    # resume after interruption
    python extract_arguments_mistral.py --resume

    # limit to first 10 files for testing
    python extract_arguments_mistral.py --max_files 10 --dry_run

    # overwrite existing llm_arguments
    python extract_arguments_mistral.py --overwrite
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Union

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

# Maximum number of characters to include in the candidate sentence review set
# sent to the model. The prompt focuses on the current candidate argument
# buckets, sourced either from RR argument labels or from llm_argument_recovery,
# rather than the full judgment.
MAX_CASE_CHARS = 18_000

# Roles that indicate the case has substantive legal content worth sending to
# the model.  FAC and NONE alone are not sufficient — those often appear in
# garbled / stub extractions.
SUBSTANTIVE_ROLES = {"ARG_PETITIONER", "ARG_RESPONDENT", "ANALYSIS", "RATIO", "PRE_RELIED", "RPC"}

# RR fallback candidates are only the sentences already labelled as arguments by
# RR. If llm_argument_recovery is present, the recovery-labelled
# PETITIONER_ARGUMENT / RESPONDENT_ARGUMENT sentences become the review set.
REVIEWABLE_ROLES = {"ARG_PETITIONER", "ARG_RESPONDENT"}
RECOVERY_ARGUMENT_LABELS = {"PETITIONER_ARGUMENT", "RESPONDENT_ARGUMENT"}
RECOVERY_SUCCESS_STATUSES = {"recovered", "recovered_with_fallbacks"}

# If a case has fewer reviewable sentences than this threshold, skip the LLM
# and write a "skipped_short_case" record instead.
MIN_REVIEW_CANDIDATE_SENTENCES = 2

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

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a legal argument-side reviewer for Indian court cases.

You will receive a list of candidate sentences from a court judgment. Each
shown sentence belongs to the current side-refinement review set. That review
set may come either from RR argument labels or from a prior argument-recovery
stage.

Each candidate sentence has this format:

  [sentence_id] orig=<current_bucket> rr=<original_rr_role> source=<candidate_source>
  recovery=<recovery_hint> cue=<weak_party_hint> sentence text

IMPORTANT:
- You are only correcting side mix-ups between ARG_PETITIONER and
  ARG_RESPONDENT within the shown review set.
- Do not add sentence IDs that were not shown.
- Do not create new facts, reasoning, or new rhetorical-role buckets.
- If a sentence is ambiguous, leave it in its current bucket.
- The cue is only a weak heuristic hint and may be wrong.
- The recovery tag is only a prior-stage hint and may be wrong.
- Read the actual sentence text, not just the orig, rr, source, recovery, or
  cue tags.
- Even if no sentence IDs need to move, you must still extract the final
  appellant and respondent argument text from the shown sentences.

Disambiguation rules:
- If a shown ARG_PETITIONER sentence actually states the
  respondent/State/defendant/prosecution side's submission, move it to
  ARG_RESPONDENT.
- If a shown ARG_RESPONDENT sentence actually states the
  petitioner/appellant/plaintiff/applicant side's submission, move it to
  ARG_PETITIONER.
- If the sentence is unclear, mixed, or not confidently attributable to the
  opposite side, do not move it.

Your tasks:
1. Return only the shown sentence IDs that should move to the opposite bucket.
2. Write a coherent appellant argument summary using the final corrected
   ARG_PETITIONER bucket.
3. Write a coherent respondent argument summary using the final corrected
   ARG_RESPONDENT bucket.

If genuinely no appellant material is present, write exactly:
"No appellant argument found."

If genuinely no respondent material is present, write exactly:
"No respondent argument found."

Set confidence to "low" whenever attribution is ambiguous or heavily inferred,
"medium" when some side-specific material is present but incomplete, and "high"
only when the side assignments are clearly separated.

Return JSON only, matching this schema exactly:
{
  "move_to_appellant_ids": [2],
  "move_to_respondent_ids": [1],
  "appellant_argument": "...",
  "respondent_argument": "...",
  "confidence": "high" | "medium" | "low"
}
"""


def count_substantive_sentences(sentences: list[dict[str, Any]]) -> int:
    return sum(
        1 for s in sentences
        if (s.get("rhetorical_role") or "NONE") in SUBSTANTIVE_ROLES
    )


def coerce_sentence_id(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


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


def iter_review_candidates(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sentence in sentences:
        text = str(sentence.get("text", "")).strip()
        if not text:
            continue
        sentence_id = coerce_sentence_id(sentence.get("sentence_id"))
        if sentence_id is None:
            continue
        role = str(sentence.get("rhetorical_role") or "NONE")
        if role not in REVIEWABLE_ROLES:
            continue
        cue = detect_party_cue(text)
        candidates.append(
            {
                "sentence_id": sentence_id,
                "current_bucket": role,
                "rhetorical_role": role,
                "rr_role": role,
                "source_mode": "rr_argument_roles",
                "recovery_label": None,
                "recovery_confidence": None,
                "recovery_sources": [],
                "cue": cue,
                "text": text,
            }
        )
    return candidates


def count_review_candidate_sentences(sentences: list[dict[str, Any]]) -> int:
    return len(iter_review_candidates(sentences))


def map_recovery_label_to_bucket(label: Any) -> Optional[str]:
    normalized = str(label or "").strip().upper()
    if normalized == "PETITIONER_ARGUMENT":
        return "ARG_PETITIONER"
    if normalized == "RESPONDENT_ARGUMENT":
        return "ARG_RESPONDENT"
    return None


def build_review_candidates(annotation: dict[str, Any]) -> dict[str, Any]:
    sentences = annotation.get("sentences", [])
    sentence_lookup = {
        sentence_id: sentence
        for sentence in sentences
        if (sentence_id := coerce_sentence_id(sentence.get("sentence_id"))) is not None
    }
    sentence_order = {
        sentence_id: index for index, sentence_id in enumerate(sentence_lookup, start=1)
    }

    rr_appellant_ids = sorted(
        (
            sid for sid, sentence in sentence_lookup.items()
            if str(sentence.get("rhetorical_role") or "NONE") == "ARG_PETITIONER"
        ),
        key=lambda sid: sentence_order.get(sid, sid),
    )
    rr_respondent_ids = sorted(
        (
            sid for sid, sentence in sentence_lookup.items()
            if str(sentence.get("rhetorical_role") or "NONE") == "ARG_RESPONDENT"
        ),
        key=lambda sid: sentence_order.get(sid, sid),
    )

    recovery = annotation.get("llm_argument_recovery")
    if isinstance(recovery, dict) and str(recovery.get("extraction_status") or "") in RECOVERY_SUCCESS_STATUSES:
        candidates: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for item in recovery.get("classifications", []):
            if not isinstance(item, dict):
                continue
            sentence_id = coerce_sentence_id(item.get("sentence_id"))
            if sentence_id is None or sentence_id in seen_ids:
                continue
            sentence = sentence_lookup.get(sentence_id)
            if not sentence:
                continue
            text = str(sentence.get("text", "")).strip()
            if not text:
                continue
            current_bucket = map_recovery_label_to_bucket(
                item.get("predicted_label") or item.get("label")
            )
            if current_bucket is None:
                continue
            seen_ids.add(sentence_id)
            candidates.append(
                {
                    "sentence_id": sentence_id,
                    "current_bucket": current_bucket,
                    "rhetorical_role": current_bucket,
                    "rr_role": str(sentence.get("rhetorical_role") or "NONE"),
                    "source_mode": "llm_argument_recovery",
                    "recovery_label": str(
                        item.get("predicted_label") or item.get("label") or ""
                    ).strip().upper() or None,
                    "recovery_confidence": str(item.get("confidence") or "").strip().lower() or None,
                    "recovery_sources": item.get("sources") or [],
                    "cue": detect_party_cue(text),
                    "text": text,
                }
            )

        candidates.sort(key=lambda candidate: sentence_order.get(candidate["sentence_id"], candidate["sentence_id"]))
        base_appellant_ids = [
            candidate["sentence_id"]
            for candidate in candidates
            if candidate["current_bucket"] == "ARG_PETITIONER"
        ]
        base_respondent_ids = [
            candidate["sentence_id"]
            for candidate in candidates
            if candidate["current_bucket"] == "ARG_RESPONDENT"
        ]
        return {
            "candidates": candidates,
            "candidate_source_mode": "llm_argument_recovery",
            "base_appellant_ids": base_appellant_ids,
            "base_respondent_ids": base_respondent_ids,
            "original_rr_appellant_ids": rr_appellant_ids,
            "original_rr_respondent_ids": rr_respondent_ids,
        }

    candidates = iter_review_candidates(sentences)
    base_appellant_ids = [
        candidate["sentence_id"]
        for candidate in candidates
        if candidate["current_bucket"] == "ARG_PETITIONER"
    ]
    base_respondent_ids = [
        candidate["sentence_id"]
        for candidate in candidates
        if candidate["current_bucket"] == "ARG_RESPONDENT"
    ]
    return {
        "candidates": candidates,
        "candidate_source_mode": "rr_argument_roles",
        "base_appellant_ids": base_appellant_ids,
        "base_respondent_ids": base_respondent_ids,
        "original_rr_appellant_ids": rr_appellant_ids,
        "original_rr_respondent_ids": rr_respondent_ids,
    }


def build_user_prompt(*, file_id: str, annotation: dict[str, Any]) -> dict[str, Any]:
    """Render a compact review set for petitioner/respondent side refinement."""
    review_bundle = build_review_candidates(annotation)
    review_candidates = review_bundle["candidates"]
    total_chars = 0
    shown_candidates: list[dict[str, Any]] = []
    shown_ids: list[int] = []
    truncated = False

    for candidate in review_candidates:
        recovery_hint = (
            f"{candidate['recovery_label']}:{candidate['recovery_confidence'] or 'none'}"
            if candidate["recovery_label"]
            else "NONE:none"
        )
        line = (
            f"[{candidate['sentence_id']}] "
            f"orig={candidate['current_bucket']} "
            f"rr={candidate['rr_role']} "
            f"source={candidate['source_mode']} "
            f"recovery={recovery_hint} "
            f"cue={candidate['cue']} "
            f"{candidate['text']}"
        )
        total_chars += len(line) + 1
        if total_chars > MAX_CASE_CHARS and shown_ids:
            truncated = True
            break
        shown_candidates.append(candidate)
        shown_ids.append(candidate["sentence_id"])

    role_counts = Counter(candidate["current_bucket"] for candidate in shown_candidates)
    cue_ids = {
        cue: [candidate["sentence_id"] for candidate in shown_candidates if candidate["cue"] == cue]
        for cue in ("APPELLANT", "RESPONDENT", "BOTH")
    }

    lines: list[str] = [
        f"Case ID: {file_id}",
        "",
        "Candidate sentence review set for ARG_PETITIONER vs ARG_RESPONDENT side refinement only.",
        f"Candidate source mode: {review_bundle['candidate_source_mode']}",
        "Each shown sentence comes from the current argument bucket before side refinement.",
        "Only list IDs that should switch sides.",
        "If a shown sentence should stay in its current bucket, omit it from both move lists.",
        "Read the actual sentence text when deciding sides and when writing the final argument summaries.",
        "",
        f"Total review candidates available: {len(review_candidates)}",
        f"Shown review candidates: {len(shown_candidates)}",
        "Current bucket counts in shown review set: "
        + (", ".join(f"{role}={count}" for role, count in sorted(role_counts.items())) or "none"),
        f"Likely appellant-cue sentence IDs: {cue_ids['APPELLANT'] or '[]'}",
        f"Likely respondent-cue sentence IDs: {cue_ids['RESPONDENT'] or '[]'}",
        f"Mixed-cue sentence IDs: {cue_ids['BOTH'] or '[]'}",
        "",
        "Shown candidate sentences:",
    ]

    for candidate in shown_candidates:
        recovery_hint = (
            f"{candidate['recovery_label']}:{candidate['recovery_confidence'] or 'none'}"
            if candidate["recovery_label"]
            else "NONE:none"
        )
        lines.append(
            f"[{candidate['sentence_id']}] "
            f"orig={candidate['current_bucket']} "
            f"rr={candidate['rr_role']} "
            f"source={candidate['source_mode']} "
            f"recovery={recovery_hint} "
            f"cue={candidate['cue']} "
            f"{candidate['text']}"
        )

    if truncated:
        lines.append("[... case text truncated for length ...]")
        lines.append("")
        lines.append("Only review the sentence IDs actually shown above.")

    return {
        "prompt_text": "\n".join(lines),
        "review_candidate_count": len(review_candidates),
        "shown_review_candidate_ids": shown_ids,
        "prompt_truncated": truncated,
        "candidate_source_mode": review_bundle["candidate_source_mode"],
        "base_appellant_ids": review_bundle["base_appellant_ids"],
        "base_respondent_ids": review_bundle["base_respondent_ids"],
        "original_rr_appellant_ids": review_bundle["original_rr_appellant_ids"],
        "original_rr_respondent_ids": review_bundle["original_rr_respondent_ids"],
        "review_candidate_lookup": {
            candidate["sentence_id"]: candidate for candidate in review_candidates
        },
    }


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


NOT_FOUND_APPELLANT = "No appellant argument found."
NOT_FOUND_RESPONDENT = "No respondent argument found."


def dedupe_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def parse_sentence_id_list(value: Any) -> list[int]:
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = re.findall(r"\d+", value)
    else:
        raw_items = []

    parsed: list[int] = []
    for item in raw_items:
        sentence_id = coerce_sentence_id(item)
        if sentence_id is not None:
            parsed.append(sentence_id)
    return dedupe_preserve_order(parsed)


def resolve_preferred_argument_bucket(
    sentence: dict[str, Any],
    *,
    current_bucket: Optional[str] = None,
) -> str:
    cue = detect_party_cue(str(sentence.get("text", "")))
    if cue == "APPELLANT":
        return "ARG_PETITIONER"
    if cue == "RESPONDENT":
        return "ARG_RESPONDENT"

    if current_bucket == "ARG_PETITIONER":
        return "ARG_PETITIONER"
    if current_bucket == "ARG_RESPONDENT":
        return "ARG_RESPONDENT"

    original_role = str(sentence.get("rhetorical_role") or "NONE")
    if original_role == "ARG_PETITIONER":
        return "ARG_PETITIONER"
    if original_role == "ARG_RESPONDENT":
        return "ARG_RESPONDENT"
    return "NONE"


def build_fallback_text(
    *,
    sentence_ids: list[int],
    sentence_lookup: dict[int, dict[str, Any]],
    not_found_text: str,
    max_chars: int = 1_500,
) -> str:
    chunks: list[str] = []
    total = 0
    for sentence_id in sentence_ids:
        sentence = sentence_lookup.get(sentence_id)
        if not sentence:
            continue
        text = normalize_whitespace(str(sentence.get("text", "")))
        if not text:
            continue
        if total + len(text) + 1 > max_chars:
            break
        chunks.append(text)
        total += len(text) + 1
    return " ".join(chunks).strip() or not_found_text


def normalize_output(
    payload: dict[str, Any],
    *,
    sentences: list[dict[str, Any]],
    shown_review_candidate_ids: list[int],
    base_appellant_ids: list[int],
    base_respondent_ids: list[int],
    original_rr_appellant_ids: list[int],
    original_rr_respondent_ids: list[int],
    review_candidate_lookup: dict[int, dict[str, Any]],
    candidate_source_mode: str,
) -> dict[str, Any]:
    sentence_lookup = {
        sentence_id: sentence
        for sentence in sentences
        if (sentence_id := coerce_sentence_id(sentence.get("sentence_id"))) is not None
    }
    sentence_order = {sentence_id: index for index, sentence_id in enumerate(sentence_lookup, start=1)}
    valid_ids = set(sentence_lookup)
    review_candidate_ids = [sid for sid in shown_review_candidate_ids if sid in valid_ids]
    review_candidate_id_set = set(review_candidate_ids)

    original_appellant_ids = sorted(
        [sid for sid in dedupe_preserve_order(base_appellant_ids) if sid in valid_ids],
        key=lambda sid: sentence_order.get(sid, sid),
    )
    original_respondent_ids = sorted(
        [sid for sid in dedupe_preserve_order(base_respondent_ids) if sid in valid_ids],
        key=lambda sid: sentence_order.get(sid, sid),
    )
    original_rr_appellant_ids = sorted(
        [sid for sid in dedupe_preserve_order(original_rr_appellant_ids) if sid in valid_ids],
        key=lambda sid: sentence_order.get(sid, sid),
    )
    original_rr_respondent_ids = sorted(
        [sid for sid in dedupe_preserve_order(original_rr_respondent_ids) if sid in valid_ids],
        key=lambda sid: sentence_order.get(sid, sid),
    )

    move_to_appellant_set = {
        sid
        for sid in parse_sentence_id_list(payload.get("move_to_appellant_ids"))
        if sid in review_candidate_id_set
    }
    move_to_respondent_set = {
        sid
        for sid in parse_sentence_id_list(payload.get("move_to_respondent_ids"))
        if sid in review_candidate_id_set
    }

    for sentence_id in sorted(move_to_appellant_set & move_to_respondent_set):
        preferred_bucket = resolve_preferred_argument_bucket(
            sentence_lookup[sentence_id],
            current_bucket=review_candidate_lookup.get(sentence_id, {}).get("current_bucket"),
        )
        if preferred_bucket == "ARG_PETITIONER":
            move_to_respondent_set.discard(sentence_id)
        elif preferred_bucket == "ARG_RESPONDENT":
            move_to_appellant_set.discard(sentence_id)
        else:
            move_to_appellant_set.discard(sentence_id)
            move_to_respondent_set.discard(sentence_id)

    appellant_set = set(original_appellant_ids)
    respondent_set = set(original_respondent_ids)

    for sentence_id in move_to_appellant_set:
        appellant_set.add(sentence_id)
        respondent_set.discard(sentence_id)

    for sentence_id in move_to_respondent_set:
        respondent_set.add(sentence_id)
        appellant_set.discard(sentence_id)

    appellant_ids = sorted(appellant_set, key=lambda sid: sentence_order.get(sid, sid))
    respondent_ids = sorted(respondent_set, key=lambda sid: sentence_order.get(sid, sid))
    move_to_appellant_ids = sorted(move_to_appellant_set, key=lambda sid: sentence_order.get(sid, sid))
    move_to_respondent_ids = sorted(move_to_respondent_set, key=lambda sid: sentence_order.get(sid, sid))

    appellant = str(payload.get("appellant_argument", "")).strip()
    respondent = str(payload.get("respondent_argument", "")).strip()
    confidence = str(payload.get("confidence", "medium")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    if not appellant:
        appellant = build_fallback_text(
            sentence_ids=appellant_ids,
            sentence_lookup=sentence_lookup,
            not_found_text=NOT_FOUND_APPELLANT,
        )
    if not respondent:
        respondent = build_fallback_text(
            sentence_ids=respondent_ids,
            sentence_lookup=sentence_lookup,
            not_found_text=NOT_FOUND_RESPONDENT,
        )

    # If both sides are absent, the model has effectively found nothing —
    # high/medium confidence on a double-not-found is always wrong.
    both_not_found = (
        appellant == NOT_FOUND_APPELLANT and respondent == NOT_FOUND_RESPONDENT
    )
    if both_not_found and confidence != "low":
        confidence = "low"

    return {
        "original_appellant_sentence_ids": original_appellant_ids,
        "original_respondent_sentence_ids": original_respondent_ids,
        "move_to_appellant_ids": move_to_appellant_ids,
        "move_to_respondent_ids": move_to_respondent_ids,
        "appellant_sentence_ids": appellant_ids,
        "respondent_sentence_ids": respondent_ids,
        "appellant_argument": appellant,
        "respondent_argument": respondent,
        "confidence": confidence,
        "candidate_source_mode": candidate_source_mode,
        "original_rr_appellant_sentence_ids": original_rr_appellant_ids,
        "original_rr_respondent_sentence_ids": original_rr_respondent_ids,
    }


def build_error_extraction(
    *,
    error: Union[Exception, str],
    base_appellant_ids: list[int],
    base_respondent_ids: list[int],
    candidate_source_mode: str,
    original_rr_appellant_ids: list[int],
    original_rr_respondent_ids: list[int],
) -> dict[str, Any]:
    message = str(error)
    return {
        "original_appellant_sentence_ids": list(base_appellant_ids),
        "original_respondent_sentence_ids": list(base_respondent_ids),
        "move_to_appellant_ids": [],
        "move_to_respondent_ids": [],
        "appellant_argument": f"[Extraction error: {message}]",
        "respondent_argument": f"[Extraction error: {message}]",
        "confidence": "low",
        "appellant_sentence_ids": list(base_appellant_ids),
        "respondent_sentence_ids": list(base_respondent_ids),
        "candidate_source_mode": candidate_source_mode,
        "original_rr_appellant_sentence_ids": list(original_rr_appellant_ids),
        "original_rr_respondent_sentence_ids": list(original_rr_respondent_ids),
        "error": message,
    }


def clear_llm_argument_roles(annotation: dict[str, Any]) -> None:
    for sentence in annotation.get("sentences", []):
        if isinstance(sentence, dict):
            sentence.pop("llm_argument_role", None)
            sentence.pop("llm_argument_role_source", None)


def build_bucket_entries(
    *,
    sentences: list[dict[str, Any]],
    sentence_ids: list[int],
    llm_role: str,
) -> list[dict[str, Any]]:
    lookup = {
        sentence_id: sentence
        for sentence in sentences
        if (sentence_id := coerce_sentence_id(sentence.get("sentence_id"))) is not None
    }
    entries: list[dict[str, Any]] = []
    for sentence_id in sentence_ids:
        sentence = lookup.get(sentence_id)
        if not sentence:
            continue
        entries.append(
            {
                "sentence_id": sentence_id,
                "text": sentence.get("text", ""),
                "original_rhetorical_role": sentence.get("rhetorical_role", "NONE") or "NONE",
                "llm_argument_role": llm_role,
            }
        )
    return entries


def build_bucket_text(entries: list[dict[str, Any]]) -> str:
    return "\n".join(str(entry.get("text", "")).strip() for entry in entries if str(entry.get("text", "")).strip())


def apply_extraction_to_annotation(
    *,
    annotation: dict[str, Any],
    extraction: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    clear_llm_argument_roles(annotation)

    sentences = annotation.get("sentences", [])
    llm_role_by_id = {
        **{sentence_id: "ARG_PETITIONER" for sentence_id in extraction["appellant_sentence_ids"]},
        **{sentence_id: "ARG_RESPONDENT" for sentence_id in extraction["respondent_sentence_ids"]},
    }

    relabelled_sentences: list[dict[str, Any]] = []
    for sentence in sentences:
        sentence_id = coerce_sentence_id(sentence.get("sentence_id"))
        if sentence_id is None or sentence_id not in llm_role_by_id:
            continue
        llm_role = llm_role_by_id[sentence_id]
        original_role = str(sentence.get("rhetorical_role") or "NONE")
        sentence["llm_argument_role"] = llm_role
        sentence["llm_argument_role_source"] = "llm_side_refinement"
        if original_role != llm_role:
            relabelled_sentences.append(
                {
                    "sentence_id": sentence_id,
                    "text": sentence.get("text", ""),
                    "original_rhetorical_role": original_role,
                    "llm_argument_role": llm_role,
                }
            )

    buckets = {
        "appellant": build_bucket_entries(
            sentences=sentences,
            sentence_ids=extraction["appellant_sentence_ids"],
            llm_role="ARG_PETITIONER",
        ),
        "respondent": build_bucket_entries(
            sentences=sentences,
            sentence_ids=extraction["respondent_sentence_ids"],
            llm_role="ARG_RESPONDENT",
        ),
    }
    return buckets, relabelled_sentences


def store_llm_arguments(
    *,
    annotation: dict[str, Any],
    extraction: dict[str, Any],
    args: argparse.Namespace,
    sentence_count: int,
    substantive_sentence_count: int,
    review_candidate_sentence_count: int,
    prompt_review_sentence_count: int,
    prompt_truncated: bool,
) -> None:
    buckets, relabelled_sentences = apply_extraction_to_annotation(
        annotation=annotation,
        extraction=extraction,
    )

    annotation["llm_arguments"] = {
        "schema_version": 3,
        "mode": "argument_side_refinement_only",
        "extraction_status": "error" if "error" in extraction else "extracted",
        "candidate_source_mode": extraction.get("candidate_source_mode", "rr_argument_roles"),
        "appellant_argument": extraction["appellant_argument"],
        "respondent_argument": extraction["respondent_argument"],
        "confidence": extraction["confidence"],
        "sentence_count": sentence_count,
        "substantive_sentence_count": substantive_sentence_count,
        "review_candidate_sentence_count": review_candidate_sentence_count,
        "prompt_review_sentence_count": prompt_review_sentence_count,
        "prompt_truncated": prompt_truncated,
        "bucket_counts": {
            "ARG_PETITIONER": len(extraction["appellant_sentence_ids"]),
            "ARG_RESPONDENT": len(extraction["respondent_sentence_ids"]),
        },
        "original_appellant_sentence_ids": extraction["original_appellant_sentence_ids"],
        "original_respondent_sentence_ids": extraction["original_respondent_sentence_ids"],
        "original_rr_appellant_sentence_ids": extraction.get("original_rr_appellant_sentence_ids", []),
        "original_rr_respondent_sentence_ids": extraction.get("original_rr_respondent_sentence_ids", []),
        "move_to_appellant_ids": extraction["move_to_appellant_ids"],
        "move_to_respondent_ids": extraction["move_to_respondent_ids"],
        "appellant_sentence_ids": extraction["appellant_sentence_ids"],
        "respondent_sentence_ids": extraction["respondent_sentence_ids"],
        "appellant_sentences": buckets["appellant"],
        "respondent_sentences": buckets["respondent"],
        "appellant_text": build_bucket_text(buckets["appellant"]),
        "respondent_text": build_bucket_text(buckets["respondent"]),
        "relabelled_sentences": relabelled_sentences,
        "model_path": args.model_path,
        "backend": args.backend,
        "raw_model_response": extraction.get("raw_model_response", ""),
    }
    if "error" in extraction:
        annotation["llm_arguments"]["error"] = extraction["error"]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Classifier classes
# ---------------------------------------------------------------------------

class LocalVLLMExtractor:
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

        self.model_path = model_path
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
            raise RuntimeError(
                f"Expected {len(items)} vLLM outputs, got {len(outputs)}."
            )

        results = []
        for item, output in zip(items, outputs):
            if not output.outputs:
                raise RuntimeError(f"Empty generation for {item['file_id']}")
            content = output.outputs[0].text
            try:
                parsed = extract_json_object(content)
                normalized = normalize_output(
                    parsed,
                    sentences=item["sentences"],
                    shown_review_candidate_ids=item["shown_review_candidate_ids"],
                    base_appellant_ids=item["base_appellant_ids"],
                    base_respondent_ids=item["base_respondent_ids"],
                    original_rr_appellant_ids=item["original_rr_appellant_ids"],
                    original_rr_respondent_ids=item["original_rr_respondent_ids"],
                    review_candidate_lookup=item["review_candidate_lookup"],
                    candidate_source_mode=item["candidate_source_mode"],
                )
            except Exception as exc:
                normalized = build_error_extraction(
                    error=exc,
                    base_appellant_ids=item["base_appellant_ids"],
                    base_respondent_ids=item["base_respondent_ids"],
                    candidate_source_mode=item["candidate_source_mode"],
                    original_rr_appellant_ids=item["original_rr_appellant_ids"],
                    original_rr_respondent_ids=item["original_rr_respondent_ids"],
                )
            results.append({**normalized, "raw_model_response": content})
        return results

    def extract(
        self,
        *,
        file_id: str,
        sentences: list[dict[str, Any]],
        prompt_text: str,
        shown_review_candidate_ids: list[int],
        base_appellant_ids: list[int],
        base_respondent_ids: list[int],
        original_rr_appellant_ids: list[int],
        original_rr_respondent_ids: list[int],
        review_candidate_lookup: dict[int, dict[str, Any]],
        candidate_source_mode: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return self.extract_batch(
                    [
                        {
                            "file_id": file_id,
                            "sentences": sentences,
                            "prompt_text": prompt_text,
                            "shown_review_candidate_ids": shown_review_candidate_ids,
                            "base_appellant_ids": base_appellant_ids,
                            "base_respondent_ids": base_respondent_ids,
                            "original_rr_appellant_ids": original_rr_appellant_ids,
                            "original_rr_respondent_ids": original_rr_respondent_ids,
                            "review_candidate_lookup": review_candidate_lookup,
                            "candidate_source_mode": candidate_source_mode,
                        }
                    ]
                )[0]
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(
            f"Argument extraction failed for {file_id}: {last_error}"
        ) from last_error


class RemoteHFExtractor:
    def __init__(self, *, model_id: str, provider: str, hf_token: Optional[str], timeout: float):
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for --backend remote_hf."
            ) from exc

        self.model_id = model_id
        self.client = InferenceClient(provider=provider, token=hf_token, timeout=timeout)

    def extract(
        self,
        *,
        file_id: str,
        sentences: list[dict[str, Any]],
        prompt_text: str,
        shown_review_candidate_ids: list[int],
        base_appellant_ids: list[int],
        base_respondent_ids: list[int],
        original_rr_appellant_ids: list[int],
        original_rr_respondent_ids: list[int],
        review_candidate_lookup: dict[int, dict[str, Any]],
        candidate_source_mode: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat_completion(
                    model=self.model_id,
                    temperature=0.0,
                    max_tokens=900,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_text},
                    ],
                )
                content = response.choices[0].message.content
                parsed = extract_json_object(content)
                normalized = normalize_output(
                    parsed,
                    sentences=sentences,
                    shown_review_candidate_ids=shown_review_candidate_ids,
                    base_appellant_ids=base_appellant_ids,
                    base_respondent_ids=base_respondent_ids,
                    original_rr_appellant_ids=original_rr_appellant_ids,
                    original_rr_respondent_ids=original_rr_respondent_ids,
                    review_candidate_lookup=review_candidate_lookup,
                    candidate_source_mode=candidate_source_mode,
                )
                return {**normalized, "raw_model_response": content}
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(
            f"Argument extraction failed for {file_id}: {last_error}"
        ) from last_error


# ---------------------------------------------------------------------------
# Batch flushing
# ---------------------------------------------------------------------------

def flush_vllm_batch(
    *,
    pending: list[dict[str, Any]],
    extractor: LocalVLLMExtractor,
    args: argparse.Namespace,
) -> tuple[int, int]:
    """Run a batch through vLLM, update annotation files in-place."""
    if not pending:
        return 0, 0

    errors = 0
    batch_inputs = [
        {
            "file_id": item["file_id"],
            "sentences": item["sentences"],
            "prompt_text": item["prompt_text"],
            "shown_review_candidate_ids": item["shown_review_candidate_ids"],
            "base_appellant_ids": item["base_appellant_ids"],
            "base_respondent_ids": item["base_respondent_ids"],
            "original_rr_appellant_ids": item["original_rr_appellant_ids"],
            "original_rr_respondent_ids": item["original_rr_respondent_ids"],
            "review_candidate_lookup": item["review_candidate_lookup"],
            "candidate_source_mode": item["candidate_source_mode"],
        }
        for item in pending
    ]

    try:
        extractions = extractor.extract_batch(batch_inputs)
    except Exception:
        # Fall back to one-by-one so one bad file doesn't kill the batch
        extractions = []
        for item in batch_inputs:
            try:
                extractions.append(
                    extractor.extract(
                        file_id=item["file_id"],
                        sentences=item["sentences"],
                        prompt_text=item["prompt_text"],
                        shown_review_candidate_ids=item["shown_review_candidate_ids"],
                        base_appellant_ids=item["base_appellant_ids"],
                        base_respondent_ids=item["base_respondent_ids"],
                        original_rr_appellant_ids=item["original_rr_appellant_ids"],
                        original_rr_respondent_ids=item["original_rr_respondent_ids"],
                        review_candidate_lookup=item["review_candidate_lookup"],
                        candidate_source_mode=item["candidate_source_mode"],
                        max_retries=args.max_retries,
                    )
                )
            except Exception as exc:
                extractions.append({
                    **build_error_extraction(
                        error=exc,
                        base_appellant_ids=item["base_appellant_ids"],
                        base_respondent_ids=item["base_respondent_ids"],
                        candidate_source_mode=item["candidate_source_mode"],
                        original_rr_appellant_ids=item["original_rr_appellant_ids"],
                        original_rr_respondent_ids=item["original_rr_respondent_ids"],
                    ),
                    "raw_model_response": "",
                })
                errors += 1

    for item, extraction in zip(pending, extractions):
        annotation = item["annotation"]
        store_llm_arguments(
            annotation=annotation,
            extraction=extraction,
            args=args,
            sentence_count=item["sentence_count"],
            substantive_sentence_count=item["substantive_sentence_count"],
            review_candidate_sentence_count=item["review_candidate_sentence_count"],
            prompt_review_sentence_count=len(item["shown_review_candidate_ids"]),
            prompt_truncated=item["prompt_truncated"],
        )
        write_json(item["path"], annotation)
        print(f"  [OK] {item['file_id']}")

    return 0, errors


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
        help="Re-extract even if llm_arguments already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already have llm_arguments (default behavior unless --overwrite).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Parse files but skip LLM calls.")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=900,
        help="Max tokens for final appellant / respondent output.",
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=8,
        help="Number of cases per vLLM batch call.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max token context window for vLLM (case text + prompt).",
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

    paths = sorted(annotations_dir.glob("*.json"))

    # Filter files already processed unless --overwrite
    if not args.overwrite:
        to_process = []
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                to_process.append(path)
                continue
            if "llm_arguments" not in data:
                to_process.append(path)
        skipped_existing = len(paths) - len(to_process)
        if skipped_existing:
            print(f"Skipping {skipped_existing} files that already have llm_arguments "
                  f"(use --overwrite to re-extract).")
        paths = to_process

    if args.max_files is not None:
        paths = paths[: args.max_files]

    print(f"Files to process: {len(paths)}")

    if args.dry_run:
        print("[dry_run] Would process:")
        for path in paths:
            print(f"  {path.name}")
        return 0

    # Build extractor
    if args.backend == "local_vllm":
        extractor = LocalVLLMExtractor(
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
        extractor = RemoteHFExtractor(
            model_id=args.model_path,
            provider=args.provider,
            hf_token=args.hf_token,
            timeout=120.0,
        )

    pending_batch: list[dict[str, Any]] = []
    total_errors = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Extracting arguments")
    except ImportError:
        iterator = paths

    for path in iterator:
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [SKIP] {path.name}: cannot parse JSON — {exc}")
            continue

        file_id = str(annotation.get("file_id") or path.stem)
        sentences = annotation.get("sentences", [])
        sentence_count = len(sentences)
        substantive_count = count_substantive_sentences(sentences)
        prompt_bundle = build_user_prompt(file_id=file_id, annotation=annotation)
        review_candidate_count = prompt_bundle["review_candidate_count"]

        # --- Pre-filter: stub / garbled cases ---
        if sentence_count == 0 or review_candidate_count < MIN_REVIEW_CANDIDATE_SENTENCES:
            clear_llm_argument_roles(annotation)
            annotation["llm_arguments"] = {
                "schema_version": 3,
                "mode": "argument_side_refinement_only",
                "extraction_status": "skipped_short_case",
                "candidate_source_mode": prompt_bundle["candidate_source_mode"],
                "appellant_argument": NOT_FOUND_APPELLANT,
                "respondent_argument": NOT_FOUND_RESPONDENT,
                "confidence": "low",
                "sentence_count": sentence_count,
                "substantive_sentence_count": substantive_count,
                "review_candidate_sentence_count": review_candidate_count,
                "prompt_review_sentence_count": len(prompt_bundle["shown_review_candidate_ids"]),
                "prompt_truncated": prompt_bundle["prompt_truncated"],
                "model_path": args.model_path,
                "backend": args.backend,
                "original_appellant_sentence_ids": prompt_bundle["base_appellant_ids"],
                "original_respondent_sentence_ids": prompt_bundle["base_respondent_ids"],
                "original_rr_appellant_sentence_ids": prompt_bundle["original_rr_appellant_ids"],
                "original_rr_respondent_sentence_ids": prompt_bundle["original_rr_respondent_ids"],
                "note": (
                    f"Case has only {sentence_count} sentence(s) with "
                    f"{review_candidate_count} side-refinement candidate(s) from "
                    f"{prompt_bundle['candidate_source_mode']} — skipped LLM call."
                ),
            }
            write_json(path, annotation)
            print(f"  [SHORT] {file_id} ({sentence_count} sents, {review_candidate_count} review candidates)")
            continue

        if args.backend == "local_vllm":
            pending_batch.append({
                "path": path,
                "file_id": file_id,
                "sentences": sentences,
                "prompt_text": prompt_bundle["prompt_text"],
                "shown_review_candidate_ids": prompt_bundle["shown_review_candidate_ids"],
                "base_appellant_ids": prompt_bundle["base_appellant_ids"],
                "base_respondent_ids": prompt_bundle["base_respondent_ids"],
                "original_rr_appellant_ids": prompt_bundle["original_rr_appellant_ids"],
                "original_rr_respondent_ids": prompt_bundle["original_rr_respondent_ids"],
                "review_candidate_lookup": prompt_bundle["review_candidate_lookup"],
                "candidate_source_mode": prompt_bundle["candidate_source_mode"],
                "prompt_truncated": prompt_bundle["prompt_truncated"],
                "sentence_count": sentence_count,
                "substantive_sentence_count": substantive_count,
                "review_candidate_sentence_count": review_candidate_count,
                "annotation": annotation,
            })
            if len(pending_batch) >= args.generation_batch_size:
                _, errs = flush_vllm_batch(
                    pending=pending_batch,
                    extractor=extractor,
                    args=args,
                )
                total_errors += errs
                pending_batch = []
        else:
            try:
                extraction = extractor.extract(
                    file_id=file_id,
                    sentences=sentences,
                    prompt_text=prompt_bundle["prompt_text"],
                    shown_review_candidate_ids=prompt_bundle["shown_review_candidate_ids"],
                    base_appellant_ids=prompt_bundle["base_appellant_ids"],
                    base_respondent_ids=prompt_bundle["base_respondent_ids"],
                    original_rr_appellant_ids=prompt_bundle["original_rr_appellant_ids"],
                    original_rr_respondent_ids=prompt_bundle["original_rr_respondent_ids"],
                    review_candidate_lookup=prompt_bundle["review_candidate_lookup"],
                    candidate_source_mode=prompt_bundle["candidate_source_mode"],
                    max_retries=args.max_retries,
                )
                store_llm_arguments(
                    annotation=annotation,
                    extraction=extraction,
                    args=args,
                    sentence_count=sentence_count,
                    substantive_sentence_count=substantive_count,
                    review_candidate_sentence_count=review_candidate_count,
                    prompt_review_sentence_count=len(prompt_bundle["shown_review_candidate_ids"]),
                    prompt_truncated=prompt_bundle["prompt_truncated"],
                )
                write_json(path, annotation)
                print(f"  [OK] {file_id}")
            except Exception as exc:
                clear_llm_argument_roles(annotation)
                annotation["llm_arguments"] = {
                    "schema_version": 3,
                    "mode": "argument_side_refinement_only",
                    "extraction_status": "error",
                    "candidate_source_mode": prompt_bundle["candidate_source_mode"],
                    "appellant_argument": f"[Error: {exc}]",
                    "respondent_argument": f"[Error: {exc}]",
                    "confidence": "low",
                    "sentence_count": sentence_count,
                    "substantive_sentence_count": substantive_count,
                    "review_candidate_sentence_count": review_candidate_count,
                    "prompt_review_sentence_count": len(prompt_bundle["shown_review_candidate_ids"]),
                    "prompt_truncated": prompt_bundle["prompt_truncated"],
                    "model_path": args.model_path,
                    "backend": args.backend,
                    "original_appellant_sentence_ids": prompt_bundle["base_appellant_ids"],
                    "original_respondent_sentence_ids": prompt_bundle["base_respondent_ids"],
                    "original_rr_appellant_sentence_ids": prompt_bundle["original_rr_appellant_ids"],
                    "original_rr_respondent_sentence_ids": prompt_bundle["original_rr_respondent_ids"],
                    "error": str(exc),
                }
                write_json(path, annotation)
                print(f"  [ERROR] {file_id}: {exc}")
                total_errors += 1

    # Flush remaining vLLM batch
    if pending_batch:
        _, errs = flush_vllm_batch(
            pending=pending_batch,
            extractor=extractor,
            args=args,
        )
        total_errors += errs

    print(f"\nDone. Errors: {total_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
