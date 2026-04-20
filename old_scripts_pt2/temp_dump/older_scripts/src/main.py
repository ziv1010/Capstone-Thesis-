from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from .export import append_case_jsonl, write_case_json, write_cases_csv
from .legal_regex import extract_legal_references
from .llm_client import HuggingFaceLLMClient, build_llm_prompt, extract_case_json_with_retry
from .ner_extract import build_locations_from_entities, extract_entities, load_spacy_model
from .paragraphize import paragraphize
from .pdf_extract import extract_pdf_text
from .postprocess import apply_leakage_firewall, build_ml_input_text
from .schema import coerce_case_record, validate_case_record, validate_llm_output
from .utils import dedupe_str_list, ensure_output_dirs, list_pdf_files, load_config, stable_case_id

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return str(value).strip() or None


def _list_of_str(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return dedupe_str_list(out)


def _normalize_removed_spans(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = _str_or_none(item.get("text"))
        reason = _str_or_none(item.get("reason"))
        paragraph_index_raw = item.get("paragraph_index", -1)
        try:
            paragraph_index = int(paragraph_index_raw)
        except Exception:
            paragraph_index = -1

        if text is None:
            continue
        cleaned.append(
            {
                "paragraph_index": paragraph_index,
                "reason": reason or "outcome_phrase",
                "text": text,
            }
        )

    return cleaned


def _fallback_case_title(file_name: str) -> str:
    stem = Path(file_name).stem
    title = stem.replace("_", " ").strip()
    return re.sub(r"\s+", " ", title)


def _normalize_decision_label(value: Any) -> str | None:
    text = _str_or_none(value)
    if text is None:
        return None

    normalized = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    if not normalized:
        return None

    alias_map = {
        "for_the_appellant": "for_appellant",
        "in_favour_of_appellant": "for_appellant",
        "in_favor_of_appellant": "for_appellant",
        "in_favour_of_petitioner": "for_appellant",
        "in_favor_of_petitioner": "for_appellant",
        "for_the_petitioner": "for_appellant",
        "for_petitioner": "for_appellant",
        "for_the_applicant": "for_appellant",
        "for_applicant": "for_appellant",
        "against_the_appellant": "against_appellant",
        "against_the_petitioner": "against_appellant",
        "against_petitioner": "against_appellant",
        "against_the_applicant": "against_appellant",
        "against_applicant": "against_appellant",
        "in_favour_of_respondent": "against_appellant",
        "in_favor_of_respondent": "against_appellant",
        "allowed": "for_appellant",
        "granted": "for_appellant",
        "rejected": "against_appellant",
        "refused": "against_appellant",
        "denied": "against_appellant",
        "adjourned": "delayed",
        "deferred": "delayed",
        "postponed": "delayed",
    }
    return alias_map.get(normalized, normalized)


def _infer_decision_label(
    explicit_decision: Any,
    outcome_label: Any,
    outcome_winner: Any,
    decision_text: Any,
) -> str | None:
    explicit = _normalize_decision_label(explicit_decision)
    if explicit:
        return explicit

    winner = _str_or_none(outcome_winner)
    if winner:
        winner_cf = winner.casefold()
        if any(
            token in winner_cf
            for token in ("appellant", "petitioner", "applicant", "claimant", "assessee")
        ):
            return "for_appellant"
        if any(
            token in winner_cf
            for token in ("respondent", "state", "defendant", "prosecution", "union")
        ):
            return "against_appellant"

    label = _normalize_decision_label(outcome_label)
    if label:
        if "dismiss" in label:
            return "dismissed"
        if any(token in label for token in ("delay", "adjourn", "defer", "postpone")):
            return "delayed"
        if any(token in label for token in ("reject", "refus", "denied")):
            return "against_appellant"
        if any(token in label for token in ("allow", "grant", "succee")):
            return "for_appellant"
        return label

    decision_text_str = _str_or_none(decision_text)
    if decision_text_str:
        dt = decision_text_str.casefold()
        if any(token in dt for token in ("dismissed", "dismissal")):
            return "dismissed"
        if any(token in dt for token in ("rejected", "refused", "not inclined to grant")):
            return "against_appellant"
        if any(token in dt for token in ("granted", "allowed", "be released on bail")):
            return "for_appellant"
        if any(token in dt for token in ("adjourned", "stand over", "deferred", "postponed")):
            return "delayed"

    return None


def _sanitize_case_record(case_record: dict[str, Any]) -> dict[str, Any]:
    case_record["source"] = _str_or_none(case_record.get("source"))
    case_record["court"] = _str_or_none(case_record.get("court"))
    case_record["bench"] = _str_or_none(case_record.get("bench"))
    case_record["case_number"] = _str_or_none(case_record.get("case_number"))
    case_record["case_title"] = _str_or_none(case_record.get("case_title"))
    case_record["case_type"] = _str_or_none(case_record.get("case_type"))

    date_value = _str_or_none(case_record.get("date"))
    case_record["date"] = date_value if (date_value and DATE_RE.match(date_value)) else None

    case_record["judge_names"] = _list_of_str(case_record.get("judge_names"))

    parties = case_record.get("parties", {})
    case_record["parties"] = {
        "petitioner_applicant": _list_of_str(parties.get("petitioner_applicant")),
        "respondent_state_defendant": _list_of_str(
            parties.get("respondent_state_defendant")
        ),
    }

    advocates = case_record.get("advocates", {})
    case_record["advocates"] = {
        "for_petitioner_applicant": _list_of_str(advocates.get("for_petitioner_applicant")),
        "for_respondent_state_defendant": _list_of_str(
            advocates.get("for_respondent_state_defendant")
        ),
    }

    locations = case_record.get("locations", {})
    case_record["locations"] = {
        "gpe": _list_of_str(locations.get("gpe")),
        "org": _list_of_str(locations.get("org")),
    }

    case_record["provisions"] = _list_of_str(case_record.get("provisions"))
    case_record["statutes"] = _list_of_str(case_record.get("statutes"))
    case_record["precedents"] = _list_of_str(case_record.get("precedents"))

    entities = case_record.get("entities", {})
    case_record["entities"] = {
        "PERSON": _list_of_str(entities.get("PERSON")),
        "ORG": _list_of_str(entities.get("ORG")),
        "GPE": _list_of_str(entities.get("GPE")),
        "DATE": _list_of_str(entities.get("DATE")),
    }

    texts = case_record.get("texts", {})
    case_record["texts"] = {
        "raw_text": str(texts.get("raw_text") or ""),
        "facts_text": _str_or_none(texts.get("facts_text")),
        "arguments_petitioner": _str_or_none(texts.get("arguments_petitioner")),
        "arguments_respondent": _str_or_none(texts.get("arguments_respondent")),
        "reasoning_text": _str_or_none(texts.get("reasoning_text")),
        "decision_text": _str_or_none(texts.get("decision_text")),
    }

    case_record["key_facts_bullets"] = _list_of_str(case_record.get("key_facts_bullets"))
    case_record["issues_bullets"] = _list_of_str(case_record.get("issues_bullets"))
    case_record["holdings_bullets"] = _list_of_str(case_record.get("holdings_bullets"))

    outcome = case_record.get("outcome", {})
    case_record["outcome"] = {
        "label": _str_or_none(outcome.get("label")),
        "winner": _str_or_none(outcome.get("winner")),
    }
    case_record["decision"] = _infer_decision_label(
        explicit_decision=case_record.get("decision"),
        outcome_label=case_record["outcome"].get("label"),
        outcome_winner=case_record["outcome"].get("winner"),
        decision_text=case_record["texts"].get("decision_text"),
    )

    ml = case_record.get("ml", {})
    case_record["ml"] = {
        "input_text": str(ml.get("input_text") or ""),
        "removed_spans": _normalize_removed_spans(ml.get("removed_spans")),
        "leakage_flag": bool(ml.get("leakage_flag", False)),
        "leaked_phrases_found": _list_of_str(ml.get("leaked_phrases_found")),
    }

    return case_record


def _load_existing_records_for_resume(json_dir: Path) -> tuple[set[str], list[dict[str, Any]]]:
    existing_case_ids: set[str] = set()
    existing_records: list[dict[str, Any]] = []

    for json_path in sorted(json_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError("JSON record is not an object")

            case_id = _str_or_none(loaded.get("case_id")) or json_path.stem
            file_name = _str_or_none(loaded.get("file_name")) or f"{json_path.stem}.pdf"
            record = coerce_case_record(loaded, case_id=case_id, file_name=file_name)
            record = _sanitize_case_record(record)
            validate_case_record(record)

            existing_case_ids.add(case_id)
            existing_records.append(record)
        except Exception as exc:
            print(
                f"[WARN] Skipping invalid existing JSON during resume: {json_path.name} ({exc})",
                flush=True,
            )

    return existing_case_ids, existing_records


def _enrich_with_deterministic_extractors(
    case_record: dict[str, Any],
    raw_text: str,
    entities: dict[str, list[str]],
    legal_refs: dict[str, list[str]],
) -> dict[str, Any]:
    case_record["texts"]["raw_text"] = raw_text

    for label in ("PERSON", "ORG", "GPE", "DATE"):
        combined = case_record["entities"].get(label, []) + entities.get(label, [])
        case_record["entities"][label] = dedupe_str_list(combined)

    derived_locations = build_locations_from_entities(case_record["entities"])
    case_record["locations"]["gpe"] = dedupe_str_list(
        case_record["locations"].get("gpe", []) + derived_locations["gpe"]
    )
    case_record["locations"]["org"] = dedupe_str_list(
        case_record["locations"].get("org", []) + derived_locations["org"]
    )

    case_record["provisions"] = dedupe_str_list(
        case_record.get("provisions", []) + legal_refs.get("provisions", [])
    )
    case_record["statutes"] = dedupe_str_list(
        case_record.get("statutes", []) + legal_refs.get("statutes", [])
    )
    case_record["precedents"] = dedupe_str_list(
        case_record.get("precedents", []) + legal_refs.get("precedents", [])
    )

    if not case_record.get("case_title"):
        case_record["case_title"] = _fallback_case_title(case_record["file_name"])

    if not case_record["ml"].get("input_text"):
        case_record["ml"]["input_text"] = build_ml_input_text(case_record["texts"])

    return case_record


def _chunk_paragraphs_for_llm(
    paragraphs: list[dict[str, Any]],
    max_chunk_chars: int,
) -> list[list[dict[str, Any]]]:
    if max_chunk_chars <= 0:
        return [paragraphs]

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_chars = 0

    for para in paragraphs:
        text = str(para.get("text", "")).strip()
        if not text:
            continue
        para_size = len(text) + 16  # include separators + small metadata overhead

        if current and current_chars + para_size > max_chunk_chars:
            chunks.append(current)
            current = []
            current_chars = 0

        if para_size > max_chunk_chars:
            start = 0
            while start < len(text):
                piece = text[start : start + max_chunk_chars]
                chunks.append([{"index": int(para.get("index", -1)), "text": piece}])
                start += max_chunk_chars
            continue

        current.append({"index": int(para.get("index", -1)), "text": text})
        current_chars += para_size

    if current:
        chunks.append(current)

    if not chunks:
        return [[]]
    return chunks


def _merge_optional_text(old_value: Any, new_value: Any, max_chars: int = 0) -> str | None:
    old_text = _str_or_none(old_value)
    new_text = _str_or_none(new_value)
    if old_text is None and new_text is None:
        return None
    if old_text is None:
        out = new_text
    elif new_text is None:
        out = old_text
    else:
        old_cf = old_text.casefold()
        new_cf = new_text.casefold()
        if new_cf in old_cf:
            out = old_text
        elif old_cf in new_cf:
            out = new_text
        else:
            out = f"{old_text}\n\n{new_text}"

    if out is not None and max_chars > 0 and len(out) > max_chars:
        return out[:max_chars]
    return out


def _merge_removed_spans(old_value: Any, new_value: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()

    for bucket in (old_value, new_value):
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            text = _str_or_none(item.get("text"))
            if text is None:
                continue
            try:
                paragraph_index = int(item.get("paragraph_index", -1))
            except Exception:
                paragraph_index = -1
            reason = _str_or_none(item.get("reason")) or "outcome_phrase"
            key = (paragraph_index, reason.casefold(), text.casefold())
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                {
                    "paragraph_index": paragraph_index,
                    "reason": reason,
                    "text": text,
                }
            )
    return merged


def _merge_llm_candidate(
    current: dict[str, Any],
    incoming: dict[str, Any],
    text_max_chars: int,
) -> dict[str, Any]:
    if not current:
        return dict(incoming)
    if not incoming:
        return dict(current)

    out = dict(current)

    for key in ("source", "court", "bench", "case_number", "case_title", "case_type", "date"):
        out[key] = _str_or_none(out.get(key)) or _str_or_none(incoming.get(key))

    out["judge_names"] = dedupe_str_list(
        _list_of_str(out.get("judge_names")) + _list_of_str(incoming.get("judge_names"))
    )

    current_parties = out.get("parties", {})
    incoming_parties = incoming.get("parties", {})
    out["parties"] = {
        "petitioner_applicant": dedupe_str_list(
            _list_of_str(current_parties.get("petitioner_applicant"))
            + _list_of_str(incoming_parties.get("petitioner_applicant"))
        ),
        "respondent_state_defendant": dedupe_str_list(
            _list_of_str(current_parties.get("respondent_state_defendant"))
            + _list_of_str(incoming_parties.get("respondent_state_defendant"))
        ),
    }

    current_advocates = out.get("advocates", {})
    incoming_advocates = incoming.get("advocates", {})
    out["advocates"] = {
        "for_petitioner_applicant": dedupe_str_list(
            _list_of_str(current_advocates.get("for_petitioner_applicant"))
            + _list_of_str(incoming_advocates.get("for_petitioner_applicant"))
        ),
        "for_respondent_state_defendant": dedupe_str_list(
            _list_of_str(current_advocates.get("for_respondent_state_defendant"))
            + _list_of_str(incoming_advocates.get("for_respondent_state_defendant"))
        ),
    }

    current_locations = out.get("locations", {})
    incoming_locations = incoming.get("locations", {})
    out["locations"] = {
        "gpe": dedupe_str_list(
            _list_of_str(current_locations.get("gpe")) + _list_of_str(incoming_locations.get("gpe"))
        ),
        "org": dedupe_str_list(
            _list_of_str(current_locations.get("org")) + _list_of_str(incoming_locations.get("org"))
        ),
    }

    for key in ("provisions", "statutes", "precedents", "key_facts_bullets", "issues_bullets", "holdings_bullets"):
        out[key] = dedupe_str_list(_list_of_str(out.get(key)) + _list_of_str(incoming.get(key)))

    current_entities = out.get("entities", {})
    incoming_entities = incoming.get("entities", {})
    out["entities"] = {
        "PERSON": dedupe_str_list(
            _list_of_str(current_entities.get("PERSON")) + _list_of_str(incoming_entities.get("PERSON"))
        ),
        "ORG": dedupe_str_list(
            _list_of_str(current_entities.get("ORG")) + _list_of_str(incoming_entities.get("ORG"))
        ),
        "GPE": dedupe_str_list(
            _list_of_str(current_entities.get("GPE")) + _list_of_str(incoming_entities.get("GPE"))
        ),
        "DATE": dedupe_str_list(
            _list_of_str(current_entities.get("DATE")) + _list_of_str(incoming_entities.get("DATE"))
        ),
    }

    current_texts = out.get("texts", {})
    incoming_texts = incoming.get("texts", {})
    out["texts"] = {
        "raw_text": str(current_texts.get("raw_text") or ""),
        "facts_text": _merge_optional_text(
            current_texts.get("facts_text"),
            incoming_texts.get("facts_text"),
            max_chars=text_max_chars,
        ),
        "arguments_petitioner": _merge_optional_text(
            current_texts.get("arguments_petitioner"),
            incoming_texts.get("arguments_petitioner"),
            max_chars=text_max_chars,
        ),
        "arguments_respondent": _merge_optional_text(
            current_texts.get("arguments_respondent"),
            incoming_texts.get("arguments_respondent"),
            max_chars=text_max_chars,
        ),
        "reasoning_text": _merge_optional_text(
            current_texts.get("reasoning_text"),
            incoming_texts.get("reasoning_text"),
            max_chars=text_max_chars,
        ),
        "decision_text": _merge_optional_text(
            current_texts.get("decision_text"),
            incoming_texts.get("decision_text"),
            max_chars=text_max_chars,
        ),
    }

    current_outcome = out.get("outcome", {})
    incoming_outcome = incoming.get("outcome", {})
    out["outcome"] = {
        "label": _str_or_none(incoming_outcome.get("label")) or _str_or_none(current_outcome.get("label")),
        "winner": _str_or_none(incoming_outcome.get("winner"))
        or _str_or_none(current_outcome.get("winner")),
    }
    out["decision"] = _str_or_none(incoming.get("decision")) or _str_or_none(out.get("decision"))

    current_ml = out.get("ml", {})
    incoming_ml = incoming.get("ml", {})
    out["ml"] = {
        "input_text": _merge_optional_text(
            current_ml.get("input_text"),
            incoming_ml.get("input_text"),
            max_chars=text_max_chars * 2 if text_max_chars > 0 else 0,
        )
        or "",
        "removed_spans": _merge_removed_spans(
            current_ml.get("removed_spans"),
            incoming_ml.get("removed_spans"),
        ),
        "leakage_flag": bool(current_ml.get("leakage_flag", False))
        or bool(incoming_ml.get("leakage_flag", False)),
        "leaked_phrases_found": dedupe_str_list(
            _list_of_str(current_ml.get("leaked_phrases_found"))
            + _list_of_str(incoming_ml.get("leaked_phrases_found"))
        ),
    }

    return out


def process_one_pdf(
    pdf_path: Path,
    config: dict[str, Any],
    nlp: Any,
    llm_client: HuggingFaceLLMClient | None,
    skip_llm: bool,
) -> dict[str, Any]:
    file_name = pdf_path.name
    case_id = stable_case_id(file_name)
    pdf_start = time.perf_counter()

    t0 = time.perf_counter()
    print(f"[STEP] {file_name} | extracting PDF text", flush=True)
    extraction = extract_pdf_text(pdf_path, mode=str(config.get("pdf_extraction_mode", "pymupdf")))
    print(
        f"[STEP] {file_name} | extracted text in {time.perf_counter() - t0:.1f}s "
        f"({len(extraction.raw_text):,} chars)",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[STEP] {file_name} | paragraphizing", flush=True)
    paragraphs = paragraphize(extraction.raw_text)
    print(
        f"[STEP] {file_name} | paragraphized in {time.perf_counter() - t0:.1f}s "
        f"({len(paragraphs):,} paragraphs)",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[STEP] {file_name} | NER extraction", flush=True)
    ner_entities = extract_entities(
        extraction.raw_text,
        nlp,
        max_chars=int(config.get("ner_max_chars", 0) or 0),
    )
    ner_count = sum(len(v) for v in ner_entities.values())
    print(
        f"[STEP] {file_name} | NER done in {time.perf_counter() - t0:.1f}s "
        f"({ner_count:,} entities)",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[STEP] {file_name} | regex references", flush=True)
    legal_refs = extract_legal_references(extraction.raw_text)
    refs_count = sum(len(v) for v in legal_refs.values())
    print(
        f"[STEP] {file_name} | regex done in {time.perf_counter() - t0:.1f}s "
        f"({refs_count:,} refs)",
        flush=True,
    )

    llm_candidate: dict[str, Any] = {}
    if not skip_llm and llm_client is not None:
        chunking_enabled = bool(config.get("llm_full_document_chunking", True))
        chunk_chars = int(config.get("llm_chunk_chars", 12000) or 12000)
        text_max_chars = int(config.get("llm_text_field_max_chars", 6000) or 6000)

        if chunking_enabled:
            llm_chunks = _chunk_paragraphs_for_llm(paragraphs, max_chunk_chars=chunk_chars)
            total_chunks = len(llm_chunks)
            print(
                f"[STEP] {file_name} | full-document LLM mode enabled "
                f"({total_chunks} chunks, chunk_chars={chunk_chars})",
                flush=True,
            )
            extraction_history: list[dict[str, Any]] = []

            for chunk_idx, chunk_paragraphs in enumerate(llm_chunks, start=1):
                chunk_start = time.perf_counter()
                chunk_text = "\n\n".join(str(p.get("text", "")).strip() for p in chunk_paragraphs if p)
                print(
                    f"[STEP] {file_name} | chunk {chunk_idx}/{total_chunks} "
                    f"({len(chunk_paragraphs):,} paragraphs, {len(chunk_text):,} chars)",
                    flush=True,
                )

                t0 = time.perf_counter()
                prompt = build_llm_prompt(
                    raw_text=chunk_text,
                    paragraphs=chunk_paragraphs,
                    ner=ner_entities,
                    regex=legal_refs,
                    max_chars_to_send=0,
                )
                print(
                    f"[STEP] {file_name} | chunk {chunk_idx}/{total_chunks} prompt built "
                    f"in {time.perf_counter() - t0:.1f}s ({len(prompt):,} chars)",
                    flush=True,
                )

                try:
                    chunk_candidate = extract_case_json_with_retry(
                        client=llm_client,
                        prompt=prompt,
                        validate_fn=validate_llm_output,
                        max_attempts=int(config.get("llm_attempts_per_chunk", 4) or 4),
                    )
                    llm_candidate = _merge_llm_candidate(
                        current=llm_candidate,
                        incoming=chunk_candidate,
                        text_max_chars=text_max_chars,
                    )
                    extraction_history.append(
                        {
                            "chunk_index": chunk_idx,
                            "status": "ok",
                            "elapsed_sec": round(time.perf_counter() - chunk_start, 2),
                            "chunk_chars": len(chunk_text),
                            "keys": sorted(chunk_candidate.keys()),
                        }
                    )
                    print(
                        f"[STEP] {file_name} | chunk {chunk_idx}/{total_chunks} merged "
                        f"in {time.perf_counter() - chunk_start:.1f}s",
                        flush=True,
                    )
                except Exception as exc:
                    extraction_history.append(
                        {
                            "chunk_index": chunk_idx,
                            "status": "failed",
                            "elapsed_sec": round(time.perf_counter() - chunk_start, 2),
                            "chunk_chars": len(chunk_text),
                            "error": repr(exc),
                        }
                    )
                    print(
                        f"[WARN] {file_name} | chunk {chunk_idx}/{total_chunks} LLM failed: {repr(exc)}",
                        flush=True,
                    )
                    traceback.print_exc()

            succeeded = sum(1 for h in extraction_history if h.get("status") == "ok")
            failed = total_chunks - succeeded
            print(
                f"[STEP] {file_name} | chunked LLM finished: "
                f"{succeeded}/{total_chunks} chunks succeeded, {failed} failed",
                flush=True,
            )
        else:
            t0 = time.perf_counter()
            print(f"[STEP] {file_name} | building LLM prompt", flush=True)
            prompt = build_llm_prompt(
                raw_text=extraction.raw_text,
                paragraphs=paragraphs,
                ner=ner_entities,
                regex=legal_refs,
                max_chars_to_send=int(config.get("max_chars_to_send", 0) or 0),
            )
            print(
                f"[STEP] {file_name} | prompt built in {time.perf_counter() - t0:.1f}s "
                f"({len(prompt):,} chars)",
                flush=True,
            )
            try:
                t0 = time.perf_counter()
                print(f"[STEP] {file_name} | LLM extraction", flush=True)
                llm_candidate = extract_case_json_with_retry(
                    client=llm_client,
                    prompt=prompt,
                    validate_fn=validate_llm_output,
                    max_attempts=int(config.get("llm_attempts_per_chunk", 4) or 4),
                )
                print(
                    f"[STEP] {file_name} | LLM extraction done in {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"[WARN] LLM extraction failed for {file_name}: {repr(exc)}")
                traceback.print_exc()
                print(f"[STEP] {file_name} | falling back to deterministic extractors", flush=True)

    case_record = coerce_case_record(llm_candidate, case_id=case_id, file_name=file_name)
    case_record = _sanitize_case_record(case_record)
    case_record = _enrich_with_deterministic_extractors(
        case_record=case_record,
        raw_text=extraction.raw_text,
        entities=ner_entities,
        legal_refs=legal_refs,
    )

    case_record = apply_leakage_firewall(
        case_record=case_record,
        paragraphs=paragraphs,
        leakage_phrases=list(config.get("leakage_phrases", [])),
    )

    # Apply deterministic IndianKanoon header fields (overrides null LLM values only)
    if extraction.source_url and not case_record.get("source"):
        case_record["source"] = extraction.source_url
    if extraction.ik_author and not case_record.get("judge_names"):
        case_record["judge_names"] = [extraction.ik_author]
    if extraction.ik_bench and not case_record.get("bench"):
        case_record["bench"] = extraction.ik_bench

    # Re-sanitize in case firewall changed structures.
    case_record = _sanitize_case_record(case_record)
    validate_case_record(case_record)
    print(f"[STEP] {file_name} | completed in {time.perf_counter() - pdf_start:.1f}s", flush=True)
    return case_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible extraction pipeline for Indian court PDF judgments/orders"
    )
    parser.add_argument("--pdf_dir", default="./data/pdfs", help="Input directory containing PDF files")
    parser.add_argument("--out_dir", default="./outputs", help="Output directory")
    parser.add_argument("--config", default="./configs/config.yaml", help="YAML config path")
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help="Optional override for CUDA_VISIBLE_DEVICES, e.g. '6,7'",
    )
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM extraction and run parser+NER+regex+exports only",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume by skipping PDFs whose JSON already exists in outputs/json (default: enabled)",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="Disable resume and reprocess all PDFs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    out_root, json_dir = ensure_output_dirs(args.out_dir)
    jsonl_path = out_root / "cases.jsonl"
    csv_path = out_root / "cases.csv"

    all_pdf_files = list_pdf_files(args.pdf_dir)
    if not all_pdf_files:
        print(f"[INFO] No PDF files found in {args.pdf_dir}")
        return

    existing_case_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    if args.resume:
        existing_case_ids, records = _load_existing_records_for_resume(json_dir)
        if existing_case_ids:
            print(
                f"[INFO] Resume enabled: found {len(existing_case_ids)} existing JSON records in {json_dir}",
                flush=True,
            )

    pdf_files: list[Path] = []
    skipped_existing = 0
    for pdf_path in all_pdf_files:
        case_id = stable_case_id(pdf_path.name)
        if args.resume and case_id in existing_case_ids:
            skipped_existing += 1
            continue
        pdf_files.append(pdf_path)

    print(
        f"[INFO] Found {len(all_pdf_files)} PDF files "
        f"({skipped_existing} already completed, {len(pdf_files)} pending)",
        flush=True,
    )

    if jsonl_path.exists():
        jsonl_path.unlink()
    for existing_record in records:
        append_case_jsonl(existing_record, jsonl_path)

    if not pdf_files:
        if records:
            write_cases_csv(records, csv_path)
            print(
                f"[DONE] Nothing pending. Rebuilt exports with {len(records)} records at {out_root}",
                flush=True,
            )
        else:
            print("[INFO] Nothing pending and no existing records were loaded", flush=True)
        return

    nlp = load_spacy_model(str(config.get("spacy_model", "en_core_web_sm")))

    llm_client: HuggingFaceLLMClient | None = None
    if not args.skip_llm:
        hf_home = str(config.get("hf_home", "")).strip()
        if hf_home:
            hf_home_path = Path(hf_home)
            hf_home_path.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(hf_home_path)
            # Keep hub/transformers cache co-located under HF_HOME.
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home_path / "hub")
            os.environ["TRANSFORMERS_CACHE"] = str(hf_home_path / "hub")
            print(f"[INFO] Using HF_HOME={hf_home_path}")

        cuda_visible_devices = args.cuda_visible_devices
        if cuda_visible_devices is None:
            cuda_visible_devices = str(config.get("llm_cuda_visible_devices", "")).strip() or None
        if cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
            print(f"[INFO] Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

        llm_client = HuggingFaceLLMClient(
            model_name=str(config.get("llm_model_name")),
            temperature=float(config.get("llm_temperature", 0.0)),
            max_new_tokens=int(config.get("llm_max_new_tokens", 4096)),
            device_map=str(config.get("llm_device_map", "auto")),
            torch_dtype=str(config.get("llm_torch_dtype", "bfloat16")),
            trust_remote_code=bool(config.get("llm_trust_remote_code", False)),
        )
        print(
            "[INFO] LLM config: "
            f"model={llm_client.model_name}, "
            f"device_map={llm_client.device_map}, "
            f"dtype={llm_client.torch_dtype}, "
            f"max_new_tokens={llm_client.max_new_tokens}, "
            f"max_chars_to_send={int(config.get('max_chars_to_send', 0) or 0)}, "
            f"full_document_chunking={bool(config.get('llm_full_document_chunking', True))}, "
            f"chunk_chars={int(config.get('llm_chunk_chars', 12000) or 12000)}, "
            f"attempts_per_chunk={int(config.get('llm_attempts_per_chunk', 4) or 4)}",
            flush=True,
        )

    new_records = 0
    total_files = len(pdf_files)
    for idx, pdf_file in enumerate(pdf_files, start=1):
        print(f"[PROGRESS] [{idx}/{total_files}] Starting {pdf_file.name}", flush=True)
        file_start = time.perf_counter()
        try:
            record = process_one_pdf(
                pdf_path=pdf_file,
                config=config,
                nlp=nlp,
                llm_client=llm_client,
                skip_llm=args.skip_llm,
            )
            write_case_json(record, json_dir)
            append_case_jsonl(record, jsonl_path)
            records.append(record)
            new_records += 1
            print(
                f"[OK] Processed {pdf_file.name} in {time.perf_counter() - file_start:.1f}s",
                flush=True,
            )
        except Exception as exc:
            print(f"[ERROR] Failed {pdf_file.name}: {exc}")

    if records:
        write_cases_csv(records, csv_path)
        print(
            f"[DONE] Wrote {len(records)} records to {out_root} "
            f"({new_records} new, {len(records) - new_records} pre-existing)",
            flush=True,
        )
    else:
        print("[INFO] No records written")


if __name__ == "__main__":
    main()
