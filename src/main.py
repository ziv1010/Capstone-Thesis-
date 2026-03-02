from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .export import append_case_jsonl, write_case_json, write_cases_csv
from .legal_regex import extract_legal_references
from .llm_client import LocalHTTPLLMClient, build_llm_prompt, extract_case_json_with_retry
from .ner_extract import build_locations_from_entities, extract_entities, load_spacy_model
from .paragraphize import paragraphize
from .pdf_extract import extract_pdf_text
from .postprocess import apply_leakage_firewall, build_ml_input_text
from .schema import coerce_case_record, validate_case_record
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

    ml = case_record.get("ml", {})
    case_record["ml"] = {
        "input_text": str(ml.get("input_text") or ""),
        "removed_spans": _normalize_removed_spans(ml.get("removed_spans")),
        "leakage_flag": bool(ml.get("leakage_flag", False)),
        "leaked_phrases_found": _list_of_str(ml.get("leaked_phrases_found")),
    }

    return case_record


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


def process_one_pdf(
    pdf_path: Path,
    config: dict[str, Any],
    nlp: Any,
    llm_client: LocalHTTPLLMClient | None,
    skip_llm: bool,
) -> dict[str, Any]:
    file_name = pdf_path.name
    case_id = stable_case_id(file_name)

    extraction = extract_pdf_text(pdf_path, mode=str(config.get("pdf_extraction_mode", "pymupdf")))
    paragraphs = paragraphize(extraction.raw_text)
    ner_entities = extract_entities(
        extraction.raw_text,
        nlp,
        max_chars=int(config.get("ner_max_chars", 0) or 0),
    )
    legal_refs = extract_legal_references(extraction.raw_text)

    llm_candidate: dict[str, Any] = {}
    if not skip_llm and llm_client is not None:
        prompt = build_llm_prompt(
            raw_text=extraction.raw_text,
            paragraphs=paragraphs,
            ner=ner_entities,
            regex=legal_refs,
            max_chars_to_send=int(config.get("max_chars_to_send", 0) or 0),
        )
        try:
            llm_candidate = extract_case_json_with_retry(
                client=llm_client,
                prompt=prompt,
                validate_fn=validate_case_record,
            )
        except Exception as exc:
            print(f"[WARN] LLM extraction failed for {file_name}: {exc}")

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
    return case_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible extraction pipeline for Indian court PDF judgments/orders"
    )
    parser.add_argument("--pdf_dir", default="./data/pdfs", help="Input directory containing PDF files")
    parser.add_argument("--out_dir", default="./outputs", help="Output directory")
    parser.add_argument("--config", default="./configs/config.yaml", help="YAML config path")
    parser.add_argument(
        "--skip_llm",
        action="store_true",
        help="Skip LLM extraction and run parser+NER+regex+exports only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    out_root, json_dir = ensure_output_dirs(args.out_dir)
    jsonl_path = out_root / "cases.jsonl"
    csv_path = out_root / "cases.csv"

    if jsonl_path.exists():
        jsonl_path.unlink()

    pdf_files = list_pdf_files(args.pdf_dir)
    if not pdf_files:
        print(f"[INFO] No PDF files found in {args.pdf_dir}")
        return

    nlp = load_spacy_model(str(config.get("spacy_model", "en_core_web_sm")))

    llm_client: LocalHTTPLLMClient | None = None
    if not args.skip_llm:
        llm_client = LocalHTTPLLMClient(
            endpoint_url=str(config.get("llm_endpoint_url")),
            model_name=str(config.get("llm_model_name")),
            timeout_seconds=int(config.get("llm_timeout_seconds", 120)),
            temperature=float(config.get("llm_temperature", 0.0)),
        )

    records: list[dict[str, Any]] = []
    for pdf_file in pdf_files:
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
            print(f"[OK] Processed {pdf_file.name}")
        except Exception as exc:
            print(f"[ERROR] Failed {pdf_file.name}: {exc}")

    if records:
        write_cases_csv(records, csv_path)
        print(f"[DONE] Wrote {len(records)} records to {out_root}")
    else:
        print("[INFO] No records written")


if __name__ == "__main__":
    main()
