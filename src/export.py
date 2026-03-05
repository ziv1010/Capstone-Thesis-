from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .utils import json_string

CSV_COLUMNS = [
    "file_name",
    "case_id",
    "court",
    "bench",
    "judge_names",
    "date",
    "case_number",
    "case_title",
    "case_type",
    "petitioner_applicant",
    "respondent_state_defendant",
    "advocates_petitioner",
    "advocates_respondent",
    "provisions",
    "statutes",
    "precedents",
    "org",
    "gpe",
    "persons",
    "key_facts_bullets",
    "facts_text",
    "arguments_petitioner",
    "arguments_respondent",
    "reasoning_text",
    "decision_text",
    "decision",
    "outcome_label",
    "outcome_winner",
    "input_text",
    "leakage_flag",
    "leaked_phrases_found",
]


def write_case_json(case_record: dict[str, Any], json_dir: Path) -> Path:
    output_path = json_dir / f"{case_record['case_id']}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(case_record, f, ensure_ascii=False, indent=2)
    return output_path


def append_case_jsonl(case_record: dict[str, Any], jsonl_path: Path) -> None:
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(case_record, ensure_ascii=False) + "\n")


def _flatten_case(case_record: dict[str, Any]) -> dict[str, Any]:
    parties = case_record.get("parties", {})
    advocates = case_record.get("advocates", {})
    entities = case_record.get("entities", {})
    locations = case_record.get("locations", {})
    texts = case_record.get("texts", {})
    outcome = case_record.get("outcome", {})
    ml = case_record.get("ml", {})

    flattened = {
        "file_name": case_record.get("file_name"),
        "case_id": case_record.get("case_id"),
        "court": case_record.get("court"),
        "bench": case_record.get("bench"),
        "judge_names": json_string(case_record.get("judge_names", [])),
        "date": case_record.get("date"),
        "case_number": case_record.get("case_number"),
        "case_title": case_record.get("case_title"),
        "case_type": case_record.get("case_type"),
        "petitioner_applicant": json_string(parties.get("petitioner_applicant", [])),
        "respondent_state_defendant": json_string(
            parties.get("respondent_state_defendant", [])
        ),
        "advocates_petitioner": json_string(advocates.get("for_petitioner_applicant", [])),
        "advocates_respondent": json_string(
            advocates.get("for_respondent_state_defendant", [])
        ),
        "provisions": json_string(case_record.get("provisions", [])),
        "statutes": json_string(case_record.get("statutes", [])),
        "precedents": json_string(case_record.get("precedents", [])),
        "org": json_string(locations.get("org", [])),
        "gpe": json_string(locations.get("gpe", [])),
        "persons": json_string(entities.get("PERSON", [])),
        "key_facts_bullets": json_string(case_record.get("key_facts_bullets", [])),
        "facts_text": texts.get("facts_text"),
        "arguments_petitioner": texts.get("arguments_petitioner"),
        "arguments_respondent": texts.get("arguments_respondent"),
        "reasoning_text": texts.get("reasoning_text"),
        "decision_text": texts.get("decision_text"),
        "decision": case_record.get("decision"),
        "outcome_label": outcome.get("label"),
        "outcome_winner": outcome.get("winner"),
        "input_text": ml.get("input_text"),
        "leakage_flag": ml.get("leakage_flag"),
        "leaked_phrases_found": json_string(ml.get("leaked_phrases_found", [])),
    }
    return flattened


def write_cases_csv(case_records: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case_record in case_records:
            writer.writerow(_flatten_case(case_record))
