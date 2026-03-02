from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

CASE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "case_id",
        "file_name",
        "source",
        "court",
        "bench",
        "judge_names",
        "date",
        "case_number",
        "case_title",
        "case_type",
        "parties",
        "advocates",
        "locations",
        "provisions",
        "statutes",
        "precedents",
        "entities",
        "texts",
        "key_facts_bullets",
        "issues_bullets",
        "holdings_bullets",
        "outcome",
        "ml",
    ],
    "properties": {
        "case_id": {"type": "string"},
        "file_name": {"type": "string"},
        "source": {"type": ["string", "null"]},
        "court": {"type": ["string", "null"]},
        "bench": {"type": ["string", "null"]},
        "judge_names": {"type": "array", "items": {"type": "string"}},
        "date": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
            ]
        },
        "case_number": {"type": ["string", "null"]},
        "case_title": {"type": ["string", "null"]},
        "case_type": {"type": ["string", "null"]},
        "parties": {
            "type": "object",
            "additionalProperties": False,
            "required": ["petitioner_applicant", "respondent_state_defendant"],
            "properties": {
                "petitioner_applicant": {"type": "array", "items": {"type": "string"}},
                "respondent_state_defendant": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "advocates": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "for_petitioner_applicant",
                "for_respondent_state_defendant",
            ],
            "properties": {
                "for_petitioner_applicant": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "for_respondent_state_defendant": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "locations": {
            "type": "object",
            "additionalProperties": False,
            "required": ["gpe", "org"],
            "properties": {
                "gpe": {"type": "array", "items": {"type": "string"}},
                "org": {"type": "array", "items": {"type": "string"}},
            },
        },
        "provisions": {"type": "array", "items": {"type": "string"}},
        "statutes": {"type": "array", "items": {"type": "string"}},
        "precedents": {"type": "array", "items": {"type": "string"}},
        "entities": {
            "type": "object",
            "additionalProperties": False,
            "required": ["PERSON", "ORG", "GPE", "DATE"],
            "properties": {
                "PERSON": {"type": "array", "items": {"type": "string"}},
                "ORG": {"type": "array", "items": {"type": "string"}},
                "GPE": {"type": "array", "items": {"type": "string"}},
                "DATE": {"type": "array", "items": {"type": "string"}},
            },
        },
        "texts": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "raw_text",
                "facts_text",
                "arguments_petitioner",
                "arguments_respondent",
                "reasoning_text",
                "decision_text",
            ],
            "properties": {
                "raw_text": {"type": "string"},
                "facts_text": {"type": ["string", "null"]},
                "arguments_petitioner": {"type": ["string", "null"]},
                "arguments_respondent": {"type": ["string", "null"]},
                "reasoning_text": {"type": ["string", "null"]},
                "decision_text": {"type": ["string", "null"]},
            },
        },
        "key_facts_bullets": {"type": "array", "items": {"type": "string"}},
        "issues_bullets": {"type": "array", "items": {"type": "string"}},
        "holdings_bullets": {"type": "array", "items": {"type": "string"}},
        "outcome": {
            "type": "object",
            "additionalProperties": False,
            "required": ["label", "winner"],
            "properties": {
                "label": {"type": ["string", "null"]},
                "winner": {"type": ["string", "null"]},
            },
        },
        "ml": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "input_text",
                "removed_spans",
                "leakage_flag",
                "leaked_phrases_found",
            ],
            "properties": {
                "input_text": {"type": "string"},
                "removed_spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["paragraph_index", "reason", "text"],
                        "properties": {
                            "paragraph_index": {"type": "integer"},
                            "reason": {"type": "string"},
                            "text": {"type": "string"},
                        },
                    },
                },
                "leakage_flag": {"type": "boolean"},
                "leaked_phrases_found": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}

SCHEMA_VALIDATOR = Draft202012Validator(CASE_JSON_SCHEMA)


def empty_case_record(case_id: str, file_name: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "file_name": file_name,
        "source": None,
        "court": None,
        "bench": None,
        "judge_names": [],
        "date": None,
        "case_number": None,
        "case_title": None,
        "case_type": None,
        "parties": {
            "petitioner_applicant": [],
            "respondent_state_defendant": [],
        },
        "advocates": {
            "for_petitioner_applicant": [],
            "for_respondent_state_defendant": [],
        },
        "locations": {
            "gpe": [],
            "org": [],
        },
        "provisions": [],
        "statutes": [],
        "precedents": [],
        "entities": {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
        },
        "texts": {
            "raw_text": "",
            "facts_text": None,
            "arguments_petitioner": None,
            "arguments_respondent": None,
            "reasoning_text": None,
            "decision_text": None,
        },
        "key_facts_bullets": [],
        "issues_bullets": [],
        "holdings_bullets": [],
        "outcome": {
            "label": None,
            "winner": None,
        },
        "ml": {
            "input_text": "",
            "removed_spans": [],
            "leakage_flag": False,
            "leaked_phrases_found": [],
        },
    }


def _overlay_with_defaults(default_obj: Any, incoming_obj: Any) -> Any:
    if isinstance(default_obj, dict):
        incoming = incoming_obj if isinstance(incoming_obj, dict) else {}
        output: dict[str, Any] = {}
        for key, default_val in default_obj.items():
            output[key] = _overlay_with_defaults(default_val, incoming.get(key))
        return output

    if isinstance(default_obj, list):
        if isinstance(incoming_obj, list):
            return incoming_obj
        return list(default_obj)

    if incoming_obj is None:
        return default_obj
    return incoming_obj


def coerce_case_record(
    candidate_record: dict[str, Any] | None,
    case_id: str,
    file_name: str,
) -> dict[str, Any]:
    defaults = empty_case_record(case_id=case_id, file_name=file_name)
    merged = _overlay_with_defaults(defaults, candidate_record or {})
    merged["case_id"] = case_id
    merged["file_name"] = file_name
    return merged


def validate_case_record(record: dict[str, Any]) -> None:
    errors = sorted(SCHEMA_VALIDATOR.iter_errors(record), key=lambda e: list(e.path))
    if not errors:
        return
    formatted = []
    for err in errors:
        path = ".".join(str(p) for p in err.path) or "$"
        formatted.append(f"{path}: {err.message}")
    raise ValueError("Schema validation failed: " + " | ".join(formatted))
