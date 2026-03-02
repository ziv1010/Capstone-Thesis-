from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .schema import CASE_JSON_SCHEMA


@dataclass
class LocalHTTPLLMClient:
    endpoint_url: str
    model_name: str
    timeout_seconds: int = 120
    temperature: float = 0.0

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            if isinstance(data.get("response"), str):
                return data["response"]
            if isinstance(data.get("text"), str):
                return data["text"]
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    if isinstance(first.get("text"), str):
                        return first["text"]
                    message = first.get("message")
                    if isinstance(message, dict) and isinstance(message.get("content"), str):
                        return message["content"]

        raise RuntimeError("Unsupported LLM response format from endpoint")


def _format_paragraphs_for_prompt(paragraphs: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for para in paragraphs:
        idx = para.get("index")
        text = str(para.get("text", "")).strip()
        if not text:
            continue
        lines.append(f"[{idx}] {text}")
    return "\n".join(lines)


def build_llm_prompt(
    raw_text: str,
    paragraphs: list[dict[str, Any]],
    ner: dict[str, list[str]],
    regex: dict[str, list[str]],
    max_chars_to_send: int = 0,
) -> str:
    truncated_text = raw_text[:max_chars_to_send] if max_chars_to_send > 0 else raw_text
    para_text = _format_paragraphs_for_prompt(paragraphs)

    instructions = {
        "task": "Extract a strict JSON case record for an Indian court judgment/order",
        "hard_rules": [
            "Return JSON ONLY. No markdown. No explanation.",
            "All schema keys must be present. Use null or [] when unknown.",
            "Use verbatim text spans from supplied paragraphs where possible.",
            "Put dispositive/outcome language ONLY in texts.decision_text and outcome.* — never in any other field.",
            "texts.facts_text must contain ONLY background facts and procedural history. No rulings.",
            "texts.arguments_petitioner must contain ONLY petitioner/applicant/prosecution side submissions.",
            "texts.arguments_respondent must contain ONLY respondent/state/defense side submissions.",
            "texts.reasoning_text may contain court analysis/reasoning paragraphs ONLY IF they do not reveal the final outcome.",
            "ml.input_text = concatenation of facts_text + arguments_petitioner + arguments_respondent ONLY. Must be completely free of outcome/disposition language.",
            "If you removed a paragraph from ml.input_text for safety, record it in ml.removed_spans with paragraph_index and reason='outcome_phrase'.",
            "Fill both arguments fields best-effort: petitioner/prosecution side vs respondent/defense side.",
            "Map sides correctly: petitioner_applicant = person who filed (bail applicant, writ petitioner, appellant); respondent_state_defendant = state/opposite party.",
            "Date must be YYYY-MM-DD or null.",
        ],
        "leakage_safety": {
            "MUST go into decision_text (NEVER into input_text)": [
                "Any sentence containing: 'is/are allowed', 'is/are dismissed', 'is/are rejected', 'is/are granted', 'is/are refused', 'is/are quashed'",
                "Any sentence containing: 'anticipatory bail is granted/rejected/refused'",
                "Any sentence containing: 'be released on bail', 'released on bail'",
                "Any sentence containing: 'petition succeeds', 'petition fails'",
                "Any sentence containing: 'prayer is granted/rejected/dismissed'",
                "Any sentence containing: 'rule is made absolute', 'rule is discharged'",
                "Any sentence containing: 'proceedings are quashed', 'proceedings stand quashed'",
                "Any sentence containing: 'not inclined to grant', 'inclined to grant'",
                "Any sentence containing: 'I am of the view that bail should/should not be granted'",
                "Any sentence containing: 'The following order is passed', 'ORDER:', 'ORAL ORDER:'",
                "Final numbered court directions like '(i) In the event of arrest... be released...'",
                "Any sentence revealing who won or lost.",
            ],
            "SAFE to include in input_text": [
                "Facts: what happened, background, FIR details, charges, procedural history.",
                "Arguments made by petitioner/applicant counsel.",
                "Arguments made by respondent/state/APP counsel.",
                "Court's neutral recitation of facts ('It appears from the record that...').",
                "Cited precedents and how they were argued (not applied as rulings).",
                "Section/provision numbers referenced in arguments.",
                "Phrases like 'in order to submit', 'an order dated X was passed by the trial court' (historical orders, not THIS court's outcome).",
            ],
        },
        "schema": CASE_JSON_SCHEMA,
        "ner_hints": ner,
        "regex_hints": regex,
    }

    prompt = (
        "You are a legal information extraction engine specializing in Indian court documents.\n"
        "Your PRIMARY responsibility is leakage safety: ml.input_text must NEVER reveal the outcome.\n"
        "Use the provided text to populate the JSON schema exactly.\n\n"
        f"INSTRUCTIONS:\n{json.dumps(instructions, ensure_ascii=False, indent=2)}\n\n"
        "RAW_TEXT (possibly truncated):\n"
        f"{truncated_text}\n\n"
        "PARAGRAPHS WITH INDICES:\n"
        f"{para_text}\n\n"
        "CRITICAL: Before returning, re-read ml.input_text and verify it contains zero outcome phrases.\n"
        "Return strict JSON only."
    )
    return prompt


def build_json_fix_prompt(
    invalid_response_text: str,
    validation_error: str,
) -> str:
    return (
        "Fix the following invalid JSON to match the schema exactly.\n"
        "Return JSON only and preserve as much extracted content as possible.\n"
        "Do not add commentary.\n\n"
        f"SCHEMA:\n{json.dumps(CASE_JSON_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"VALIDATION_ERROR:\n{validation_error}\n\n"
        f"INVALID_RESPONSE:\n{invalid_response_text}"
    )


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_json_from_llm_text(response_text: str) -> dict[str, Any]:
    cleaned = _strip_fences(response_text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse a JSON object from LLM response")


def extract_case_json_with_retry(
    client: LocalHTTPLLMClient,
    prompt: str,
    validate_fn: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    first_response = client.generate(prompt)
    try:
        first_record = parse_json_from_llm_text(first_response)
        validate_fn(first_record)
        return first_record
    except Exception as first_error:
        fix_prompt = build_json_fix_prompt(first_response, str(first_error))

    second_response = client.generate(fix_prompt)
    second_record = parse_json_from_llm_text(second_response)
    validate_fn(second_record)
    return second_record
