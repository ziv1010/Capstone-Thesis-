from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LocalHTTPLLMClient:
    endpoint_url: str
    model_name: str
    timeout_sec: int = 120
    temperature: float = 0.0

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": float(self.temperature),
        }
        response = requests.post(self.endpoint_url, json=payload, timeout=self.timeout_sec)
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
                    msg = first.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        return msg["content"]

        raise RuntimeError("Unsupported local LLM response format")


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidate = cleaned[start : end + 1]
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError("Could not parse JSON object from LLM response")


def predict_with_llm_json(
    client: LocalHTTPLLMClient,
    prompt: str,
    retry_on_parse_error: bool = True,
) -> tuple[dict[str, Any], str]:
    raw = client.generate(prompt)
    try:
        parsed = _extract_json_object(raw)
        return parsed, raw
    except Exception:
        if not retry_on_parse_error:
            raise

    fix_prompt = (
        "Return STRICT JSON ONLY. No markdown, no explanation.\n"
        "Use keys exactly: pred_label, pred_winner, confidence, rationale, cited_case_ids.\n\n"
        f"Original prompt:\n{prompt}\n\n"
        f"Previous malformed output:\n{raw}"
    )
    raw2 = client.generate(fix_prompt)
    parsed2 = _extract_json_object(raw2)
    return parsed2, raw2
