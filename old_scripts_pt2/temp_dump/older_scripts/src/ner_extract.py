from __future__ import annotations

import re
from typing import Any

import spacy
from spacy.language import Language

from .utils import dedupe_str_list

SUPPORTED_LABELS = ("PERSON", "ORG", "GPE", "DATE")


def load_spacy_model(model_name: str) -> Language:
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. Run: python -m spacy download {model_name}"
        ) from exc


def _normalize_entity(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" ,;:\n\t")
    return cleaned


def extract_entities(
    text: str,
    nlp: Language,
    max_chars: int = 0,
) -> dict[str, list[str]]:
    target_text = text or ""
    if max_chars and max_chars > 0:
        target_text = target_text[:max_chars]

    doc = nlp(target_text)
    output: dict[str, list[str]] = {label: [] for label in SUPPORTED_LABELS}
    for ent in doc.ents:
        if ent.label_ not in output:
            continue
        normalized = _normalize_entity(ent.text)
        if normalized:
            output[ent.label_].append(normalized)

    for label in SUPPORTED_LABELS:
        output[label] = dedupe_str_list(output[label])

    return output


def build_locations_from_entities(entities: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "gpe": dedupe_str_list(list(entities.get("GPE", []))),
        "org": dedupe_str_list(list(entities.get("ORG", []))),
    }
