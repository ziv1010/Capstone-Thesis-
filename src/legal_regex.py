from __future__ import annotations

import re

from .utils import dedupe_str_list

PROVISION_PATTERNS = [
    # "Section 328", "Sec. 482", "u/s 482 CrPC", "under section 26(2)(iv)" etc.
    # Requires at least one digit in the provision number to avoid matching bare words.
    re.compile(
        r"\b(?:Section|Sec\.?|under\s+section)\s+\d[\dA-Za-z()./-]*"
        r"(?:\s*(?:and|,)\s*\d[\dA-Za-z()./-]*)*"
        r"(?:\s+(?:of\s+the\s+)?"
        r"(?:IPC|I\.P\.C\.|CrPC|Cr\.P\.C\.|CPC|C\.P\.C\.|Constitution(?:\s+of\s+India)?|"
        r"Indian\s+Evidence\s+Act|NDPS\s+Act|POCSO\s+Act|"
        r"Prevention\s+of\s+Corruption\s+Act|[\w\s]+Act(?:,\s*\d{4})?))?",
        re.IGNORECASE,
    ),
    # Short-form: "u/s 328", "U/S 482"
    re.compile(
        r"\bu/?s\s+\d[\dA-Za-z()./-]*"
        r"(?:\s+(?:IPC|I\.P\.C\.|CrPC|Cr\.P\.C\.|CPC|C\.P\.C\.))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bArticle\s+\d+[A-Za-z-]*\b(?:\s+of\s+the\s+Constitution(?:\s+of\s+India)?)?",
        re.IGNORECASE,
    ),
]

STATUTE_PATTERNS = [
    re.compile(r"\b[A-Z][A-Za-z&(),.\- ]+ Act,?\s*(?:18|19|20)\d{2}\b"),
    re.compile(
        r"\b(?:Indian Penal Code|Code of Criminal Procedure|Code of Civil Procedure|"
        r"Constitution of India|Indian Evidence Act|Arbitration and Conciliation Act)\b",
        re.IGNORECASE,
    ),
]

PRECEDENT_PATTERNS = [
    re.compile(
        r"\b[A-Z][A-Za-z0-9&.,'()\- ]+\s+v(?:s\.?|\.)\s+[A-Z][A-Za-z0-9&.,'()\- ]+\b"
    ),
    re.compile(r"\b\d{4}\s*\(\d+\)\s*SCC\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bAIR\s*\d{4}\s*[A-Z]{2,}\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bSLP\s*\(?.*?\)?\s*No\.?\s*\d+/?\d*\b", re.IGNORECASE),
    re.compile(r"\bCrLJ\s*\d+\b", re.IGNORECASE),
    re.compile(r"\bABA\s*\(Stamp\)\s*No\.?\s*\d+/?\d*\b", re.IGNORECASE),
]


def _collect(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    matches: list[str] = []
    for pattern in patterns:
        for m in pattern.finditer(text):
            candidate = re.sub(r"\s+", " ", m.group(0)).strip(" ,.;:\n\t")
            if candidate:
                matches.append(candidate)
    return dedupe_str_list(matches)


def extract_legal_references(text: str) -> dict[str, list[str]]:
    base_text = text or ""
    return {
        "provisions": _collect(PROVISION_PATTERNS, base_text),
        "statutes": _collect(STATUTE_PATTERNS, base_text),
        "precedents": _collect(PRECEDENT_PATTERNS, base_text),
    }
