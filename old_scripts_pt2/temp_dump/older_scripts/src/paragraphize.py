from __future__ import annotations

import re
from typing import Any

NUMBERED_OR_BULLET_RE = re.compile(
    r"^\s*(?:\(?\d+[.)]|[ivxlcdmIVXLCDM]+[.)]|[A-Za-z][.)]|[-*•])\s+"
)
INDENTED_RE = re.compile(r"^\s{2,}\S")


def normalize_text(raw_text: str) -> str:
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t ]+$", "", line) for line in text.split("\n")]
    return "\n".join(lines).strip()


def _split_block(block: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    for line in block.split("\n"):
        if not line.strip():
            if current:
                paragraphs.append(" ".join(s.strip() for s in current).strip())
                current = []
            continue

        is_new_marker = bool(NUMBERED_OR_BULLET_RE.match(line) or INDENTED_RE.match(line))
        if is_new_marker and current:
            paragraphs.append(" ".join(s.strip() for s in current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        paragraphs.append(" ".join(s.strip() for s in current).strip())

    return [p for p in paragraphs if p]


def paragraphize(raw_text: str) -> list[dict[str, Any]]:
    normalized = normalize_text(raw_text)
    if not normalized:
        return []

    blocks = re.split(r"\n\s*\n+", normalized)
    paragraphs: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        paragraphs.extend(_split_block(block))

    return [{"index": idx, "text": paragraph} for idx, paragraph in enumerate(paragraphs)]
