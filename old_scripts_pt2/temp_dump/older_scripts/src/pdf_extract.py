from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

PAGE_SEPARATOR_TEMPLATE = "\n\n<<<PAGE_{page_num}>>>\n\n"

# IndianKanoon PDFs always print this footer on each page
_IK_URL_RE = re.compile(r"Indian Kanoon - (https?://indiankanoon\.org/doc/\d+/?)", re.IGNORECASE)
_IK_AUTHOR_RE = re.compile(r"^Author:\s*(.+)$", re.MULTILINE)
_IK_BENCH_RE = re.compile(r"^Bench:\s*(.+)$", re.MULTILINE)


@dataclass
class PDFExtractionResult:
    raw_text: str
    page_texts: list[str]
    source_url: str | None = field(default=None)
    ik_author: str | None = field(default=None)
    ik_bench: str | None = field(default=None)


def _extract_with_pymupdf(pdf_path: Path) -> PDFExtractionResult:
    page_texts: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text = page.get_text("text") or ""
            page_texts.append(text.strip())

    chunks: list[str] = []
    for idx, page_text in enumerate(page_texts, start=1):
        chunks.append(PAGE_SEPARATOR_TEMPLATE.format(page_num=idx))
        chunks.append(page_text)

    raw_text = "\n".join(chunks).strip()
    return PDFExtractionResult(raw_text=raw_text, page_texts=page_texts)


def _extract_with_pdfplumber(pdf_path: Path) -> PDFExtractionResult:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is not installed. Install it or use pdf_extraction_mode=pymupdf"
        ) from exc

    page_texts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_texts.append(text.strip())

    chunks: list[str] = []
    for idx, page_text in enumerate(page_texts, start=1):
        chunks.append(PAGE_SEPARATOR_TEMPLATE.format(page_num=idx))
        chunks.append(page_text)

    raw_text = "\n".join(chunks).strip()
    return PDFExtractionResult(raw_text=raw_text, page_texts=page_texts)


def _parse_ik_metadata(result: PDFExtractionResult) -> PDFExtractionResult:
    """Extract IndianKanoon-specific metadata from raw text (source URL, author, bench)."""
    text = result.raw_text
    url_m = _IK_URL_RE.search(text)
    if url_m:
        result.source_url = url_m.group(1).rstrip("/")
    author_m = _IK_AUTHOR_RE.search(text)
    if author_m:
        result.ik_author = author_m.group(1).strip()
    bench_m = _IK_BENCH_RE.search(text)
    if bench_m:
        result.ik_bench = bench_m.group(1).strip()
    return result


def extract_pdf_text(pdf_path: str | Path, mode: str = "pymupdf") -> PDFExtractionResult:
    path = Path(pdf_path)
    normalized_mode = (mode or "pymupdf").strip().lower()
    if normalized_mode == "pdfplumber":
        result = _extract_with_pdfplumber(path)
    else:
        result = _extract_with_pymupdf(path)
    return _parse_ik_metadata(result)
