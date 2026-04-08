#!/usr/bin/env python3
"""Extract text from court-judgment PDFs into UTF-8 text files."""

from __future__ import annotations

import argparse
import fnmatch
import io
import logging
import os
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF

LOGGER = logging.getLogger("pdf_text_extractor")
MIN_CHARS_FOR_VALID_TEXT = 200


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def clean_text(text: str) -> str:
    """Normalise whitespace while preserving readable line breaks."""
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_with_pymupdf(pdf_path: Path) -> str:
    """Prefer the PDF text layer when it is present."""
    document = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in document]
    finally:
        document.close()
    return "\n".join(pages)


def extract_with_ocr(pdf_path: Path) -> str:
    """Fallback OCR path for scanned pages."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        LOGGER.error("OCR requested for %s but pytesseract/Pillow are unavailable.", pdf_path.name)
        return ""

    document = fitz.open(pdf_path)
    try:
        pages = []
        for page_number, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(dpi=300)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            page_text = pytesseract.image_to_string(image, lang="eng")
            LOGGER.debug("OCR page %s from %s yielded %s chars", page_number, pdf_path.name, len(page_text))
            pages.append(page_text)
    finally:
        document.close()

    return "\n".join(pages)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract readable text from a single PDF."""
    LOGGER.info("Extracting %s", pdf_path.name)
    text = extract_with_pymupdf(pdf_path)
    if len(text.strip()) < MIN_CHARS_FOR_VALID_TEXT:
        LOGGER.warning(
            "PyMuPDF extracted only %s chars from %s; falling back to OCR.",
            len(text.strip()),
            pdf_path.name,
        )
        text = extract_with_ocr(pdf_path)

    cleaned = clean_text(text)
    LOGGER.info("Extracted %s chars from %s", len(cleaned), pdf_path.name)
    return cleaned


def discover_pdf_files(input_dir: Path, glob_pattern: str) -> List[Path]:
    """Discover PDF files with case-insensitive matching."""
    matches = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".pdf":
            continue
        if fnmatch.fnmatch(path.name.lower(), glob_pattern.lower()):
            matches.append(path)
    return sorted(matches)


def run(input_dir: Path, output_dir: Path, glob_pattern: str, overwrite: bool) -> List[Path]:
    """Process PDFs and write one .txt file per source PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = discover_pdf_files(input_dir, glob_pattern)

    if not pdf_files:
        LOGGER.warning("No PDF files matched %r under %s", glob_pattern, input_dir)
        return []

    LOGGER.info("Found %s PDF file(s) under %s", len(pdf_files), input_dir)
    written_files: List[Path] = []

    for pdf_path in pdf_files:
        file_id = pdf_path.stem
        target_path = output_dir / f"{file_id}.txt"

        if target_path.exists() and not overwrite:
            LOGGER.info("Skipping %s because %s already exists", pdf_path.name, target_path.name)
            written_files.append(target_path)
            continue

        text = extract_pdf_text(pdf_path)
        target_path.write_text(text, encoding="utf-8")
        LOGGER.info("Saved %s", target_path)
        written_files.append(target_path)

    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from a folder of PDF judgments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", "--input-dir", required=True, help="Directory containing PDF files.")
    parser.add_argument("--output_dir", "--output-dir", required=True, help="Directory for extracted .txt files.")
    parser.add_argument("--glob_pattern", "--glob-pattern", default="*.pdf", help="Filename pattern for PDFs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files.")
    parser.add_argument("--log_level", "--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    run(
        input_dir=Path(args.input_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        glob_pattern=args.glob_pattern,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
