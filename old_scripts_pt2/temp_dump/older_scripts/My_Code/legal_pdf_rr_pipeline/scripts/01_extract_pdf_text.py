#!/usr/bin/env python3
"""
01_extract_pdf_text.py
Extract text from Indian court proceeding PDFs using PyMuPDF.
Falls back to OCR (pytesseract) if text extraction yields too little text.
"""

import argparse
import logging
import os
import re
import sys

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MIN_CHARS_FOR_VALID_TEXT = 200  # per-page threshold before OCR fallback


def _clean_text(text: str) -> str:
    """Collapse whitespace, remove form-feed chars, strip stray unicode."""
    text = text.replace("\x0c", "\n")           # form-feed → newline
    text = re.sub(r"[ \t]+", " ", text)         # collapse horizontal ws
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse blank lines
    text = text.strip()
    return text


def _extract_with_pymupdf(pdf_path: str) -> str:
    """Text-layer extraction via PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def _extract_with_ocr(pdf_path: str) -> str:
    """OCR fallback: render each page as image, then pytesseract."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        logger.error("pytesseract / Pillow not installed – cannot OCR.")
        return ""

    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng")
        pages.append(text)
        logger.debug("OCR page %d: %d chars", page_num, len(text))
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a single PDF, with OCR fallback."""
    logger.info("Extracting text from: %s", pdf_path)

    text = _extract_with_pymupdf(pdf_path)
    if len(text.strip()) < MIN_CHARS_FOR_VALID_TEXT:
        logger.warning(
            "PyMuPDF extracted only %d chars (< %d). Falling back to OCR.",
            len(text.strip()), MIN_CHARS_FOR_VALID_TEXT,
        )
        text = _extract_with_ocr(pdf_path)

    text = _clean_text(text)
    logger.info("Extracted %d chars from %s", len(text), os.path.basename(pdf_path))
    return text


def run(input_dir: str, output_dir: str) -> list[str]:
    """Process all PDFs in input_dir and save .txt files to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    pdf_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".pdf",))
    )
    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)
        return []

    logger.info("Found %d PDF(s) in %s", len(pdf_files), input_dir)
    output_paths = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        doc_id = os.path.splitext(pdf_file)[0]
        txt_path = os.path.join(output_dir, f"{doc_id}.txt")

        text = extract_pdf_text(pdf_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        logger.info("Saved: %s", txt_path)
        output_paths.append(txt_path)

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract text from PDFs.")
    parser.add_argument("--input_dir", required=True, help="Folder with PDFs")
    parser.add_argument("--output_dir", required=True, help="Output folder for .txt files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
