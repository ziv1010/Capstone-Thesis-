#!/usr/bin/env python3
"""
Convert all PDFs in the financial_fraud folder to .txt using PyMuPDF only.
Skips files that already have a corresponding .txt.
"""

import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

PDF_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/DATA_SET_BUILDER_AND_EXPLORER/Bucket_testing/financial_fraud"
)
TXT_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-"
    "/DATA_SET_BUILDER_AND_EXPLORER/Bucket_testing/ff_pipeline/txts"
)


def pdf_to_txt(pdf_path: Path) -> None:
    txt_path = TXT_DIR / (pdf_path.stem + ".txt")
    if txt_path.exists():
        return  # already converted

    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        text = "\n".join(pages).strip()
        txt_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"  [ERROR] {pdf_path.name}: {e}")


def main():
    TXT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs):,} PDFs in {PDF_DIR}")

    already_done = sum(1 for p in pdfs if (TXT_DIR / (p.stem + ".txt")).exists())
    to_convert   = len(pdfs) - already_done
    print(f"  Already converted : {already_done:,}")
    print(f"  To convert        : {to_convert:,}")

    for pdf in tqdm(pdfs, unit="pdf", desc="Converting"):
        pdf_to_txt(pdf)

    txt_count = len(list(TXT_DIR.glob("*.txt")))
    print(f"\nDone. {txt_count:,} .txt files now present in {TXT_DIR}")


if __name__ == "__main__":
    main()
