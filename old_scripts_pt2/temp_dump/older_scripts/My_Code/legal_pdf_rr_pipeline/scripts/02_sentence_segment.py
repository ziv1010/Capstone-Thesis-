#!/usr/bin/env python3
"""
02_sentence_segment.py
Split extracted plain-text court documents into sentences using spaCy.
Saves one JSON per document with doc_id, sentence_id, and text.
"""

import argparse
import json
import logging
import os

import spacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def segment_sentences(text: str, nlp) -> list[str]:
    """Return a list of sentence strings using spaCy sentencizer."""
    doc = nlp(text)
    sentences = [" ".join(sent.text.split()) for sent in doc.sents if sent.text.strip()]
    return sentences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(input_dir: str, output_dir: str) -> list[str]:
    """Process all .txt files in input_dir and save JSON to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 5_000_000

    txt_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".txt")
    )
    if not txt_files:
        logger.warning("No .txt files found in %s", input_dir)
        return []

    logger.info("Found %d text file(s) in %s", len(txt_files), input_dir)
    output_paths = []

    for txt_file in txt_files:
        txt_path = os.path.join(input_dir, txt_file)
        doc_id = os.path.splitext(txt_file)[0]

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        sentences = segment_sentences(text, nlp)
        logger.info("Doc %s: %d sentences", doc_id, len(sentences))

        doc_json = {
            "doc_id": doc_id,
            "sentences": [
                {"sentence_id": i, "sent_id": i, "text": s} for i, s in enumerate(sentences)
            ],
        }

        json_path = os.path.join(output_dir, f"{doc_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, indent=2, ensure_ascii=False)

        logger.info("Saved: %s", json_path)
        output_paths.append(json_path)

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Segment text into sentences.")
    parser.add_argument("--input_dir", required=True, help="Folder with .txt files")
    parser.add_argument("--output_dir", required=True, help="Output folder for .json files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
