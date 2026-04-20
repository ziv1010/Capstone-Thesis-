#!/usr/bin/env python3
"""
03_prepare_toinlegalbert_input.py
Read per-document sentence JSONs and build a single inference-ready JSON:
[
    {"doc_id": "...", "segments": ["...", "..."]},
    ...
]
"""

import argparse
import json
import logging
import os

logger = logging.getLogger(__name__)


def run(input_dir: str, output_path: str) -> str:
    """Read sentence JSONs and produce one combined inference JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    json_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".json")
    )
    if not json_files:
        logger.warning("No JSON files found in %s", input_dir)
        return output_path

    logger.info("Preparing inference input from %d file(s)", len(json_files))

    documents = []
    for jf in json_files:
        with open(os.path.join(input_dir, jf), "r", encoding="utf-8") as f:
            doc = json.load(f)

        segments = [s["text"] for s in doc["sentences"]]
        documents.append({
            "doc_id": doc["doc_id"],
            "segments": segments,
        })
        logger.info("Doc %s: %d segments", doc["doc_id"], len(segments))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    logger.info("Saved inference input: %s (%d docs)", output_path, len(documents))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare ToInLegalBERT input.")
    parser.add_argument("--input_dir", required=True, help="Folder with sentence JSONs")
    parser.add_argument("--output_path", required=True, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()
