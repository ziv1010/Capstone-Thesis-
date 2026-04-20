#!/usr/bin/env python3
"""
05_group_labels.py
Group sentence-level predictions by rhetorical role into a structured
output per document.
"""

import argparse
import json
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

ROLE_KEYS = {
    "None": "none",
    "Facts": "facts",
    "Issue": "issue",
    "Arguments of Petitioner": "arguments_of_petitioner",
    "Arguments of Respondent": "arguments_of_respondent",
    "Reasoning": "reasoning",
    "Decision": "decision",
}


def group_predictions(predictions: list[dict]) -> dict[str, list[str]]:
    """Group sentence texts by their predicted label."""
    grouped = defaultdict(list)
    for p in predictions:
        key = ROLE_KEYS.get(p["label"], "none")
        grouped[key].append(p["text"])
    return dict(grouped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(input_dir: str, output_dir: str) -> list[str]:
    """Read per-doc prediction JSONs and produce structured outputs."""
    os.makedirs(output_dir, exist_ok=True)

    pred_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith("_predictions.json")
    )
    if not pred_files:
        logger.warning("No prediction files found in %s", input_dir)
        return []

    logger.info("Grouping predictions from %d file(s)", len(pred_files))
    output_paths = []

    for pf in pred_files:
        with open(os.path.join(input_dir, pf), "r", encoding="utf-8") as f:
            data = json.load(f)

        doc_id = data["doc_id"]
        grouped = group_predictions(data["predictions"])

        structured = {
            "doc_id": doc_id,
            "facts": grouped.get("facts", []),
            "issue": grouped.get("issue", []),
            "arguments_of_petitioner": grouped.get("arguments_of_petitioner", []),
            "arguments_of_respondent": grouped.get("arguments_of_respondent", []),
            "reasoning": grouped.get("reasoning", []),
            "decision": grouped.get("decision", []),
            "none": grouped.get("none", []),
        }

        out_path = os.path.join(output_dir, f"{doc_id}_structured.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, indent=2, ensure_ascii=False)

        logger.info("Saved: %s", out_path)
        output_paths.append(out_path)

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Group sentence predictions into structured output."
    )
    parser.add_argument("--input_dir", required=True,
                        help="Folder with *_predictions.json files")
    parser.add_argument("--output_dir", required=True,
                        help="Folder for structured output JSONs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
