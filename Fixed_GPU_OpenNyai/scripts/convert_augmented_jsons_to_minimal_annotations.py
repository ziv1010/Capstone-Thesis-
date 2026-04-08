#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.output_formatter import entity_label, iter_sentence_entities, rhetorical_role, unwrap_raw_result


DEFAULT_INPUT_DIR = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/outputs/old_outputs/food_law_case _output/augmented_jsons"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/Fixed_GPU_OpenNyai/outputs/converted_annotations/food_law_case_old_augmented_jsons_minimal"
)

LABEL_TO_SCORE = {
    "appellant_won": 1,
    "postponed_or_procedural": 0,
    "appellant_lost": -1,
}
VALID_RR_LABELS = {
    "PREAMBLE",
    "FAC",
    "ISSUE",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "ANALYSIS",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "STA",
    "RLC",
    "RPC",
    "RATIO",
    "NONE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert old augmented JSONs into minimal annotation-format JSONs that keep only "
            "sentence-level RR, NER entities, and a numeric decision_label."
        )
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N files. 0 means all files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output JSON files in the target annotations folder.",
    )
    return parser.parse_args()


def _decision_score(payload: dict[str, Any]) -> int:
    raw_score = payload.get("case_outcome_score")
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        score = None

    if score in {-1, 0, 1}:
        return score

    label = str(payload.get("case_outcome_label", "")).strip().lower()
    if label in LABEL_TO_SCORE:
        return LABEL_TO_SCORE[label]

    raise ValueError(
        f"Missing valid decision score/label. score={raw_score!r} label={payload.get('case_outcome_label')!r}"
    )


def _normalise_role(annotation: dict[str, Any]) -> str:
    role = str(rhetorical_role(annotation) or "NONE").upper()
    if role not in VALID_RR_LABELS:
        return "NONE"
    return role


def _normalise_entity(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(entity.get("text", "")).strip(),
        "label": entity_label(entity),
        "start": entity.get("start"),
        "end": entity.get("end"),
    }


def convert_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw_doc = unwrap_raw_result(payload)
    raw_annotations = raw_doc.get("annotations", [])
    if not isinstance(raw_annotations, list):
        raise ValueError("raw_result.annotations is not a list.")

    preamble_end_char_offset = None
    raw_data = raw_doc.get("data", {})
    if isinstance(raw_data, dict):
        preamble_end_char_offset = raw_data.get("preamble_end_char_offset")

    sentences = []
    ner_counts: Counter[str] = Counter()
    rr_counts: Counter[str] = Counter()

    for sentence_index, annotation in enumerate(raw_annotations, start=1):
        role = _normalise_role(annotation)
        rr_counts[role] += 1

        entities = []
        for entity in iter_sentence_entities(annotation):
            normalized_entity = _normalise_entity(entity)
            entities.append(normalized_entity)
            label = normalized_entity.get("label")
            if label:
                ner_counts[str(label)] += 1

        sentences.append(
            {
                "sentence_id": sentence_index,
                "text": str(annotation.get("text", "")).strip(),
                "rhetorical_role": role,
                "start": annotation.get("start"),
                "end": annotation.get("end"),
                "entities": entities,
            }
        )

    return {
        "file_id": str(payload.get("file_id", "")),
        "chunk": str(payload.get("chunk", "")),
        "rr_available": True,
        "preamble_end_char_offset": preamble_end_char_offset,
        "decision_label": _decision_score(payload),
        "sentences": sentences,
        "ner_by_label": dict(ner_counts.most_common()),
        "rr_by_role": dict(rr_counts.most_common()),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob("*.json"))
    if args.limit > 0:
        files = files[: args.limit]

    converted = 0
    skipped = 0
    errors: list[dict[str, str]] = []
    decision_counts: Counter[int] = Counter()
    ner_counts: Counter[str] = Counter()
    rr_counts: Counter[str] = Counter()

    for path in files:
        out_path = annotations_dir / path.name
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            converted_payload = convert_payload(payload)
        except Exception as exc:
            errors.append({"file_name": path.name, "error": str(exc)})
            continue

        out_path.write_text(json.dumps(converted_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        converted += 1
        decision_counts.update([int(converted_payload["decision_label"])])
        ner_counts.update(converted_payload["ner_by_label"])
        rr_counts.update(converted_payload["rr_by_role"])

    summary = {
        "input_dir": str(input_dir),
        "output_root": str(output_root),
        "annotations_dir": str(annotations_dir),
        "files_seen": len(files),
        "files_converted": converted,
        "files_skipped_existing": skipped,
        "files_with_errors": len(errors),
        "decision_counts": {str(key): value for key, value in sorted(decision_counts.items())},
        "ner_by_label": dict(ner_counts.most_common()),
        "rr_by_role": dict(rr_counts.most_common()),
        "errors": errors,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Converted {converted} file(s) into {annotations_dir}")
    print(f"Skipped existing: {skipped}")
    print(f"Errors: {len(errors)}")
    print(f"Summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
