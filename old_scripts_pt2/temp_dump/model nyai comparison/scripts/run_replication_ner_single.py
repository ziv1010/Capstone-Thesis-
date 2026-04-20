#!/usr/bin/env python3
"""Run replication-style Legal NER on one UTF-8 text file."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import spacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_root", required=True, help="Path to REPLICATION_OpenNyai/external/legal_NER.")
    parser.add_argument("--text_path", required=True, help="Path to the input .txt file.")
    parser.add_argument("--output_path", required=True, help="Path to write the JSON output.")
    parser.add_argument("--model_name", default="en_legal_ner_trf")
    parser.add_argument("--preamble_model_name", default="en_core_web_sm")
    parser.add_argument("--run_type", choices=["sent", "doc"], default="sent")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--no_postprocess", action="store_true")
    return parser.parse_args()


def prepare_repo_imports(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def normalize_label(raw_label: str) -> str:
    return str(raw_label).strip().upper()


def normalize_judgment_text_with_mapping(judgment_text: str, base_offset: int) -> tuple[str, list[int]]:
    output_chars: list[str] = []
    mapping: list[int] = []
    index = 0
    length = len(judgment_text)

    while index < length:
        char = judgment_text[index]
        if char.isalnum() or char == "_":
            start = index
            index += 1
            while index < length and (judgment_text[index].isalnum() or judgment_text[index] in {"_", " ", "-"}):
                index += 1
            run_end = index
            newline_start = index
            while index < length and judgment_text[index] == "\n":
                index += 1
            if newline_start < index:
                for original_index in range(start, run_end):
                    output_chars.append(judgment_text[original_index])
                    mapping.append(base_offset + original_index)
                output_chars.append(" ")
                mapping.append(base_offset + newline_start)
            else:
                for original_index in range(start, run_end):
                    output_chars.append(judgment_text[original_index])
                    mapping.append(base_offset + original_index)
        else:
            output_chars.append(char)
            mapping.append(base_offset + index)
            index += 1

    return "".join(output_chars), mapping


def build_mapping_for_combined_text(parts: list[tuple[str, list[int]]], combined_text: str) -> list[int]:
    mapping: list[int] = []
    cursor = 0
    previous_reference = parts[0][1][0] if parts and parts[0][1] else 0

    for part_text, part_mapping in parts:
        if not part_text:
            continue
        while cursor < len(combined_text) and not combined_text.startswith(part_text, cursor):
            mapping.append(previous_reference)
            cursor += 1
        if not combined_text.startswith(part_text, cursor):
            raise RuntimeError("Unable to align combined spaCy doc text with source segments.")
        mapping.extend(part_mapping)
        previous_reference = part_mapping[-1] if part_mapping else previous_reference
        cursor += len(part_text)

    while cursor < len(combined_text):
        mapping.append(previous_reference)
        cursor += 1

    if len(mapping) != len(combined_text):
        raise RuntimeError("Combined text mapping length mismatch.")
    return mapping


def map_entity_back_to_original(start_char: int, end_char: int, char_mapping: list[int]) -> tuple[int, int]:
    start_original = char_mapping[start_char]
    end_original = char_mapping[end_char - 1] + 1
    return start_original, end_original


def ensure_sentence_boundaries(doc):
    if doc is None or len(doc) == 0:
        return doc
    if doc.has_annotation("SENT_START"):
        return doc
    doc[0].is_sent_start = True
    for token in doc[1:]:
        token.is_sent_start = False
    return doc


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    text_path = Path(args.text_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepare_repo_imports(repo_root)
    from data_preparation import get_sentence_docs, seperate_and_clean_preamble
    from postprocessing_utils import postprocessing

    if args.use_gpu:
        spacy.prefer_gpu(args.gpu_id)

    legal_nlp = spacy.load(args.model_name)
    preamble_nlp = spacy.load(args.preamble_model_name)
    text = text_path.read_text(encoding="utf-8")

    preamble_text, preamble_end = seperate_and_clean_preamble(text, preamble_nlp)
    judgment_text_original = text[preamble_end:]
    judgment_text, judgment_mapping = normalize_judgment_text_with_mapping(judgment_text_original, preamble_end)

    preamble_doc = legal_nlp(preamble_text) if preamble_text.strip() else legal_nlp.make_doc("")
    preamble_doc = ensure_sentence_boundaries(preamble_doc)
    preamble_mapping = list(range(len(preamble_doc.text)))

    if args.run_type == "doc":
        judgment_doc = legal_nlp(judgment_text) if judgment_text.strip() else legal_nlp.make_doc("")
        judgment_doc = ensure_sentence_boundaries(judgment_doc)
        judgment_combined_mapping = judgment_mapping[: len(judgment_doc.text)]
    else:
        sentence_boundary_doc = preamble_nlp(judgment_text)
        sentence_spans = [sent for sent in sentence_boundary_doc.sents if sent.text.strip()]
        sentence_texts = [sent.text for sent in sentence_spans]
        sentence_mappings = [judgment_mapping[sent.start_char : sent.end_char] for sent in sentence_spans]
        sentence_docs = list(legal_nlp.pipe(sentence_texts)) if sentence_texts else []
        if sentence_docs:
            sentence_docs = [ensure_sentence_boundaries(doc) for doc in sentence_docs]
            judgment_doc = spacy.tokens.Doc.from_docs(sentence_docs)
            judgment_combined_mapping = build_mapping_for_combined_text(
                [(doc.text, mapping) for doc, mapping in zip(sentence_docs, sentence_mappings)],
                judgment_doc.text,
            )
        else:
            judgment_doc = legal_nlp.make_doc("")
            judgment_combined_mapping = []
        judgment_doc = ensure_sentence_boundaries(judgment_doc)

    docs_to_merge = [doc for doc in [preamble_doc, judgment_doc] if doc.text]
    parts_to_merge: list[tuple[str, list[int]]] = []
    if preamble_doc.text:
        parts_to_merge.append((preamble_doc.text, preamble_mapping))
    if judgment_doc.text:
        parts_to_merge.append((judgment_doc.text, judgment_combined_mapping))

    if not docs_to_merge:
        combined_doc = legal_nlp.make_doc("")
        combined_mapping: list[int] = []
    elif len(docs_to_merge) == 1:
        combined_doc = docs_to_merge[0]
        combined_mapping = parts_to_merge[0][1]
    else:
        combined_doc = spacy.tokens.Doc.from_docs(docs_to_merge)
        combined_mapping = build_mapping_for_combined_text(parts_to_merge, combined_doc.text)
    combined_doc = ensure_sentence_boundaries(combined_doc)

    if not args.no_postprocess and combined_doc.text:
        try:
            combined_doc = postprocessing(combined_doc)
            combined_doc = ensure_sentence_boundaries(combined_doc)
        except Exception:
            pass

    entities: list[dict[str, object]] = []
    seen = set()
    for ent in combined_doc.ents:
        start, end = map_entity_back_to_original(ent.start_char, ent.end_char, combined_mapping)
        key = (start, end, normalize_label(ent.label_))
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": normalize_label(ent.label_),
            }
        )
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"], item["text"]))

    sentence_annotations = []
    for sent in combined_doc.sents:
        start = combined_mapping[sent.start_char] if combined_mapping else 0
        end = combined_mapping[sent.end_char - 1] + 1 if combined_mapping and sent.end_char > sent.start_char else start
        sentence_annotations.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )

    payload = {
        "file_id": text_path.stem,
        "source_path": str(text_path),
        "runtime": {
            "repo_root": str(repo_root),
            "model_name": args.model_name,
            "preamble_model_name": args.preamble_model_name,
            "run_type": args.run_type,
            "do_postprocess": not args.no_postprocess,
            "use_gpu": args.use_gpu,
            "gpu_id": args.gpu_id,
        },
        "entity_count": len(entities),
        "entities": entities,
        "sentences": sentence_annotations,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
