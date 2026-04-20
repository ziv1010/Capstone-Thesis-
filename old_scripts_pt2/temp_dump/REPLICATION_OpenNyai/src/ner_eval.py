"""Paper-faithful evaluation for the Legal NER pretrained model."""

from __future__ import annotations

import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm

from src.common import safe_div, write_csv, write_json, write_jsonl


NER_LABELS = [
    "CASE_NUMBER",
    "COURT",
    "DATE",
    "GPE",
    "JUDGE",
    "LAWYER",
    "ORG",
    "OTHER_PERSON",
    "PETITIONER",
    "PRECEDENT",
    "PROVISION",
    "RESPONDENT",
    "STATUTE",
    "WITNESS",
]


def prepare_legal_ner_imports(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def normalize_label(raw_label: str) -> str:
    return str(raw_label).strip().upper()


def load_test_rows(dataset_root: Path) -> list[dict[str, Any]]:
    dataset = load_dataset(str(dataset_root / "InLegalNER"))
    test_split = dataset["test"]
    rows: list[dict[str, Any]] = []
    for index in range(len(test_split)):
        row = dict(test_split[index])
        row["benchmark_doc_id"] = str(row.get("id") or f"nerdoc{index:05d}")
        rows.append(row)
    return rows


def extract_gold_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for annotation in row.get("annotations", []):
        for result in annotation.get("result", []):
            value = result.get("value", {})
            labels = value.get("labels", []) or []
            if not labels:
                continue
            spans.append(
                {
                    "start": int(value["start"]),
                    "end": int(value["end"]),
                    "text": str(value.get("text", "")),
                    "label": normalize_label(labels[0]),
                }
            )
    spans.sort(key=lambda item: (item["start"], item["end"], item["label"], item["text"]))
    return spans


def normalize_judgment_text_with_mapping(judgment_text: str, base_offset: int) -> tuple[str, list[int]]:
    """Mirror legal_NER's newline cleanup while keeping a char map to the original text."""
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

    normalized = "".join(output_chars)
    if len(normalized) != len(mapping):
        raise RuntimeError("Judgment-text normalization produced an invalid mapping.")
    return normalized, mapping


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


def map_entity_back_to_original(
    start_char: int,
    end_char: int,
    char_mapping: list[int],
) -> tuple[int, int]:
    if start_char >= len(char_mapping) or end_char <= 0:
        raise RuntimeError("Entity offsets are outside the mapped text.")
    start_original = char_mapping[start_char]
    end_original = char_mapping[end_char - 1] + 1
    return start_original, end_original


def ensure_sentence_boundaries(doc: Any) -> Any:
    if doc is None or len(doc) == 0:
        return doc
    if doc.has_annotation("SENT_START"):
        return doc
    doc[0].is_sent_start = True
    for token in doc[1:]:
        token.is_sent_start = False
    return doc


def compute_binary_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    strict_accuracy = safe_div(tp, tp + fp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "strict_accuracy": strict_accuracy,
    }


def score_predictions(
    rows: list[dict[str, Any]],
    predictions_by_doc_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_label_tp: Counter[str] = Counter()
    per_label_fp: Counter[str] = Counter()
    per_label_fn: Counter[str] = Counter()
    per_doc_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    exact_match_docs = 0

    for row in rows:
        doc_id = row["benchmark_doc_id"]
        gold_spans = extract_gold_spans(row)
        predicted_spans = predictions_by_doc_id.get(doc_id, {}).get("entities", [])

        gold_set = {(item["start"], item["end"], item["label"]) for item in gold_spans}
        predicted_set = {(item["start"], item["end"], item["label"]) for item in predicted_spans}
        tp_items = gold_set & predicted_set
        fp_items = predicted_set - gold_set
        fn_items = gold_set - predicted_set

        for _, _, label in tp_items:
            per_label_tp[label] += 1
        for _, _, label in fp_items:
            per_label_fp[label] += 1
        for _, _, label in fn_items:
            per_label_fn[label] += 1

        exact_match = int(not fp_items and not fn_items)
        exact_match_docs += exact_match

        doc_metrics = compute_binary_metrics(len(tp_items), len(fp_items), len(fn_items))
        per_doc_rows.append(
            {
                "doc_id": doc_id,
                "gold_entities": len(gold_set),
                "pred_entities": len(predicted_set),
                "true_positive": len(tp_items),
                "false_positive": len(fp_items),
                "false_negative": len(fn_items),
                "precision": doc_metrics["precision"],
                "recall": doc_metrics["recall"],
                "f1": doc_metrics["f1"],
                "strict_accuracy": doc_metrics["strict_accuracy"],
                "exact_match": exact_match,
            }
        )
        prediction_rows.append(
            {
                "doc_id": doc_id,
                "num_entities": len(predicted_spans),
                "entities": predicted_spans,
            }
        )

    true_positive = sum(per_label_tp.values())
    false_positive = sum(per_label_fp.values())
    false_negative = sum(per_label_fn.values())
    overall = compute_binary_metrics(true_positive, false_positive, false_negative)

    per_label_rows: list[dict[str, Any]] = []
    per_label_payload: dict[str, Any] = {}
    for label in sorted(set(NER_LABELS) | set(per_label_tp) | set(per_label_fp) | set(per_label_fn)):
        label_tp = per_label_tp[label]
        label_fp = per_label_fp[label]
        label_fn = per_label_fn[label]
        label_metrics = compute_binary_metrics(label_tp, label_fp, label_fn)
        row_payload = {
            "label": label,
            "gold_support": label_tp + label_fn,
            "pred_support": label_tp + label_fp,
            "true_positive": label_tp,
            "false_positive": label_fp,
            "false_negative": label_fn,
            **label_metrics,
        }
        per_label_rows.append(row_payload)
        per_label_payload[label] = dict(row_payload)

    report = {
        "task": "NER",
        "metric_definition": "Strict entity-level span+label exact match over full judgment texts.",
        "documents_evaluated": len(rows),
        "exact_match_documents": exact_match_docs,
        "document_exact_match_accuracy": safe_div(exact_match_docs, len(rows)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        **overall,
        "per_label": per_label_payload,
    }
    return report, per_doc_rows, per_label_rows


def run_ner_replication(
    *,
    dataset_root: Path,
    output_dir: Path,
    repo_root: Path,
    logger: Any,
    use_gpu: bool,
    gpu_id: int,
    model_name: str,
    preamble_model_name: str,
    run_type: str,
    do_postprocess: bool,
) -> dict[str, Any]:
    import spacy

    prepare_legal_ner_imports(repo_root)
    from data_preparation import seperate_and_clean_preamble
    from postprocessing_utils import postprocessing

    if use_gpu:
        spacy.prefer_gpu(gpu_id)

    logger.info("Loading spaCy models: %s and %s", model_name, preamble_model_name)
    legal_nlp = spacy.load(model_name)
    preamble_nlp = spacy.load(preamble_model_name)

    rows = load_test_rows(dataset_root)
    logger.info("Loaded %s InLegalNER test documents", len(rows))

    predictions_by_doc_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for row in tqdm(rows, desc="NER documents", unit="doc"):
        doc_id = row["benchmark_doc_id"]
        text = row["data"]["text"]
        try:
            preamble_text, preamble_end = seperate_and_clean_preamble(text, preamble_nlp)
            judgment_text_original = text[preamble_end:]
            judgment_text, judgment_mapping = normalize_judgment_text_with_mapping(
                judgment_text_original,
                preamble_end,
            )

            preamble_doc = legal_nlp(preamble_text) if preamble_text.strip() else legal_nlp.make_doc("")
            preamble_doc = ensure_sentence_boundaries(preamble_doc)
            preamble_mapping = list(range(len(preamble_doc.text)))

            if run_type == "doc":
                judgment_doc = legal_nlp(judgment_text) if judgment_text.strip() else legal_nlp.make_doc("")
                judgment_doc = ensure_sentence_boundaries(judgment_doc)
                judgment_combined_mapping = judgment_mapping[: len(judgment_doc.text)]
            else:
                sentence_boundary_doc = preamble_nlp(judgment_text)
                sentence_spans = [sent for sent in sentence_boundary_doc.sents if sent.text.strip()]
                sentence_texts = [sent.text for sent in sentence_spans]
                sentence_mappings = [
                    judgment_mapping[sent.start_char : sent.end_char]
                    for sent in sentence_spans
                ]
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
            parts_to_merge = []
            if preamble_doc.text:
                parts_to_merge.append((preamble_doc.text, preamble_mapping))
            if judgment_doc.text:
                parts_to_merge.append((judgment_doc.text, judgment_combined_mapping))

            if not docs_to_merge:
                combined_doc = legal_nlp.make_doc("")
                combined_mapping = []
            elif len(docs_to_merge) == 1:
                combined_doc = docs_to_merge[0]
                combined_mapping = parts_to_merge[0][1]
            else:
                combined_doc = spacy.tokens.Doc.from_docs(docs_to_merge)
                combined_mapping = build_mapping_for_combined_text(parts_to_merge, combined_doc.text)
            combined_doc = ensure_sentence_boundaries(combined_doc)

            if do_postprocess and combined_doc.text:
                try:
                    combined_doc = postprocessing(combined_doc)
                    combined_doc = ensure_sentence_boundaries(combined_doc)
                except Exception:
                    logger.exception("NER postprocessing failed for %s; continuing with raw model output.", doc_id)

            entities: list[dict[str, Any]] = []
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
            predictions_by_doc_id[doc_id] = {"entities": entities}
        except Exception:
            failure = {
                "doc_id": doc_id,
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            logger.exception("NER inference failed for %s", doc_id)

    successful_rows = [row for row in rows if row["benchmark_doc_id"] in predictions_by_doc_id]
    report, doc_rows, per_label_rows = score_predictions(successful_rows, predictions_by_doc_id)
    report["failed_documents"] = [item["doc_id"] for item in failures]
    report["num_failed_documents"] = len(failures)
    report["runtime"] = {
        "model_name": model_name,
        "preamble_model_name": preamble_model_name,
        "use_gpu": use_gpu,
        "gpu_id": gpu_id,
        "run_type": run_type,
        "do_postprocess": do_postprocess,
    }

    write_json(output_dir / "test_metrics.json", report)
    write_csv(output_dir / "per_label_metrics.csv", per_label_rows)
    write_csv(output_dir / "document_metrics.csv", doc_rows)
    write_json(output_dir / "failures.json", failures)
    write_jsonl(output_dir / "predictions.jsonl", _prediction_jsonl_rows(successful_rows, predictions_by_doc_id))
    return report


def _prediction_jsonl_rows(
    rows: Iterable[dict[str, Any]],
    predictions_by_doc_id: dict[str, dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        doc_id = row["benchmark_doc_id"]
        yield {
            "doc_id": doc_id,
            "text_length": len(row["data"]["text"]),
            "predicted_entities": predictions_by_doc_id[doc_id]["entities"],
        }
