#!/usr/bin/env python3
"""Benchmark OpenNyAI pretrained NER and rhetorical-role models on local test splits."""

from __future__ import annotations

import argparse
import csv
import email
import importlib
import inspect
import json
import logging
import subprocess
import sys
import tempfile
import traceback
import urllib.parse
import urllib.request
import warnings
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from datasets import load_dataset
from tqdm import tqdm


LOGGER = logging.getLogger("benchmark_opennyai_pretrained")

MODEL_WHEEL_FALLBACK_URLS = {
    "en_legal_ner_trf": "https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl",
}

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

RR_LABELS = [
    "ANALYSIS",
    "ARG_PETITIONER",
    "ARG_RESPONDENT",
    "FAC",
    "ISSUE",
    "NONE",
    "PREAMBLE",
    "PRE_NOT_RELIED",
    "PRE_RELIED",
    "RATIO",
    "RLC",
    "RPC",
    "STA",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_dataset_root = script_dir / "ground_truth_datasets_Opennyai"
    default_output_root = script_dir / "outputs" / "opennyai_pretrained_benchmark"
    default_log_root = script_dir / "logs" / "opennyai_pretrained_benchmark"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=default_dataset_root,
        help="Directory containing InLegalNER/ and InRhetoricalRoles/.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=default_output_root,
        help="Directory where benchmark reports will be written.",
    )
    parser.add_argument(
        "--log_root",
        type=Path,
        default=default_log_root,
        help="Directory where benchmark logs will be written.",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["ner", "rr"],
        choices=["ner", "rr"],
        help="Which OpenNyAI components to benchmark.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Run OpenNyAI on GPU. Strongly recommended for this benchmark.",
    )
    parser.add_argument(
        "--ner_doc_batch_size",
        type=int,
        default=16,
        help="Number of NER documents to send to OpenNyAI per batch.",
    )
    parser.add_argument(
        "--rr_doc_batch_size",
        type=int,
        default=8,
        help="Number of RR documents to send to OpenNyAI per batch.",
    )
    parser.add_argument(
        "--preprocessing_model",
        type=str,
        default="en_core_web_trf",
        help="spaCy preprocessing model to pass into OpenNyAI Data(...).",
    )
    parser.add_argument(
        "--preprocessing_mini_batch_size",
        type=int,
        default=40000,
        help="mini_batch_size passed into OpenNyAI Data(...).",
    )
    parser.add_argument(
        "--ner_mini_batch_size",
        type=int,
        default=40000,
        help="ner_mini_batch_size passed into OpenNyAI Pipeline(...).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configure_logging(log_root: Path) -> Path:
    ensure_dir(log_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_root / f"benchmark_{timestamp}.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    return log_path


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for index in range(0, len(items), size):
        yield items[index : index + size]


def normalize_label(raw_label: str) -> str:
    return str(raw_label).strip().upper()


def import_opennyai_api() -> tuple[Any, Any]:
    """Import OpenNyAI defensively across minor package-layout differences."""
    import opennyai

    try:
        from opennyai.utils.preprocess import Data
    except Exception:
        Data = getattr(opennyai, "Data", None)

    try:
        from opennyai import Pipeline
    except Exception:
        from opennyai.pipeline import Pipeline

    if Data is None or Pipeline is None:
        raise ImportError("Unable to import OpenNyAI Data/Pipeline classes from the installed package.")
    return Data, Pipeline


def filter_supported_kwargs(target: Any, kwargs: Dict[str, Any], label: str) -> Dict[str, Any]:
    signature = inspect.signature(target)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return kwargs

    supported_names = set(signature.parameters)
    supported = {key: value for key, value in kwargs.items() if key in supported_names}
    ignored = sorted(set(kwargs) - supported_names)
    if ignored:
        LOGGER.warning("Ignoring unsupported %s kwargs for this OpenNyAI version: %s", label, ignored)
    return supported


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def _repair_wheel_filename(downloaded_wheel: Path) -> Path:
    with zipfile.ZipFile(downloaded_wheel) as wheel_file:
        metadata_path = next(name for name in wheel_file.namelist() if name.endswith(".dist-info/METADATA"))
        wheel_path = next(name for name in wheel_file.namelist() if name.endswith(".dist-info/WHEEL"))
        metadata_message = email.message_from_bytes(wheel_file.read(metadata_path))
        wheel_message = email.message_from_bytes(wheel_file.read(wheel_path))

    distribution_name = metadata_message.get("Name", downloaded_wheel.stem).replace("-", "_")
    version = metadata_message.get("Version", "0.0.0")
    tag = wheel_message.get_all("Tag", ["py3-none-any"])[0]

    repaired_wheel = downloaded_wheel.with_name(f"{distribution_name}-{version}-{tag}.whl")
    if repaired_wheel != downloaded_wheel:
        downloaded_wheel.replace(repaired_wheel)
    return repaired_wheel


def _install_model_wheel_with_filename_repair(model_name: str, model_url: str) -> None:
    LOGGER.info("Downloading %s model wheel from %s", model_name, model_url)
    with tempfile.TemporaryDirectory() as temporary_directory:
        parsed_url = urllib.parse.urlparse(model_url)
        download_name = Path(parsed_url.path).name or f"{model_name}.whl"
        download_path = Path(temporary_directory) / download_name
        _download_file(model_url, download_path)
        repaired_wheel = _repair_wheel_filename(download_path)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(repaired_wheel)], check=True)


def ensure_ner_model_available(model_name: str) -> None:
    """Install the requested spaCy model package if OpenNyAI has not installed it yet."""
    import spacy

    if model_name in spacy.util.get_installed_models():
        return

    LOGGER.info("spaCy model %s is missing. Attempting installation.", model_name)
    model_url = MODEL_WHEEL_FALLBACK_URLS.get(model_name)
    try:
        from opennyai.utils.download import PIP_INSTALLER_URLS, install

        model_url = PIP_INSTALLER_URLS.get(model_name, model_url)
        if model_url is None:
            raise RuntimeError(f"No download URL found for model {model_name}.")
        try:
            install(model_url)
        except Exception as exc:
            LOGGER.warning(
                "OpenNyAI installer failed for %s (%s). Falling back to wheel filename repair.",
                model_name,
                exc,
            )
            _install_model_wheel_with_filename_repair(model_name, model_url)
    except Exception:
        if model_url is None:
            raise
        _install_model_wheel_with_filename_repair(model_name, model_url)


def patch_opennyai_ner_empty_section_bug() -> None:
    """Work around an OpenNyAI NER crash when preamble or judgment slices are empty."""
    eru = importlib.import_module("opennyai.ner.InLegalNER.entity_recognizer_utils")
    inlegal_module = importlib.import_module("opennyai.ner.InLegalNER.InLegalNER")

    if getattr(eru, "_codex_safe_patch_applied", False):
        return

    original_get_sentence_docs = eru.get_sentence_docs
    original_process_nlp_in_chunks = eru.process_nlp_in_chunks

    def ensure_sentence_boundaries(doc: Any) -> Any:
        if doc is None or len(doc) == 0:
            return doc
        if doc.has_annotation("SENT_START"):
            return doc
        doc[0].is_sent_start = True
        for token in doc[1:]:
            token.is_sent_start = False
        return doc

    def process_preamble_section(section_doc: Any, legal_nlp: Any, mini_batch_size: int, do_sentence_level: bool) -> Any:
        if section_doc is None or not section_doc.text.strip():
            return None
        processed_doc = original_process_nlp_in_chunks(section_doc.text, mini_batch_size, legal_nlp)
        return ensure_sentence_boundaries(processed_doc) if do_sentence_level else processed_doc

    def process_judgement_section(section_doc: Any, legal_nlp: Any, mini_batch_size: int, do_sentence_level: bool) -> Any:
        if section_doc is None or not section_doc.text.strip():
            return None
        if do_sentence_level:
            try:
                return ensure_sentence_boundaries(original_get_sentence_docs(section_doc, legal_nlp))
            except Exception:
                pass
        processed_doc = original_process_nlp_in_chunks(section_doc.text, mini_batch_size, legal_nlp)
        return ensure_sentence_boundaries(processed_doc) if do_sentence_level else processed_doc

    def safe_extract_entities_from_judgment_text(
        to_process: Dict[str, Any],
        legal_nlp: Any,
        mini_batch_size: int,
        do_sentence_level: bool = True,
    ) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            preamble = to_process.get("preamble_doc")
            judgement = to_process.get("judgement_doc")

            doc_preamble = process_preamble_section(preamble, legal_nlp, mini_batch_size, do_sentence_level)
            doc_judgement = process_judgement_section(judgement, legal_nlp, mini_batch_size, do_sentence_level)

            docs_to_merge = [doc for doc in (doc_preamble, doc_judgement) if doc is not None]
            if not docs_to_merge:
                return legal_nlp.make_doc("")
            if len(docs_to_merge) == 1:
                return docs_to_merge[0]
            return eru.spacy.tokens.Doc.from_docs(docs_to_merge)

    eru.extract_entities_from_judgment_text = safe_extract_entities_from_judgment_text
    inlegal_module.extract_entities_from_judgment_text = safe_extract_entities_from_judgment_text
    eru._codex_safe_patch_applied = True
    LOGGER.info("Applied OpenNyAI NER empty-section compatibility patch.")


def build_data_object(
    data_cls: Any,
    texts: List[str],
    file_ids: List[str],
    preprocessing_model: str,
    preprocessing_mini_batch_size: int,
    use_gpu: bool,
) -> Any:
    kwargs = {
        "input_text": texts,
        "file_ids": file_ids,
        "preprocessing_nlp_model": preprocessing_model,
        "mini_batch_size": preprocessing_mini_batch_size,
        "use_gpu": use_gpu,
        "use_cache": True,
        "verbose": False,
    }
    return data_cls(**filter_supported_kwargs(data_cls, kwargs, "Data"))


def build_pipeline_object(
    pipeline_cls: Any,
    component: str,
    use_gpu: bool,
    ner_mini_batch_size: int,
) -> Any:
    kwargs: Dict[str, Any] = {
        "components": [component],
        "use_gpu": use_gpu,
        "verbose": False,
    }
    if component == "NER":
        kwargs.update(
            {
                "ner_model_name": "en_legal_ner_trf",
                "ner_do_sentence_level": True,
                "ner_do_postprocess": True,
                "ner_mini_batch_size": ner_mini_batch_size,
            }
        )
    return pipeline_cls(**filter_supported_kwargs(pipeline_cls, kwargs, "Pipeline"))


def log_runtime_environment(use_gpu: bool) -> None:
    if use_gpu:
        import spacy

        spacy_gpu_enabled = spacy.prefer_gpu()
        LOGGER.info("spacy.prefer_gpu()=%s", spacy_gpu_enabled)
    import torch

    gpu_available = torch.cuda.is_available()
    LOGGER.info("torch.__version__=%s", torch.__version__)
    LOGGER.info("torch.cuda.is_available()=%s", gpu_available)
    if gpu_available:
        LOGGER.info("Visible GPU count: %s", torch.cuda.device_count())
        LOGGER.info("Primary GPU: %s", torch.cuda.get_device_name(0))
    if use_gpu and not gpu_available:
        raise RuntimeError("--use_gpu was set but CUDA is unavailable in this environment.")


def load_inlegalner_test_rows(dataset_root: Path) -> List[Dict[str, Any]]:
    dataset_dir = dataset_root / "InLegalNER"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")
    dataset = load_dataset(str(dataset_dir))
    test_split = dataset["test"]
    return [test_split[index] for index in range(len(test_split))]


def load_inrhetoricalroles_test_rows(dataset_root: Path) -> List[Dict[str, Any]]:
    json_path = dataset_root / "InRhetoricalRoles" / "test.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a list in {json_path}, found {type(payload).__name__}")
    return payload


def extract_labelstudio_spans(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for annotation in row.get("annotations", []):
        for result in annotation.get("result", []):
            value = result.get("value", {})
            labels = value.get("labels", []) or []
            if not labels:
                continue
            start = int(value.get("start", 0))
            end = int(value.get("end", 0))
            text = str(value.get("text", ""))
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                    "label": normalize_label(labels[0]),
                }
            )
    spans.sort(key=lambda item: (item["start"], item["end"], item["label"], item["text"]))
    return spans


def extract_predicted_entities(prediction: Dict[str, Any], source_text: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    seen = set()
    for sentence in prediction.get("annotations", []):
        for entity in sentence.get("entities", []) or []:
            labels = entity.get("labels", []) or []
            if not labels:
                continue
            start = int(entity.get("start", 0))
            end = int(entity.get("end", 0))
            label = normalize_label(labels[0])
            key = (start, end, label)
            if key in seen:
                continue
            seen.add(key)
            text = str(entity.get("text") or source_text[start:end])
            entities.append({"start": start, "end": end, "text": text, "label": label})
    entities.sort(key=lambda item: (item["start"], item["end"], item["label"], item["text"]))
    return entities


def extract_predicted_rr_segments(prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for annotation in prediction.get("annotations", []):
        labels = annotation.get("labels", []) or []
        if not labels:
            continue
        segments.append(
            {
                "start": int(annotation.get("start", 0)),
                "end": int(annotation.get("end", 0)),
                "text": str(annotation.get("text", "")),
                "label": normalize_label(labels[0]),
            }
        )
    segments.sort(key=lambda item: (item["start"], item["end"], item["label"], item["text"]))
    return segments


def overlap_length(first: Dict[str, Any], second: Dict[str, Any]) -> int:
    return max(0, min(first["end"], second["end"]) - max(first["start"], second["start"]))


def choose_rr_prediction_for_gold_segment(
    gold_segment: Dict[str, Any], predicted_segments: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    label_overlaps: Counter[str] = Counter()
    best_segment_for_label: Dict[str, Dict[str, Any]] = {}
    total_overlap = 0

    for predicted_segment in predicted_segments:
        overlap = overlap_length(gold_segment, predicted_segment)
        if overlap <= 0:
            continue
        total_overlap += overlap
        label = predicted_segment["label"]
        label_overlaps[label] += overlap
        previous_best = best_segment_for_label.get(label)
        if previous_best is None or overlap > previous_best["overlap"]:
            best_segment_for_label[label] = {"overlap": overlap, "segment": predicted_segment}

    if not label_overlaps:
        return {
            "pred_label": "NO_PREDICTION",
            "pred_text": "",
            "best_overlap_chars": 0,
            "overlap_coverage": 0.0,
        }

    pred_label, best_overlap_chars = max(
        sorted(label_overlaps.items()),
        key=lambda item: item[1],
    )
    best_segment = best_segment_for_label[pred_label]["segment"]
    gold_length = max(gold_segment["end"] - gold_segment["start"], 1)
    return {
        "pred_label": pred_label,
        "pred_text": best_segment["text"],
        "best_overlap_chars": best_overlap_chars,
        "overlap_coverage": safe_div(total_overlap, gold_length),
    }


def compute_binary_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
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


def score_ner_documents(
    rows: Sequence[Dict[str, Any]],
    predictions_by_doc_id: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    aggregate_gold = set()
    aggregate_pred = set()
    per_label_tp: Counter[str] = Counter()
    per_label_fp: Counter[str] = Counter()
    per_label_fn: Counter[str] = Counter()
    per_doc_rows: List[Dict[str, Any]] = []
    exact_match_docs = 0

    for row in rows:
        doc_id = row["benchmark_doc_id"]
        text = row["data"]["text"]
        prediction = predictions_by_doc_id.get(doc_id, {"annotations": []})
        gold_spans = extract_labelstudio_spans(row)
        pred_spans = extract_predicted_entities(prediction, text)

        gold_set = {(span["start"], span["end"], span["label"]) for span in gold_spans}
        pred_set = {(span["start"], span["end"], span["label"]) for span in pred_spans}
        tp_items = gold_set & pred_set
        fp_items = pred_set - gold_set
        fn_items = gold_set - pred_set

        aggregate_gold.update((doc_id, start, end, label) for start, end, label in gold_set)
        aggregate_pred.update((doc_id, start, end, label) for start, end, label in pred_set)

        for _, _, label in tp_items:
            per_label_tp[label] += 1
        for _, _, label in fp_items:
            per_label_fp[label] += 1
        for _, _, label in fn_items:
            per_label_fn[label] += 1

        if not fp_items and not fn_items:
            exact_match_docs += 1

        doc_metrics = compute_binary_metrics(len(tp_items), len(fp_items), len(fn_items))
        per_doc_rows.append(
            {
                "doc_id": doc_id,
                "text_length": len(text),
                "gold_entities": len(gold_set),
                "pred_entities": len(pred_set),
                "true_positive": len(tp_items),
                "false_positive": len(fp_items),
                "false_negative": len(fn_items),
                "precision": doc_metrics["precision"],
                "recall": doc_metrics["recall"],
                "f1": doc_metrics["f1"],
                "strict_accuracy": doc_metrics["strict_accuracy"],
                "exact_match": int(not fp_items and not fn_items),
            }
        )

    total_gold = len(aggregate_gold)
    total_pred = len(aggregate_pred)
    true_positive = sum(per_label_tp.values())
    false_positive = sum(per_label_fp.values())
    false_negative = sum(per_label_fn.values())
    metrics = compute_binary_metrics(true_positive, false_positive, false_negative)

    per_label = {}
    label_names = sorted(set(NER_LABELS) | set(per_label_tp) | set(per_label_fp) | set(per_label_fn))
    for label in label_names:
        label_tp = per_label_tp[label]
        label_fp = per_label_fp[label]
        label_fn = per_label_fn[label]
        label_metrics = compute_binary_metrics(label_tp, label_fp, label_fn)
        per_label[label] = {
            "gold_support": label_tp + label_fn,
            "pred_support": label_tp + label_fp,
            "true_positive": label_tp,
            "false_positive": label_fp,
            "false_negative": label_fn,
            **label_metrics,
        }

    report = {
        "task": "NER",
        "metric_definition": "Strict entity-level span+label exact match. strict_accuracy = TP / (TP + FP + FN).",
        "documents_evaluated": len(rows),
        "exact_match_documents": exact_match_docs,
        "document_exact_match_accuracy": safe_div(exact_match_docs, len(rows)),
        "total_gold_entities": total_gold,
        "total_predicted_entities": total_pred,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        **metrics,
        "per_label": per_label,
    }
    return report, per_doc_rows


def compute_classification_metrics(
    aligned_rows: Sequence[Dict[str, Any]],
    label_names: Sequence[str],
) -> Dict[str, Any]:
    total = len(aligned_rows)
    correct = sum(1 for row in aligned_rows if row["gold_label"] == row["pred_label"])
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_label: Dict[str, Dict[str, Any]] = {}
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0

    for row in aligned_rows:
        confusion[row["gold_label"]][row["pred_label"]] += 1

    for label in label_names:
        tp = sum(1 for row in aligned_rows if row["gold_label"] == label and row["pred_label"] == label)
        fp = sum(1 for row in aligned_rows if row["gold_label"] != label and row["pred_label"] == label)
        fn = sum(1 for row in aligned_rows if row["gold_label"] == label and row["pred_label"] != label)
        support = sum(1 for row in aligned_rows if row["gold_label"] == label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support
        per_label[label] = {
            "support": support,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "num_segments": total,
        "accuracy": safe_div(correct, total),
        "macro_f1": safe_div(macro_f1_sum, len(label_names)),
        "weighted_f1": safe_div(weighted_f1_sum, total),
        "per_label": per_label,
        "gold_distribution": dict(sorted(Counter(row["gold_label"] for row in aligned_rows).items())),
        "pred_distribution": dict(sorted(Counter(row["pred_label"] for row in aligned_rows).items())),
        "confusion_matrix": {
            gold_label: dict(sorted(pred_counts.items()))
            for gold_label, pred_counts in sorted(confusion.items())
        },
    }


def score_rr_documents(
    rows: Sequence[Dict[str, Any]],
    predictions_by_doc_id: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    aligned_rows: List[Dict[str, Any]] = []
    per_doc_rows: List[Dict[str, Any]] = []

    for row in rows:
        doc_id = row["benchmark_doc_id"]
        gold_segments = extract_labelstudio_spans(row)
        predicted_segments = extract_predicted_rr_segments(predictions_by_doc_id.get(doc_id, {"annotations": []}))

        doc_correct = 0
        for segment_index, gold_segment in enumerate(gold_segments):
            chosen = choose_rr_prediction_for_gold_segment(gold_segment, predicted_segments)
            match = int(chosen["pred_label"] == gold_segment["label"])
            doc_correct += match
            aligned_rows.append(
                {
                    "doc_id": doc_id,
                    "segment_index": segment_index,
                    "gold_start": gold_segment["start"],
                    "gold_end": gold_segment["end"],
                    "gold_label": gold_segment["label"],
                    "pred_label": chosen["pred_label"],
                    "match": match,
                    "best_overlap_chars": chosen["best_overlap_chars"],
                    "overlap_coverage": chosen["overlap_coverage"],
                    "gold_text": gold_segment["text"],
                    "pred_text": chosen["pred_text"],
                }
            )

        segment_count = len(gold_segments)
        per_doc_rows.append(
            {
                "doc_id": doc_id,
                "gold_segments": segment_count,
                "pred_segments": len(predicted_segments),
                "correct_segments": doc_correct,
                "accuracy": safe_div(doc_correct, segment_count),
            }
        )

    classification_metrics = compute_classification_metrics(aligned_rows, RR_LABELS)
    mean_overlap_coverage = safe_div(
        sum(row["overlap_coverage"] for row in aligned_rows),
        len(aligned_rows),
    )

    report = {
        "task": "Rhetorical_Role",
        "metric_definition": (
            "Gold-segment accuracy. Each gold segment is assigned the predicted label with the largest "
            "character-overlap mass across overlapping OpenNyAI predicted segments."
        ),
        "documents_evaluated": len(rows),
        "mean_overlap_coverage": mean_overlap_coverage,
        **classification_metrics,
    }
    return report, aligned_rows, per_doc_rows


def prepare_ner_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows = []
    for index, row in enumerate(rows):
        prepared = dict(row)
        prepared["benchmark_doc_id"] = str(row.get("id") or f"nerdoc{index:05d}")
        prepared_rows.append(prepared)
    return prepared_rows


def prepare_rr_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows = []
    for index, row in enumerate(rows):
        prepared = dict(row)
        prepared["benchmark_doc_id"] = f"rrdoc{index:05d}"
        prepared_rows.append(prepared)
    return prepared_rows


def run_predictions(
    rows: Sequence[Dict[str, Any]],
    task_name: str,
    component_name: str,
    doc_batch_size: int,
    preprocessing_model: str,
    preprocessing_mini_batch_size: int,
    use_gpu: bool,
    ner_mini_batch_size: int,
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]]]:
    data_cls, pipeline_cls = import_opennyai_api()
    if component_name == "NER":
        ensure_ner_model_available("en_legal_ner_trf")
        patch_opennyai_ner_empty_section_bug()

    pipeline = build_pipeline_object(
        pipeline_cls=pipeline_cls,
        component=component_name,
        use_gpu=use_gpu,
        ner_mini_batch_size=ner_mini_batch_size,
    )
    predictions_by_doc_id: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, str]] = []

    progress = tqdm(list(chunked(list(rows), doc_batch_size)), desc=f"{task_name} batches", unit="batch")
    for batch_rows in progress:
        batch_doc_ids = [row["benchmark_doc_id"] for row in batch_rows]
        batch_texts = [row["data"]["text"] for row in batch_rows]

        try:
            data_object = build_data_object(
                data_cls=data_cls,
                texts=batch_texts,
                file_ids=batch_doc_ids,
                preprocessing_model=preprocessing_model,
                preprocessing_mini_batch_size=preprocessing_mini_batch_size,
                use_gpu=use_gpu,
            )
            predictions = pipeline(data_object)
            for prediction in predictions:
                predictions_by_doc_id[str(prediction.get("id"))] = prediction
        except Exception:
            LOGGER.exception("Batch failure while running %s for doc_ids=%s", task_name, batch_doc_ids)
            for row in batch_rows:
                single_doc_id = row["benchmark_doc_id"]
                try:
                    data_object = build_data_object(
                        data_cls=data_cls,
                        texts=[row["data"]["text"]],
                        file_ids=[single_doc_id],
                        preprocessing_model=preprocessing_model,
                        preprocessing_mini_batch_size=preprocessing_mini_batch_size,
                        use_gpu=use_gpu,
                    )
                    predictions = pipeline(data_object)
                    predictions_by_doc_id[str(predictions[0].get("id"))] = predictions[0]
                except Exception:
                    LOGGER.exception("Single-document failure while running %s for doc_id=%s", task_name, single_doc_id)
                    failures.append(
                        {
                            "doc_id": single_doc_id,
                            "task": task_name,
                            "traceback": traceback.format_exc(),
                        }
                    )
    return predictions_by_doc_id, failures


def benchmark_ner(args: argparse.Namespace) -> Dict[str, Any]:
    LOGGER.info("Loading InLegalNER test split from %s", args.dataset_root)
    rows = prepare_ner_rows(load_inlegalner_test_rows(args.dataset_root))
    LOGGER.info("Loaded %s NER test documents", len(rows))

    predictions_by_doc_id, failures = run_predictions(
        rows=rows,
        task_name="NER",
        component_name="NER",
        doc_batch_size=args.ner_doc_batch_size,
        preprocessing_model=args.preprocessing_model,
        preprocessing_mini_batch_size=args.preprocessing_mini_batch_size,
        use_gpu=args.use_gpu,
        ner_mini_batch_size=args.ner_mini_batch_size,
    )

    successful_rows = [row for row in rows if row["benchmark_doc_id"] in predictions_by_doc_id]
    report, per_doc_rows = score_ner_documents(successful_rows, predictions_by_doc_id)
    report["failed_documents"] = [failure["doc_id"] for failure in failures]
    report["num_failed_documents"] = len(failures)

    output_dir = ensure_dir(args.output_root / "ner")
    write_json(output_dir / "test_metrics.json", report)
    write_csv(output_dir / "test_doc_metrics.csv", per_doc_rows)
    write_json(output_dir / "test_failures.json", failures)
    return report


def benchmark_rr(args: argparse.Namespace) -> Dict[str, Any]:
    LOGGER.info("Loading InRhetoricalRoles test split from %s", args.dataset_root)
    rows = prepare_rr_rows(load_inrhetoricalroles_test_rows(args.dataset_root))
    LOGGER.info("Loaded %s RR test documents", len(rows))

    predictions_by_doc_id, failures = run_predictions(
        rows=rows,
        task_name="Rhetorical_Role",
        component_name="Rhetorical_Role",
        doc_batch_size=args.rr_doc_batch_size,
        preprocessing_model=args.preprocessing_model,
        preprocessing_mini_batch_size=args.preprocessing_mini_batch_size,
        use_gpu=args.use_gpu,
        ner_mini_batch_size=args.ner_mini_batch_size,
    )

    successful_rows = [row for row in rows if row["benchmark_doc_id"] in predictions_by_doc_id]
    report, aligned_rows, per_doc_rows = score_rr_documents(successful_rows, predictions_by_doc_id)
    report["failed_documents"] = [failure["doc_id"] for failure in failures]
    report["num_failed_documents"] = len(failures)

    output_dir = ensure_dir(args.output_root / "rhetorical_roles")
    write_json(output_dir / "test_metrics.json", report)
    write_csv(output_dir / "test_segment_alignment.csv", aligned_rows)
    write_csv(output_dir / "test_doc_metrics.csv", per_doc_rows)
    write_json(output_dir / "test_failures.json", failures)
    return report


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    log_path = configure_logging(args.log_root)
    LOGGER.info("CLI command: %s", " ".join(sys.argv))
    LOGGER.info("Benchmark log path: %s", log_path)
    log_runtime_environment(args.use_gpu)

    reports: Dict[str, Any] = {}

    if "ner" in args.components:
        reports["ner"] = benchmark_ner(args)
    if "rr" in args.components:
        reports["rr"] = benchmark_rr(args)

    summary_path = args.output_root / "benchmark_summary.json"
    write_json(summary_path, reports)
    LOGGER.info("Wrote benchmark summary to %s", summary_path)


if __name__ == "__main__":
    main()
