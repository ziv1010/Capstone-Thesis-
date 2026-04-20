#!/usr/bin/env python3
"""Run OpenNyAI summarization over existing annotation JSONs, then label outcomes.

This script is designed for the JSON files produced by `run_ner_rr_custom.py`.
It converts those lightweight sentence-level annotation payloads back into the
OpenNyAI-style annotation structure expected by the standalone
`ExtractiveSummarizer`, and then applies the same LLM-based case-outcome
classification flow used by `add_case_outcome_labels_mistral.py`.

Output layout:
  <output_dir>/
    enriched_jsons/
    logs/
    case_outcomes.jsonl
    run_summary.json

If `--write_repo_style_outputs` is set, the script also writes the intermediate
repo-style `combined/`, `annotations/`, `ner/`, `rhetorical_roles/`, and
`summaries/` folders.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from add_case_outcome_labels_mistral import (  # noqa: E402
    DEFAULT_MODEL_ID,
    LABEL_TO_SCORE,
    apply_classification_to_payload,
    build_classifier,
    collect_input_texts,
)
from src.config import (  # noqa: E402
    build_output_layout,
    configure_local_runtime,
    document_output_paths,
    ensure_output_layout,
    load_project_environment,
)
from src.io_utils import configure_logging, write_json, write_text  # noqa: E402
from src.output_formatter import (  # noqa: E402
    build_combined_output,
    build_ner_output,
    build_rhetorical_roles_output,
    build_sentence_annotations_output,
    build_summary_output,
)
from src.pipeline_runner import patch_opennyai_summarizer_device_mismatch  # noqa: E402
from src.validators import normalize_summary_length, validate_gpu_request  # noqa: E402


DEFAULT_ANNOTATIONS_DIR = (
    PROJECT_ROOT / "fin_fraud_extract" / "annotations"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "summary_label_from_annotations"
)

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations_dir", default=str(DEFAULT_ANNOTATIONS_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--summary_length",
        type=float,
        default=0.0,
        help="Summary length fraction in the 0.0-1.0 range. Use 0.0 for adaptive mode.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Require GPU for the OpenNyAI summarizer stage.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value shared by summarizer and outcome-label stages.",
    )
    parser.add_argument("--max_files", type=int, help="Cap number of annotation JSONs processed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir first and recompute every file.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run summarization and write combined outputs, but skip LLM outcome labeling.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose progress from the OpenNyAI summarizer.",
    )
    parser.add_argument(
        "--write_repo_style_outputs",
        action="store_true",
        help="Also write intermediate combined/annotations/ner/rhetorical_roles/summaries folders.",
    )

    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", choices=["auto", "local_vllm", "remote_hf"], default="auto")
    parser.add_argument("--provider", default="auto")
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    parser.add_argument("--sleep_seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_output_tokens", type=int, default=200)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=16,
        help="Number of cases to batch into one local vLLM generate call.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tokenizer_mode", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    return parser.parse_args()


def normalize_rr_role(value: Any) -> str:
    role = str(value or "NONE").strip().upper() or "NONE"
    if role not in VALID_RR_LABELS:
        return "NONE"
    return role


def build_openai_style_entity(entity: dict[str, Any]) -> dict[str, Any]:
    label = str(entity.get("label") or "").strip()
    labels = [label] if label else []
    return {
        "text": str(entity.get("text", "")).strip(),
        "start": entity.get("start"),
        "end": entity.get("end"),
        "labels": labels,
    }


def convert_annotation_payload_to_raw_doc(payload: dict[str, Any], *, file_id: str) -> dict[str, Any]:
    raw_annotations: list[dict[str, Any]] = []
    sentence_texts: list[str] = []

    sentences = payload.get("sentences")
    if not isinstance(sentences, list):
        raise ValueError("Input payload does not contain a valid 'sentences' list.")

    for fallback_index, sentence in enumerate(sentences, start=1):
        if not isinstance(sentence, dict):
            continue

        sentence_id = sentence.get("sentence_id")
        try:
            numeric_sentence_id = int(sentence_id)
        except (TypeError, ValueError):
            numeric_sentence_id = fallback_index

        text = str(sentence.get("text", "")).strip()
        if not text:
            continue

        entities = sentence.get("entities")
        if not isinstance(entities, list):
            entities = []

        raw_annotations.append(
            {
                "id": f"{file_id}_{numeric_sentence_id}",
                "start": sentence.get("start"),
                "end": sentence.get("end"),
                "text": text,
                "labels": [normalize_rr_role(sentence.get("rhetorical_role"))],
                "entities": [
                    build_openai_style_entity(entity)
                    for entity in entities
                    if isinstance(entity, dict)
                ],
            }
        )
        sentence_texts.append(text)

    if not raw_annotations:
        raise ValueError("Input payload did not contain any non-empty sentences.")

    preamble_end_char_offset = payload.get("preamble_end_char_offset")
    return {
        "id": file_id,
        "data": {
            "text": "\n".join(sentence_texts),
            "preamble_end_char_offset": preamble_end_char_offset,
        },
        "annotations": raw_annotations,
    }


def set_default_summary_flags(raw_doc: dict[str, Any]) -> None:
    for annotation in raw_doc.get("annotations", []):
        labels = annotation.get("labels") or []
        role = str(labels[0]) if labels else "NONE"
        if role == "PREAMBLE":
            annotation["in_summary"] = True
            annotation["summary_section"] = "PREAMBLE"
        else:
            annotation["in_summary"] = False
            annotation.pop("summary_section", None)
        annotation.pop("sent_score", None)


def has_summarizable_sentences(raw_doc: dict[str, Any]) -> bool:
    for annotation in raw_doc.get("annotations", []):
        labels = annotation.get("labels") or []
        role = str(labels[0]) if labels else "NONE"
        if role not in {"NONE", "PREAMBLE"}:
            return True
    return False


def summarize_raw_doc(
    *,
    raw_doc: dict[str, Any],
    summarizer: Any,
    file_id: str,
    logger: Any,
) -> tuple[str, str | None]:
    if not has_summarizable_sentences(raw_doc):
        set_default_summary_flags(raw_doc)
        raw_doc["summary"] = {}
        raw_doc["summaries"] = {}
        return "skipped_no_eligible_sentences", None

    try:
        summary_result = summarizer([raw_doc])
        summary_payload = summary_result[0] if summary_result else {}
        summary_mapping = summary_payload.get("summaries") or summary_payload.get("summary") or {}
        if not isinstance(summary_mapping, dict):
            summary_mapping = {}
        raw_doc["summary"] = summary_mapping
        raw_doc["summaries"] = summary_mapping
        return "ok", None
    except Exception as exc:  # pragma: no cover - runtime behavior
        logger.exception("OpenNyAI summarizer failed for %s", file_id)
        set_default_summary_flags(raw_doc)
        raw_doc["summary"] = {}
        raw_doc["summaries"] = {}
        raw_doc["summary_generation_error"] = str(exc)
        return "error", str(exc)


def build_pipeline_like_combined_payload(
    *,
    file_id: str,
    source_path: Path,
    chunk: str,
    raw_doc: dict[str, Any],
) -> dict[str, Any]:
    combined_payload = build_combined_output(
        file_id=file_id,
        internal_id=file_id,
        source_path=str(source_path),
        raw_doc=raw_doc,
    )
    if chunk:
        combined_payload["chunk"] = chunk
    return combined_payload


def persist_repo_style_outputs(
    *,
    file_id: str,
    raw_doc: dict[str, Any],
    layout: Any,
    combined_payload: dict[str, Any],
) -> None:
    output_paths = document_output_paths(layout, file_id)
    summary_output, summary_text = build_summary_output(file_id, raw_doc)

    write_json(output_paths["combined"], combined_payload)
    write_json(output_paths["annotations"], build_sentence_annotations_output(file_id, raw_doc))
    write_json(output_paths["ner"], build_ner_output(file_id, raw_doc))
    write_json(output_paths["rhetorical_roles"], build_rhetorical_roles_output(file_id, raw_doc))
    write_json(output_paths["summary_json"], summary_output)
    write_text(output_paths["summary_text"], summary_text)


def merge_summary_metadata_into_sentences(
    original_payload: dict[str, Any],
    raw_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    merged_sentences: list[dict[str, Any]] = []
    raw_annotations = raw_doc.get("annotations", [])
    annotations_by_index: dict[int, dict[str, Any]] = {}

    for annotation in raw_annotations:
        annotation_id = str(annotation.get("id", ""))
        try:
            sentence_index = int(annotation_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        annotations_by_index[sentence_index] = annotation

    for fallback_index, sentence in enumerate(original_payload.get("sentences", []), start=1):
        if not isinstance(sentence, dict):
            continue
        merged_sentence = copy.deepcopy(sentence)
        try:
            sentence_index = int(sentence.get("sentence_id"))
        except (TypeError, ValueError):
            sentence_index = fallback_index
        annotation = annotations_by_index.get(sentence_index)
        if annotation is not None:
            merged_sentence["in_summary"] = bool(annotation.get("in_summary", False))
            if annotation.get("summary_section") is not None:
                merged_sentence["summary_section"] = annotation.get("summary_section")
            if annotation.get("sent_score") is not None:
                merged_sentence["summary_sent_score"] = annotation.get("sent_score")
        merged_sentences.append(merged_sentence)
    return merged_sentences


def build_enriched_payload(
    *,
    original_payload: dict[str, Any],
    combined_payload: dict[str, Any],
    summary_status: str,
    summary_error: str | None,
) -> dict[str, Any]:
    enriched_payload = copy.deepcopy(original_payload)
    raw_result = combined_payload.get("raw_result", {})
    summary_mapping = raw_result.get("summary") or raw_result.get("summaries") or {}
    if not isinstance(summary_mapping, dict):
        summary_mapping = {}

    enriched_payload["sentences"] = merge_summary_metadata_into_sentences(
        original_payload,
        raw_result,
    )
    enriched_payload["opennyai_summary"] = summary_mapping
    enriched_payload["summary_status"] = summary_status
    enriched_payload["summary_error"] = summary_error

    if "case_outcome_label" in combined_payload:
        enriched_payload["case_outcome_label"] = combined_payload.get("case_outcome_label")
    if "case_outcome_score" in combined_payload:
        enriched_payload["case_outcome_score"] = combined_payload.get("case_outcome_score")
    if "llm_case_outcome" in combined_payload:
        enriched_payload["llm_case_outcome"] = combined_payload.get("llm_case_outcome")

    return enriched_payload


def flush_local_vllm_batch(
    *,
    pending_batch: list[dict[str, Any]],
    classifier: Any,
    args: argparse.Namespace,
    logger: Any,
) -> tuple[int, int]:
    if not pending_batch:
        return 0, 0

    skipped = 0
    errors = 0
    batch_inputs = [
        {
            "file_id": item["file_id"],
            "decision_text": item["extracted"]["decision_text"],
            "rpc_texts": item["extracted"]["rpc_texts"],
        }
        for item in pending_batch
    ]

    try:
        classifications = classifier.classify_batch(batch_inputs)
        if len(classifications) != len(pending_batch):
            raise RuntimeError(
                f"Expected {len(pending_batch)} vLLM outputs, but received {len(classifications)}."
            )

        for item, classification in zip(pending_batch, classifications):
            item["record"].update(classification)
            item["record"]["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=item["combined_payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            write_json(
                item["enriched_path"],
                build_enriched_payload(
                    original_payload=item["original_payload"],
                    combined_payload=item["combined_payload"],
                    summary_status=item["summary_status"],
                    summary_error=item["summary_error"],
                ),
            )
        return skipped, errors
    except Exception:  # pragma: no cover - runtime behavior
        logger.exception("Local vLLM batch failed. Retrying the cases one by one.")

    for item in pending_batch:
        try:
            classification = classifier.classify(
                file_id=item["file_id"],
                decision_text=item["extracted"]["decision_text"],
                rpc_texts=item["extracted"]["rpc_texts"],
                max_retries=args.max_retries,
                max_output_tokens=args.max_output_tokens,
            )
            item["record"].update(classification)
            item["record"]["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=item["combined_payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            write_json(
                item["enriched_path"],
                build_enriched_payload(
                    original_payload=item["original_payload"],
                    combined_payload=item["combined_payload"],
                    summary_status=item["summary_status"],
                    summary_error=item["summary_error"],
                ),
            )
        except Exception as exc:  # pragma: no cover - runtime behavior
            item["record"]["status"] = "error"
            item["record"]["error"] = str(exc)
            errors += 1
            write_json(
                item["enriched_path"],
                build_enriched_payload(
                    original_payload=item["original_payload"],
                    combined_payload=item["combined_payload"],
                    summary_status=item["summary_status"],
                    summary_error=item["summary_error"],
                ),
            )

    return skipped, errors


def resolve_backend(args: argparse.Namespace, logger: Any) -> str:
    if args.backend in {"local_vllm", "remote_hf"}:
        return args.backend

    try:
        import vllm  # noqa: F401

        logger.info("Resolved --backend auto to local_vllm.")
        return "local_vllm"
    except Exception:
        logger.info("Resolved --backend auto to remote_hf because vllm is unavailable in this environment.")
        return "remote_hf"


def main() -> int:
    load_project_environment(PROJECT_ROOT)
    load_dotenv()
    args = parse_args()

    if args.cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()

    configure_local_runtime(PROJECT_ROOT)

    annotations_dir = Path(args.annotations_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    enriched_dir = output_dir / "enriched_jsons"
    manifest_path = output_dir / "case_outcomes.jsonl"
    run_summary_path = output_dir / "run_summary.json"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory does not exist: {annotations_dir}")
    if not annotations_dir.is_dir():
        raise NotADirectoryError(f"Annotations path is not a directory: {annotations_dir}")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    layout = build_output_layout(output_dir)
    if args.write_repo_style_outputs:
        ensure_output_layout(layout)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        layout.logs.mkdir(parents=True, exist_ok=True)
    enriched_dir.mkdir(parents=True, exist_ok=True)

    logger, log_path = configure_logging(layout.logs, logger_name="opennyai_summary_label")
    logger.info("Annotations directory: %s", annotations_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("CUDA_VISIBLE_DEVICES=%s", os.getenv("CUDA_VISIBLE_DEVICES", "<all>"))

    summary_length = normalize_summary_length(args.summary_length, logger)
    validate_gpu_request(args.use_gpu, logger)

    if not args.dry_run and not args.hf_token:
        raise EnvironmentError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
        )

    patch_opennyai_summarizer_device_mismatch(logger)
    from opennyai.summarizer.ExtractiveSummarizer import ExtractiveSummarizer

    summarizer = ExtractiveSummarizer(
        use_gpu=args.use_gpu,
        verbose=args.verbose,
        summary_length=summary_length,
    )

    classifier = None
    if not args.dry_run:
        args.backend = resolve_backend(args, logger)
        classifier = build_classifier(args)

    paths = sorted(annotations_dir.glob("*.json"))
    if args.max_files is not None:
        paths = paths[: args.max_files]

    skipped_existing_labeled = 0
    summarized_files = 0
    summary_errors = 0
    label_skipped_missing = 0
    label_errors = 0
    pending_local_vllm_batch: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    from tqdm import tqdm

    for path in tqdm(paths, desc="Summarize + label"):
        try:
            input_payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            records.append(
                {
                    "file_id": path.stem,
                    "source_annotation_json": str(path),
                    "status": "error",
                    "error": f"Failed to load annotation JSON: {exc}",
                }
            )
            label_errors += 1
            continue

        file_id = str(input_payload.get("file_id") or path.stem).strip() or path.stem
        chunk = str(input_payload.get("chunk") or "").strip()
        enriched_path = enriched_dir / path.name

        if not args.overwrite and enriched_path.exists():
            skipped_existing_labeled += 1
            records.append(
                {
                    "file_id": file_id,
                    "source_annotation_json": str(path),
                    "enriched_json": str(enriched_path),
                    "status": "skipped_existing_labeled",
                }
            )
            continue

        summary_status = "ok"
        summary_error = None

        try:
            raw_doc = convert_annotation_payload_to_raw_doc(input_payload, file_id=file_id)
            summary_status, summary_error = summarize_raw_doc(
                raw_doc=raw_doc,
                summarizer=summarizer,
                file_id=file_id,
                logger=logger,
            )
            if summary_status == "error":
                summary_errors += 1

            combined_payload = build_pipeline_like_combined_payload(
                file_id=file_id,
                source_path=path,
                chunk=chunk,
                raw_doc=raw_doc,
            )
            if args.write_repo_style_outputs:
                persist_repo_style_outputs(
                    file_id=file_id,
                    raw_doc=raw_doc,
                    layout=layout,
                    combined_payload=combined_payload,
                )
            summarized_files += 1
        except Exception as exc:
            logger.exception("Failed preparing summary outputs for %s", file_id)
            records.append(
                {
                    "file_id": file_id,
                    "source_annotation_json": str(path),
                    "status": "error",
                    "error": str(exc),
                }
            )
            label_errors += 1
            continue

        extracted = collect_input_texts(combined_payload)
        record = {
            "file_id": file_id,
            "source_annotation_json": str(path),
            "enriched_json": str(enriched_path),
            "summary_status": summary_status,
            "summary_error": summary_error,
            "decision_text": extracted["decision_text"],
            "rpc_texts": extracted["rpc_texts"],
            "case_outcome_label": None,
            "case_outcome_score": None,
            "status": "pending",
        }
        records.append(record)

        if not extracted["decision_text"] and not extracted["rpc_texts"]:
            record["status"] = "skipped_missing_decision_and_rpc"
            label_skipped_missing += 1
            write_json(
                enriched_path,
                build_enriched_payload(
                    original_payload=input_payload,
                    combined_payload=combined_payload,
                    summary_status=summary_status,
                    summary_error=summary_error,
                ),
            )
            continue

        if args.dry_run:
            record["status"] = "dry_run_only"
            write_json(
                enriched_path,
                build_enriched_payload(
                    original_payload=input_payload,
                    combined_payload=combined_payload,
                    summary_status=summary_status,
                    summary_error=summary_error,
                ),
            )
            continue

        if args.backend == "local_vllm":
            pending_local_vllm_batch.append(
                {
                    "file_id": file_id,
                    "original_payload": input_payload,
                    "combined_payload": combined_payload,
                    "extracted": extracted,
                    "record": record,
                    "enriched_path": enriched_path,
                    "summary_status": summary_status,
                    "summary_error": summary_error,
                }
            )
            if len(pending_local_vllm_batch) >= args.generation_batch_size:
                batch_skipped, batch_errors = flush_local_vllm_batch(
                    pending_batch=pending_local_vllm_batch,
                    classifier=classifier,
                    args=args,
                    logger=logger,
                )
                label_skipped_missing += batch_skipped
                label_errors += batch_errors
                pending_local_vllm_batch = []
            continue

        try:
            classification = classifier.classify(
                file_id=file_id,
                decision_text=extracted["decision_text"],
                rpc_texts=extracted["rpc_texts"],
                max_retries=args.max_retries,
                max_output_tokens=args.max_output_tokens,
            )
            record.update(classification)
            record["status"] = "ok"
            apply_classification_to_payload(
                combined_payload=combined_payload,
                extracted=extracted,
                classification=classification,
                args=args,
            )
            write_json(
                enriched_path,
                build_enriched_payload(
                    original_payload=input_payload,
                    combined_payload=combined_payload,
                    summary_status=summary_status,
                    summary_error=summary_error,
                ),
            )
        except Exception as exc:  # pragma: no cover - runtime behavior
            record["status"] = "error"
            record["error"] = str(exc)
            label_errors += 1
            write_json(
                enriched_path,
                build_enriched_payload(
                    original_payload=input_payload,
                    combined_payload=combined_payload,
                    summary_status=summary_status,
                    summary_error=summary_error,
                ),
            )

        if args.backend == "remote_hf" and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    if pending_local_vllm_batch:
        batch_skipped, batch_errors = flush_local_vllm_batch(
            pending_batch=pending_local_vllm_batch,
            classifier=classifier,
            args=args,
            logger=logger,
        )
        label_skipped_missing += batch_skipped
        label_errors += batch_errors

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "annotations_dir": str(annotations_dir),
        "output_dir": str(output_dir),
        "enriched_dir": str(enriched_dir),
        "log_file": str(log_path),
        "backend": args.backend,
        "model_id": args.model_id,
        "provider": args.provider if args.backend == "remote_hf" else "local_gpu",
        "use_gpu": args.use_gpu,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "summary_length": summary_length,
        "write_repo_style_outputs": args.write_repo_style_outputs,
        "files_seen": len(paths),
        "files_summarized_this_run": summarized_files,
        "files_skipped_existing_labeled": skipped_existing_labeled,
        "summary_error_files": summary_errors,
        "successful_labels": sum(1 for item in records if item["status"] == "ok"),
        "dry_run_files": sum(1 for item in records if item["status"] == "dry_run_only"),
        "skipped_missing_decision_and_rpc": label_skipped_missing,
        "error_files": sum(1 for item in records if item["status"] == "error"),
        "label_counts": {
            label: sum(1 for item in records if item.get("case_outcome_label") == label)
            for label in LABEL_TO_SCORE
        },
        "manifest_path": str(manifest_path),
    }
    write_json(run_summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
