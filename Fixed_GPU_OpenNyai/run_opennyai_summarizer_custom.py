#!/usr/bin/env python3
"""Run standalone OpenNyAI summarization over existing annotation JSON files.

This script reads the sentence-level annotation JSONs produced by
`run_ner_rr_custom.py`, preserves the original NER + RR structure, and writes
one enriched JSON per document with:

- the original sentence-level payload
- per-sentence summary metadata (`in_summary`, `summary_section`, `summary_sent_score`)
- a top-level `opennyai_summary`

The worker model mirrors `run_ner_rr_custom.py`: one process per GPU, batched
document handling inside each worker, and resume behavior based on existing
output files.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import configure_local_runtime, load_project_environment
from src.pipeline_runner import patch_opennyai_summarizer_device_mismatch
from src.validators import normalize_summary_length, validate_gpu_request


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


def collect_annotation_files(input_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for json_path in sorted(input_dir.glob("*.json")):
        entries.append(
            {
                "file_id": json_path.stem,
                "abs_path": str(json_path),
                "filename": json_path.name,
            }
        )
    return entries


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

        try:
            numeric_sentence_id = int(sentence.get("sentence_id"))
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

    return {
        "id": file_id,
        "data": {
            "text": "\n".join(sentence_texts),
            "preamble_end_char_offset": payload.get("preamble_end_char_offset"),
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


def merge_summary_metadata_into_sentences(
    original_payload: dict[str, Any],
    raw_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    merged_sentences: list[dict[str, Any]] = []
    annotations_by_index: dict[int, dict[str, Any]] = {}

    for annotation in raw_doc.get("annotations", []):
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


def build_summary_enriched_payload(
    *,
    original_payload: dict[str, Any],
    raw_doc: dict[str, Any],
    summary_status: str,
    summary_error: str | None,
) -> dict[str, Any]:
    payload = copy.deepcopy(original_payload)
    summary_mapping = raw_doc.get("summary") or raw_doc.get("summaries") or {}
    if not isinstance(summary_mapping, dict):
        summary_mapping = {}

    payload["sentences"] = merge_summary_metadata_into_sentences(original_payload, raw_doc)
    payload["opennyai_summary"] = summary_mapping
    payload["summary_status"] = summary_status
    payload["summary_error"] = summary_error
    return payload


def build_summary_error_payload(
    *,
    original_payload: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(original_payload)
    payload.setdefault("sentences", [])
    payload["opennyai_summary"] = {}
    payload["summary_status"] = "error"
    payload["summary_error"] = error_message
    return payload


def _make_worker_logger(worker_id: int) -> logging.Logger:
    logger = logging.getLogger(f"opennyai_summarizer.worker{worker_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def _summarize_one(
    *,
    entry: dict[str, Any],
    summarizer: Any,
    logger: logging.Logger,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    input_path = Path(entry["abs_path"])
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            {
                "file_id": input_path.stem,
                "sentences": [],
                "opennyai_summary": {},
                "summary_status": "error",
                "summary_error": f"Failed to load annotation JSON: {exc}",
            },
            f"Failed to load annotation JSON: {exc}",
        )

    file_id = str(payload.get("file_id") or input_path.stem).strip() or input_path.stem
    try:
        raw_doc = convert_annotation_payload_to_raw_doc(payload, file_id=file_id)
    except Exception as exc:
        logger.warning("Skipping %s because its annotation payload is unusable: %s", file_id, exc)
        return (
            build_summary_error_payload(
                original_payload=payload,
                error_message=str(exc),
            ),
            str(exc),
        )

    if not has_summarizable_sentences(raw_doc):
        set_default_summary_flags(raw_doc)
        raw_doc["summary"] = {}
        raw_doc["summaries"] = {}
        return (
            build_summary_enriched_payload(
                original_payload=payload,
                raw_doc=raw_doc,
                summary_status="skipped_no_eligible_sentences",
                summary_error=None,
            ),
            None,
        )

    try:
        summary_result = summarizer([raw_doc])
        summary_payload = summary_result[0] if summary_result else {}
        summary_mapping = summary_payload.get("summaries") or summary_payload.get("summary") or {}
        if not isinstance(summary_mapping, dict):
            summary_mapping = {}
        raw_doc["summary"] = summary_mapping
        raw_doc["summaries"] = summary_mapping
        return (
            build_summary_enriched_payload(
                original_payload=payload,
                raw_doc=raw_doc,
                summary_status="ok",
                summary_error=None,
            ),
            None,
        )
    except Exception as exc:
        logger.exception("OpenNyAI summarizer failed for %s", file_id)
        set_default_summary_flags(raw_doc)
        raw_doc["summary"] = {}
        raw_doc["summaries"] = {}
        raw_doc["summary_generation_error"] = str(exc)
        return (
            build_summary_enriched_payload(
                original_payload=payload,
                raw_doc=raw_doc,
                summary_status="error",
                summary_error=str(exc),
            ),
            str(exc),
        )


def _summarize_batch(
    *,
    worker_id: int,
    summarizer: Any,
    batch_entries: List[dict[str, Any]],
    logger: logging.Logger,
) -> List[Tuple[dict[str, Any] | None, str | None]]:
    if not batch_entries:
        return []

    prepared: list[dict[str, Any]] = []
    immediate_results: dict[int, Tuple[dict[str, Any] | None, str | None]] = {}
    raw_docs: list[dict[str, Any]] = []

    for index, entry in enumerate(batch_entries):
        input_path = Path(entry["abs_path"])
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            immediate_results[index] = (
                {
                    "file_id": input_path.stem,
                    "sentences": [],
                    "opennyai_summary": {},
                    "summary_status": "error",
                    "summary_error": f"Failed to load annotation JSON: {exc}",
                },
                f"Failed to load annotation JSON: {exc}",
            )
            continue

        file_id = str(payload.get("file_id") or input_path.stem).strip() or input_path.stem
        try:
            raw_doc = convert_annotation_payload_to_raw_doc(payload, file_id=file_id)
        except Exception as exc:
            logger.warning("Skipping %s because its annotation payload is unusable: %s", file_id, exc)
            immediate_results[index] = (
                build_summary_error_payload(
                    original_payload=payload,
                    error_message=str(exc),
                ),
                str(exc),
            )
            continue

        prepared.append(
            {
                "original_index": index,
                "entry": entry,
                "payload": payload,
                "file_id": file_id,
                "raw_doc": raw_doc,
            }
        )

        if not has_summarizable_sentences(raw_doc):
            set_default_summary_flags(raw_doc)
            raw_doc["summary"] = {}
            raw_doc["summaries"] = {}
            immediate_results[index] = (
                build_summary_enriched_payload(
                    original_payload=payload,
                    raw_doc=raw_doc,
                    summary_status="skipped_no_eligible_sentences",
                    summary_error=None,
                ),
                None,
            )
        else:
            raw_docs.append(raw_doc)

    if raw_docs:
        try:
            summary_results = summarizer(raw_docs)
            if len(summary_results) != len(raw_docs):
                raise RuntimeError(
                    f"Expected {len(raw_docs)} summary results, received {len(summary_results)}."
                )

            summary_index = 0
            batch_results: List[Tuple[dict[str, Any] | None, str | None] | None] = [None] * len(batch_entries)
            for index, result in immediate_results.items():
                batch_results[index] = result

            for prepared_item in prepared:
                original_index = prepared_item["original_index"]
                existing = batch_results[original_index]
                if existing is not None:
                    continue

                summary_payload = summary_results[summary_index]
                summary_index += 1
                summary_mapping = summary_payload.get("summaries") or summary_payload.get("summary") or {}
                if not isinstance(summary_mapping, dict):
                    summary_mapping = {}

                raw_doc = prepared_item["raw_doc"]
                raw_doc["summary"] = summary_mapping
                raw_doc["summaries"] = summary_mapping
                batch_results[original_index] = (
                    build_summary_enriched_payload(
                        original_payload=prepared_item["payload"],
                        raw_doc=raw_doc,
                        summary_status="ok",
                        summary_error=None,
                    ),
                    None,
                )

            final_results: List[Tuple[dict[str, Any] | None, str | None]] = []
            for index, result in enumerate(batch_results):
                if result is None:
                    file_id = batch_entries[index].get("file_id", f"index_{index}")
                    fallback_error = f"Missing batch result for {file_id}"
                    final_results.append(
                        (
                            {
                                "file_id": file_id,
                                "sentences": [],
                                "opennyai_summary": {},
                                "summary_status": "error",
                                "summary_error": fallback_error,
                            },
                            fallback_error,
                        )
                    )
                else:
                    final_results.append(result)
            return final_results
        except Exception as exc:
            preview = ", ".join(item["file_id"] for item in prepared[:3])
            if len(prepared) > 3:
                preview += ", ..."
            print(
                f"[worker {worker_id}] batch failed for {len(prepared)} doc(s)"
                f" ({preview}): {exc}. Retrying individually.",
                file=sys.stderr,
                flush=True,
            )

    return [
        _summarize_one(entry=entry, summarizer=summarizer, logger=logger)
        for entry in batch_entries
    ]


def _maybe_report_progress(
    worker_id: int,
    processed: int,
    total: int,
    file_id: str,
    payload: Optional[Dict[str, Any]],
    elapsed_s: float,
    started_at: float,
) -> None:
    if processed != 1 and processed % 10 != 0 and processed != total:
        return

    elapsed_total = time.time() - started_at
    rate = processed / elapsed_total if elapsed_total > 0 else 0
    eta_s = (total - processed) / rate if rate > 0 else 0

    if payload is None:
        status = "failed"
        summary_sections = 0
        sent_count = 0
    else:
        status = str(payload.get("summary_status", "ok"))
        summary_sections = len(payload.get("opennyai_summary", {}))
        sent_count = len(payload.get("sentences", []))

    print(
        f"[worker {worker_id}] [{processed}/{total}] {file_id}"
        f"  sents={sent_count}"
        f"  sections={summary_sections}"
        f"  status={status}"
        f"  {elapsed_s:.1f}s/doc  ETA {eta_s/60:.1f}m",
        flush=True,
    )


def _worker(
    worker_id: int,
    gpu_id: int,
    file_entries: List[Dict[str, Any]],
    output_dir: str,
    use_gpu: bool,
    pipeline_batch_size: int,
    summary_length: float,
) -> Tuple[int, int]:
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    warnings.filterwarnings("ignore")
    logger = _make_worker_logger(worker_id)

    from opennyai.summarizer.ExtractiveSummarizer import ExtractiveSummarizer

    patch_opennyai_summarizer_device_mismatch(logger)
    print(f"[worker {worker_id}] GPU {gpu_id} — loading summarizer ({len(file_entries)} docs) ...", flush=True)
    summarizer = ExtractiveSummarizer(
        use_gpu=use_gpu,
        verbose=False,
        summary_length=summary_length,
    )
    print(f"[worker {worker_id}] GPU {gpu_id} — model ready, starting ...", flush=True)

    enriched_dir = Path(output_dir) / "enriched_jsons"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    errors = 0
    t_start = time.time()
    total = len(file_entries)
    processed = 0
    pending_batch: List[Dict[str, Any]] = []

    def flush_pending_batch() -> None:
        nonlocal done, errors, processed, pending_batch
        if not pending_batch:
            return

        batch_started = time.time()
        results = _summarize_batch(
            worker_id=worker_id,
            summarizer=summarizer,
            batch_entries=pending_batch,
            logger=logger,
        )
        per_doc_elapsed = (time.time() - batch_started) / max(len(pending_batch), 1)

        for entry, result in zip(pending_batch, results):
            processed += 1
            payload, error = result
            file_id = entry["file_id"]
            out_path = enriched_dir / entry["filename"]

            if payload is None:
                errors += 1
                _maybe_report_progress(worker_id, processed, total, file_id, None, per_doc_elapsed, t_start)
                continue

            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            if error:
                errors += 1
            else:
                done += 1
            _maybe_report_progress(worker_id, processed, total, file_id, payload, per_doc_elapsed, t_start)

        pending_batch = []

    for entry in file_entries:
        out_path = enriched_dir / entry["filename"]
        if out_path.exists():
            done += 1
            processed += 1
            continue

        pending_batch.append(entry)
        if len(pending_batch) >= pipeline_batch_size:
            flush_pending_batch()

    flush_pending_batch()
    print(f"[worker {worker_id}] done — {done} written, {errors} errors.", flush=True)
    return done, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone OpenNyAI summarizer over annotation JSON files."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing annotation JSONs.")
    parser.add_argument("--output_dir", required=True, help="Directory to write summary-enriched JSONs.")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--max_docs", type=int, default=0, help="Limit total number of documents (0 = all).")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel GPU workers (0 = all available GPUs, 1 = single process).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU IDs to use, e.g. '0,1,4,5'. Overrides --workers.",
    )
    parser.add_argument(
        "--pipeline_batch_size",
        "--pipeline-batch-size",
        type=int,
        default=1,
        help="Number of documents each worker sends through the summarizer in one call.",
    )
    parser.add_argument(
        "--summary_length",
        "--summary-length",
        type=float,
        default=0.0,
        help="Summary length fraction in the 0.0-1.0 range. Use 0.0 for adaptive mode.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output_dir first and recompute every file.")
    return parser.parse_args()


def _available_gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _aggregate_summary(output_root: Path) -> dict[str, Any]:
    enriched_dir = output_root / "enriched_jsons"
    total_sentences = 0
    total_summary_sentences = 0
    summary_status_counts: Counter[str] = Counter()
    summary_section_counts: Counter[str] = Counter()
    ner_counts: Counter[str] = Counter()
    rr_counts: Counter[str] = Counter()
    doc_count = 0

    for json_file in enriched_dir.glob("*.json"):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        doc_count += 1
        summary_status_counts.update([str(payload.get("summary_status", "unknown"))])
        ner_counts.update(payload.get("ner_by_label", {}))
        rr_counts.update(payload.get("rr_by_role", {}))

        for sentence in payload.get("sentences", []):
            total_sentences += 1
            if sentence.get("in_summary"):
                total_summary_sentences += 1

        summary_mapping = payload.get("opennyai_summary", {})
        if isinstance(summary_mapping, dict):
            summary_section_counts.update(summary_mapping.keys())

    return {
        "documents_written": doc_count,
        "total_sentences": total_sentences,
        "total_summary_sentences": total_summary_sentences,
        "summary_status_counts": dict(summary_status_counts),
        "summary_section_counts": dict(summary_section_counts),
        "ner_by_label": dict(ner_counts.most_common()),
        "rr_by_role": dict(rr_counts.most_common()),
    }


def prime_summarizer_assets(summary_length: float) -> None:
    from opennyai.summarizer.ExtractiveSummarizer import ExtractiveSummarizer

    summarizer = ExtractiveSummarizer(
        use_gpu=False,
        verbose=False,
        summary_length=summary_length,
    )
    del summarizer


def main() -> None:
    load_project_environment(PROJECT_ROOT)
    configure_local_runtime(PROJECT_ROOT)

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if args.pipeline_batch_size <= 0:
        print("ERROR: --pipeline_batch_size must be a positive integer.")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"ERROR: --input_dir does not exist: {input_dir}")
        sys.exit(1)

    if args.overwrite and output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_dir = output_dir / "enriched_jsons"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    logger = logging.getLogger("opennyai_summarizer.main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(logging.StreamHandler(sys.stdout))

    summary_length = normalize_summary_length(args.summary_length, logger)
    validate_gpu_request(args.use_gpu, logger)

    all_files = collect_annotation_files(input_dir)
    print(f"Found {len(all_files):,} annotation JSON file(s) under: {input_dir}")

    todo = [entry for entry in all_files if not (enriched_dir / entry["filename"]).exists()]
    already_done = len(all_files) - len(todo)
    if already_done:
        print(f"Resuming — skipping {already_done:,} already-processed files.")

    if args.max_docs and args.max_docs < len(todo):
        todo = todo[: args.max_docs]
        print(f"Limiting to {args.max_docs} documents (--max_docs).")

    if not todo:
        print("Nothing left to process.")
        return

    print("Priming OpenNyAI summarizer assets ...")
    prime_summarizer_assets(summary_length)

    if args.use_gpu:
        n_gpus = _available_gpu_count()
        if args.gpus:
            gpu_ids = [int(token.strip()) for token in args.gpus.split(",") if token.strip()]
            invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= n_gpus]
            if invalid:
                print(f"ERROR: GPU IDs {invalid} are not available (only {n_gpus} GPUs found).")
                sys.exit(1)
        else:
            gpu_ids = list(range(n_gpus))

        n_workers = min(args.workers if args.workers > 0 else len(gpu_ids), len(todo), len(gpu_ids))
        gpu_ids = gpu_ids[:n_workers]

        if n_workers == 0:
            print("WARNING: --use_gpu set but no GPUs found, falling back to CPU.")
            n_workers = 1
            gpu_ids = [0]
            args.use_gpu = False
    else:
        n_workers = 1
        gpu_ids = [0]

    device_str = f"GPU(s) {gpu_ids}" if args.use_gpu else "CPU"
    print(
        f"Processing {len(todo):,} documents with {n_workers} worker(s) on {device_str}"
        f" (pipeline_batch_size={args.pipeline_batch_size}, summary_length={summary_length}) ...\n"
    )

    chunks = [[] for _ in range(n_workers)]
    for index, entry in enumerate(todo):
        chunks[index % n_workers].append(entry)

    wall_start = time.time()

    if n_workers == 1:
        done, errors = _worker(
            0,
            gpu_ids[0],
            chunks[0],
            str(output_dir),
            args.use_gpu,
            args.pipeline_batch_size,
            summary_length,
        )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.starmap(
                _worker,
                [
                    (
                        worker_id,
                        gpu_ids[worker_id],
                        chunks[worker_id],
                        str(output_dir),
                        args.use_gpu,
                        args.pipeline_batch_size,
                        summary_length,
                    )
                    for worker_id in range(n_workers)
                ],
            )
        done = sum(result[0] for result in results)
        errors = sum(result[1] for result in results)

    wall_elapsed = time.time() - wall_start
    aggregate = _aggregate_summary(output_dir)
    summary = {
        "input_dir": str(input_dir),
        "enriched_dir": str(enriched_dir),
        "total_files_found": len(all_files),
        "total_processed": aggregate["documents_written"],
        "total_errors": errors,
        "num_workers": n_workers,
        "use_gpu": args.use_gpu,
        "pipeline_batch_size": args.pipeline_batch_size,
        "summary_length": summary_length,
        "wall_time_s": round(wall_elapsed, 1),
        **aggregate,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  Done in {wall_elapsed/60:.1f}m  ({n_workers} worker(s))")
    print(f"  {done:,} docs written  ({errors} errors)")
    print(f"  Total sentences        : {aggregate['total_sentences']:,}")
    print(f"  Total summary sentences: {aggregate['total_summary_sentences']:,}")
    print("\n  Summary status counts:")
    for status, count in Counter(aggregate["summary_status_counts"]).most_common():
        print(f"    {status:<28} {count:>8,}")
    print("\n  Summary section counts:")
    for section, count in Counter(aggregate["summary_section_counts"]).most_common():
        print(f"    {section:<28} {count:>8,}")
    print(f"\n  Enriched JSONs: {enriched_dir}/")
    print(f"  Summary      : {summary_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
