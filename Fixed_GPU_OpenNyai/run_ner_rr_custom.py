"""
run_ner_rr_custom.py
====================
Run OpenNyAI NER + Rhetorical Role (RR) on every .txt file under a given
input directory and write one JSON file per document.

Supports multi-GPU parallel processing: one worker process per GPU, each
loads its own pipeline and processes an independent slice of the file list.
Each worker can also batch multiple documents into a single OpenNyAI call.

Output layout:
  <output_dir>/
    annotations/
      Allahabad_HC_1901_4.json
      Allahabad_HC_1918_122.json
      ...
    summary.json

Per-document JSON format:
{
  "file_id":   "Allahabad_HC_1901_4",
  "chunk":     "chunk_0001",
  "rr_available": true,
  "sentences": [
    {
      "sentence_id": 1,
      "text": "...",
      "rhetorical_role": "PREAMBLE",
      "start": 0,
      "end": 95,
      "entities": [
        {"text": "Supreme Court", "label": "COURT", "start": 5, "end": 18}
      ]
    },
    ...
  ],
  "ner_by_label": {"COURT": 3, "JUDGE": 5, ...},
  "rr_by_role":  {"PREAMBLE": 4, "FAC": 12, ...}
}

Resume support:
  Already-written .json files are skipped — safe to interrupt and re-run.

Usage:
  bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh                          # all GPUs
  bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --workers 4             # 4 GPUs
  bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --pipeline_batch_size 4 # 4 docs per call
  bash Fixed_GPU_OpenNyai/run_ner_rr_custom.sh --quick                 # 20 docs sanity check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import sys
import time
import unicodedata
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Cache redirect ────────────────────────────────────────────────────────────
# Set before any GPU imports so every worker process inherits correct paths.
PROJECT_ROOT = Path(__file__).resolve().parent
_cache   = PROJECT_ROOT / ".cache"
_runtime = PROJECT_ROOT / ".runtime_home"
os.environ.setdefault("HOME",                  str(_runtime))
os.environ.setdefault("HF_HOME",               str(_cache / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_cache / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE",    str(_cache / "huggingface" / "transformers"))
os.environ.setdefault("TORCH_HOME",            str(_cache / "torch"))


# ── Collect input files ───────────────────────────────────────────────────────

def collect_txt_files(input_dir: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for txt in sorted(input_dir.rglob("*.txt")):
        rel   = txt.relative_to(input_dir)
        parts = rel.parts
        entries.append({
            "file_id":  txt.stem,
            "chunk":    parts[0] if len(parts) > 1 else "",
            "abs_path": str(txt),          # str so it's picklable across processes
            "rel_path": str(rel),
            "pipeline_file_id": make_pipeline_file_id(str(rel)),
        })
    return entries


def make_pipeline_file_id(rel_path: str) -> str:
    """Build a stable ASCII-only identifier for OpenNyAI internals."""
    normalized = unicodedata.normalize("NFKD", rel_path)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_only).strip("._-")
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    prefix = slug[:80] if slug else "doc"
    return f"{prefix}_{digest}"


def load_deferred_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}

    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}

    if not isinstance(raw, list):
        return {}

    records: Dict[str, Dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        file_id = str(item.get("file_id", "")).strip()
        if not file_id:
            continue
        records[file_id] = item
    return records


def save_deferred_records(path: Path, records: Dict[str, Dict[str, Any]]) -> None:
    ordered = [records[file_id] for file_id in sorted(records)]
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False))


def _annotation_path(ann_dir: Path, file_id: str) -> Path:
    return ann_dir / f"{file_id}.json"


def _remaining_entries(
    file_entries: List[Dict[str, Any]],
    ann_dir: Path,
    deferred_ids: set[str],
) -> List[Dict[str, Any]]:
    remaining: List[Dict[str, Any]] = []
    for entry in file_entries:
        if entry["file_id"] in deferred_ids:
            continue
        if _annotation_path(ann_dir, entry["file_id"]).exists():
            continue
        remaining.append(entry)
    return remaining


def _terminate_process(proc: Optional[mp.Process], worker_id: int) -> None:
    if proc is None:
        return
    if not proc.is_alive():
        proc.join(timeout=1)
        return

    proc.terminate()
    proc.join(timeout=10)
    if proc.is_alive():
        print(f"[supervisor] worker {worker_id} did not stop on terminate(); killing.", flush=True)
        proc.kill()
        proc.join(timeout=5)


# ── Pipeline helpers ──────────────────────────────────────────────────────────

RR_LABELS = {
    "PREAMBLE", "FAC", "ISSUE", "ARG_PETITIONER", "ARG_RESPONDENT",
    "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO", "NONE",
}


def _parse_combined_output(combined: list) -> Dict[str, Any]:
    raw             = combined[0]
    preamble_offset = raw.get("data", {}).get("preamble_end_char_offset")
    annotations     = raw.get("annotations", [])

    sentences: List[Dict[str, Any]] = []
    ner_label_counter: Counter = Counter()
    rr_role_counter:   Counter = Counter()

    for idx, ann in enumerate(annotations):
        labels = ann.get("labels", [])
        role   = str(labels[0]).upper() if labels else "NONE"
        if role not in RR_LABELS:
            role = "NONE"
        rr_role_counter[role] += 1

        entities: List[Dict[str, Any]] = []
        for ent in ann.get("entities", []):
            ent_labels = ent.get("labels", [])
            label      = str(ent_labels[0]) if ent_labels else "UNKNOWN"
            ner_label_counter[label] += 1
            entities.append({
                "text":  ent.get("text", "").strip(),
                "label": label,
                "start": ent.get("start"),
                "end":   ent.get("end"),
            })

        sentences.append({
            "sentence_id":     idx + 1,
            "text":            ann.get("text", "").strip(),
            "rhetorical_role": role,
            "start":           ann.get("start"),
            "end":             ann.get("end"),
            "entities":        entities,
        })

    return {
        "rr_available":             True,
        "preamble_end_char_offset": preamble_offset,
        "sentences":                sentences,
        "ner_by_label":             dict(ner_label_counter.most_common()),
        "rr_by_role":               dict(rr_role_counter.most_common()),
    }


def _process_one(
    Data,
    pipeline,
    ner_nlp,
    file_id: str,
    pipeline_file_id: str,
    text: str,
    use_gpu: bool,
) -> Optional[Dict]:
    """Run NER+RR on a single document. Falls back to NER-only on E030."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            data_obj = Data(
                input_text=[text],
                preprocessing_nlp_model="en_core_web_trf",
                use_gpu=use_gpu,
                verbose=False,
                file_ids=[pipeline_file_id],
            )
            return _parse_combined_output(pipeline([data_obj[0]]))
        except Exception as exc:
            if "E030" not in str(exc) and "Sentence boundaries" not in str(exc):
                print(f"    [W{os.getpid()}] ERROR {file_id}: {exc}", file=sys.stderr, flush=True)
                return None

        # NER-only fallback — en_core_web_trf failed to set sentence boundaries
        try:
            doc               = ner_nlp(text)
            ner_label_counter: Counter = Counter()
            entities: List[Dict] = []
            for ent in doc.ents:
                ner_label_counter[ent.label_] += 1
                entities.append({"text": ent.text.strip(), "label": ent.label_,
                                  "start": ent.start_char, "end": ent.end_char})
            return {
                "rr_available":             False,
                "rr_skip_reason":           "E030 sentence boundaries unset in RR preprocessor",
                "preamble_end_char_offset": None,
                "sentences":                [],
                "entities":                 entities,
                "ner_by_label":             dict(ner_label_counter.most_common()),
                "rr_by_role":               {},
            }
        except Exception as exc2:
            print(f"    [W{os.getpid()}] NER-only also failed {file_id}: {exc2}",
                  file=sys.stderr, flush=True)
            return None


def _process_batch(
    worker_id: int,
    Data,
    pipeline,
    ner_nlp,
    batch_entries: List[Dict[str, Any]],
    use_gpu: bool,
) -> List[Optional[Dict]]:
    """Run NER+RR on a batch of documents, then fall back to per-doc retries if needed."""
    if not batch_entries:
        return []

    if len(batch_entries) == 1:
        item = batch_entries[0]
        return [
            _process_one(
                Data,
                pipeline,
                ner_nlp,
                item["file_id"],
                item["pipeline_file_id"],
                item["text"],
                use_gpu,
            )
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            data_obj = Data(
                input_text=[item["text"] for item in batch_entries],
                preprocessing_nlp_model="en_core_web_trf",
                use_gpu=use_gpu,
                verbose=False,
                file_ids=[item["pipeline_file_id"] for item in batch_entries],
            )
            pipeline_result = pipeline([data_obj[idx] for idx in range(len(batch_entries))])
            raw_results = pipeline_result if isinstance(pipeline_result, list) else [pipeline_result]
            if len(raw_results) != len(batch_entries):
                raise RuntimeError(
                    f"pipeline returned {len(raw_results)} result(s) for {len(batch_entries)} input document(s)"
                )
            return [_parse_combined_output([raw_result]) for raw_result in raw_results]
        except Exception as exc:
            preview = ", ".join(item["file_id"] for item in batch_entries[:3])
            if len(batch_entries) > 3:
                preview += ", ..."
            print(
                f"[worker {worker_id}] batch failed for {len(batch_entries)} doc(s)"
                f" ({preview}): {exc}. Retrying individually.",
                file=sys.stderr,
                flush=True,
            )

    return [
        _process_one(
            Data,
            pipeline,
            ner_nlp,
            item["file_id"],
            item["pipeline_file_id"],
            item["text"],
            use_gpu,
        )
        for item in batch_entries
    ]


def _maybe_report_progress(
    worker_id: int,
    processed: int,
    total: int,
    file_id: str,
    result: Optional[Dict],
    elapsed_s: float,
    started_at: float,
) -> None:
    if processed != 1 and processed % 10 != 0 and processed != total:
        return

    elapsed_total = time.time() - started_at
    rate = processed / elapsed_total if elapsed_total > 0 else 0
    eta_s = (total - processed) / rate if rate > 0 else 0

    if result is None:
        rr_flag = " (failed)"
        sent_count = 0
        ent_count = 0
    else:
        rr_flag = "" if result.get("rr_available", True) else " (NER-only)"
        sent_count = len(result.get("sentences", []))
        ent_count = sum(result.get("ner_by_label", {}).values())

    print(
        f"[worker {worker_id}] [{processed}/{total}] {file_id}"
        f"  sents={sent_count}"
        f"  ents={ent_count}"
        f"{rr_flag}  {elapsed_s:.1f}s/doc  ETA {eta_s/60:.1f}m",
        flush=True,
    )


# ── Worker process ────────────────────────────────────────────────────────────

def _worker(
    worker_id:   int,
    gpu_id:      int,
    file_entries: List[Dict[str, Any]],
    ann_dir:     str,
    use_gpu:     bool,
    pipeline_batch_size: int,
    total_docs: int,
    processed_offset: int,
    status_queue,
) -> None:
    """
    Runs in a separate process. Owns one GPU (set via CUDA_VISIBLE_DEVICES).
    """
    def emit(event_type: str, **payload: Any) -> None:
        if status_queue is None:
            return
        try:
            status_queue.put({"type": event_type, "worker_id": worker_id, **payload})
        except Exception:
            pass

    # ── Must be set before any CUDA/torch or opennyai import ─────────────
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Give each worker its own HOME so opennyai's temp files don't collide.
    # (~/.opennyai/temp_hsln/ is written on first run and causes a race otherwise.)
    worker_home = Path(ann_dir).parent / f".worker_home_{worker_id}"
    worker_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(worker_home)

    # Mirror the cache redirects from the main process (setdefault won't
    # override HOME we just set, but sets the rest correctly).
    _proj = Path(__file__).resolve().parent
    _c    = _proj / ".cache"
    os.environ.setdefault("HF_HOME",               str(_c / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_c / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE",    str(_c / "huggingface" / "transformers"))
    os.environ.setdefault("TORCH_HOME",            str(_c / "torch"))

    # Silence noisy library warnings inside workers
    warnings.filterwarnings("ignore")

    stale_temp_hsln = worker_home / ".opennyai" / "temp_hsln"
    if stale_temp_hsln.exists():
        shutil.rmtree(stale_temp_hsln, ignore_errors=True)

    emit("boot", pid=os.getpid(), assigned_docs=len(file_entries))

    try:
        from opennyai import Pipeline
        from opennyai.utils.preprocess import Data
        import spacy
        import torch
    except ImportError:
        print(f"[worker {worker_id}] ERROR: opennyai/spacy not found.", flush=True)
        emit("worker_crashed", error="opennyai/spacy import failed")
        return

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    print(f"[worker {worker_id}] GPU {gpu_id} — loading pipeline ({len(file_entries)} docs) ...",
          flush=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = Pipeline(
            components=["NER", "Rhetorical_Role"],
            use_gpu=use_gpu,
            verbose=False,
        )

    if use_gpu:
        spacy.prefer_gpu()
    ner_nlp = spacy.load("en_legal_ner_trf")

    print(f"[worker {worker_id}] GPU {gpu_id} — models ready, starting ...", flush=True)
    emit("ready", pid=os.getpid(), assigned_docs=len(file_entries))

    ann_path  = Path(ann_dir)
    done      = 0
    errors    = 0
    t_start   = time.time()
    processed_local = 0
    pending_batch: List[Dict[str, Any]] = []

    def flush_pending_batch() -> None:
        nonlocal pending_batch, done, errors, processed_local
        if not pending_batch:
            return

        emit(
            "batch_start",
            items=[
                {
                    "index": item["worker_index"],
                    "file_id": item["file_id"],
                    "rel_path": item["rel_path"],
                    "chunk": item["chunk"],
                }
                for item in pending_batch
            ],
        )

        batch_started = time.time()
        results = _process_batch(worker_id, Data, pipeline, ner_nlp, pending_batch, use_gpu)
        per_doc_elapsed = (time.time() - batch_started) / max(len(pending_batch), 1)

        for item, result in zip(pending_batch, results):
            processed_local += 1
            processed_global = processed_offset + processed_local
            if result is None:
                errors += 1
                _maybe_report_progress(
                    worker_id,
                    processed_global,
                    total_docs,
                    item["file_id"],
                    None,
                    per_doc_elapsed,
                    t_start,
                )
                emit(
                    "doc_done",
                    index=item["worker_index"],
                    file_id=item["file_id"],
                    status="failed",
                )
                continue

            out_file = ann_path / f"{item['file_id']}.json"
            record = {"file_id": item["file_id"], "chunk": item["chunk"], **result}
            out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            done += 1
            _maybe_report_progress(
                worker_id,
                processed_global,
                total_docs,
                item["file_id"],
                result,
                per_doc_elapsed,
                t_start,
            )
            emit(
                "doc_done",
                index=item["worker_index"],
                file_id=item["file_id"],
                status="written",
            )

        pending_batch = []

    for entry in file_entries:
        out_file = ann_path / f"{entry['file_id']}.json"
        if out_file.exists():
            done += 1
            processed_local += 1
            emit(
                "doc_done",
                index=entry["worker_index"],
                file_id=entry["file_id"],
                status="skipped_existing",
            )
            continue

        try:
            text = Path(entry["abs_path"]).read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            print(f"[worker {worker_id}] SKIP (read error) {entry['rel_path']}: {exc}", flush=True)
            errors += 1
            processed_local += 1
            emit(
                "doc_done",
                index=entry["worker_index"],
                file_id=entry["file_id"],
                status="read_error",
            )
            continue

        if not text:
            errors += 1
            processed_local += 1
            emit(
                "doc_done",
                index=entry["worker_index"],
                file_id=entry["file_id"],
                status="empty",
            )
            continue

        pending_batch.append({
            "file_id": entry["file_id"],
            "chunk": entry["chunk"],
            "pipeline_file_id": entry["pipeline_file_id"],
            "text": text,
            "rel_path": entry["rel_path"],
            "worker_index": entry["worker_index"],
        })
        if len(pending_batch) >= pipeline_batch_size:
            flush_pending_batch()

    flush_pending_batch()

    print(f"[worker {worker_id}] done — {done} written, {errors} errors.", flush=True)
    emit("worker_done", done=done, errors=errors)
    return


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OpenNyAI NER + RR on custom .txt files; one JSON per document."
    )
    p.add_argument("--input_dir",  required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--use_gpu",    action="store_true")
    p.add_argument("--max_docs",   type=int, default=0,
                   help="Limit total number of documents (0 = all).")
    p.add_argument("--workers",    type=int, default=0,
                   help="Number of parallel GPU workers (0 = all available GPUs, 1 = single process).")
    p.add_argument("--gpus",       type=str, default="",
                   help="Comma-separated GPU IDs to use, e.g. '0,1,4,5'. "
                        "Overrides --workers. Default: all available GPUs.")
    p.add_argument(
        "--pipeline_batch_size",
        "--pipeline-batch-size",
        type=int,
        default=1,
        help="Number of documents each worker sends through OpenNyAI in one pipeline call.",
    )
    p.add_argument(
        "--freeze_timeout_s",
        "--freeze-timeout-s",
        type=int,
        default=900,
        help="Defer the current document/batch if a worker stays on it longer than this many seconds (0 = disable).",
    )
    p.add_argument(
        "--retry_deferred",
        action="store_true",
        help="Retry documents listed in deferred_docs.json instead of skipping them on resume.",
    )
    return p.parse_args()


def _available_gpu_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def main() -> None:
    args    = parse_args()
    in_dir  = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if args.pipeline_batch_size <= 0:
        print("ERROR: --pipeline_batch_size must be a positive integer.")
        sys.exit(1)

    if not in_dir.is_dir():
        print(f"ERROR: --input_dir does not exist: {in_dir}")
        sys.exit(1)

    ann_dir      = out_dir / "annotations"
    summary_path = out_dir / "summary.json"
    deferred_path = out_dir / "deferred_docs.json"
    ann_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect & filter files ─────────────────────────────────────────────
    all_files = collect_txt_files(in_dir)
    print(f"Found {len(all_files):,} .txt files under: {in_dir}")

    deferred_records = load_deferred_records(deferred_path)
    deferred_skip_ids = set() if args.retry_deferred else set(deferred_records)

    todo = [
        f for f in all_files
        if not _annotation_path(ann_dir, f["file_id"]).exists() and f["file_id"] not in deferred_skip_ids
    ]
    already_done = sum(1 for f in all_files if _annotation_path(ann_dir, f["file_id"]).exists())
    if already_done:
        print(f"Resuming — skipping {already_done:,} already-processed files.")
    if deferred_skip_ids:
        action = "Retrying" if args.retry_deferred else "Skipping"
        print(f"{action} {len(deferred_skip_ids):,} deferred document(s) from: {deferred_path}")

    if args.max_docs and args.max_docs < len(todo):
        todo = todo[: args.max_docs]
        print(f"Limiting to {args.max_docs} documents (--max_docs).")

    if not todo:
        print("Nothing left to process.")
        return

    # ── Determine which GPUs / how many workers ────────────────────────────
    if args.use_gpu:
        n_gpus = _available_gpu_count()
        if args.gpus:
            gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
            invalid = [g for g in gpu_ids if g >= n_gpus]
            if invalid:
                print(f"ERROR: GPU IDs {invalid} are not available (only {n_gpus} GPUs found).")
                sys.exit(1)
        else:
            gpu_ids = list(range(n_gpus))

        n_workers = min(args.workers if args.workers > 0 else len(gpu_ids), len(todo), len(gpu_ids))
        gpu_ids   = gpu_ids[:n_workers]

        if n_workers == 0:
            print("WARNING: --use_gpu set but no GPUs found, falling back to CPU.")
            n_workers = 1
            gpu_ids   = [0]
            args.use_gpu = False
    else:
        n_workers = 1
        gpu_ids   = [0]

    gpu_str = f"GPU(s) {gpu_ids}" if args.use_gpu else "CPU"
    print(
        f"Processing {len(todo):,} documents with {n_workers} worker(s) on {gpu_str}"
        f" (pipeline_batch_size={args.pipeline_batch_size}) ...\n"
    )

    # ── Split work across workers ──────────────────────────────────────────
    chunks = [[] for _ in range(n_workers)]
    for i, entry in enumerate(todo):
        worker_id = i % n_workers
        chunks[worker_id].append({**entry, "worker_index": len(chunks[worker_id])})

    wall_start = time.time()
    total_errors = 0
    deferred_this_run = 0

    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    worker_states: Dict[int, Dict[str, Any]] = {
        wid: {
            "worker_id": wid,
            "gpu_id": gpu_ids[wid],
            "entries": chunks[wid],
            "process": None,
            "current_batch": None,
            "current_started_at": None,
            "started_at": None,
            "ready": False,
        }
        for wid in range(n_workers)
    }

    def spawn_worker(state: Dict[str, Any]) -> None:
        remaining = _remaining_entries(state["entries"], ann_dir, set(deferred_records))
        if not remaining:
            state["process"] = None
            state["current_batch"] = None
            state["current_started_at"] = None
            state["ready"] = True
            return

        processed_offset = len(state["entries"]) - len(remaining)
        proc = ctx.Process(
            target=_worker,
            args=(
                state["worker_id"],
                state["gpu_id"],
                remaining,
                str(ann_dir),
                args.use_gpu,
                args.pipeline_batch_size,
                len(state["entries"]),
                processed_offset,
                status_queue,
            ),
        )
        proc.start()
        state["process"] = proc
        state["current_batch"] = None
        state["current_started_at"] = None
        state["started_at"] = time.time()
        state["ready"] = False
        print(
            f"[supervisor] worker {state['worker_id']} on GPU {state['gpu_id']} "
            f"starting with {len(remaining)} remaining doc(s).",
            flush=True,
        )

    def note_deferred(state: Dict[str, Any], item: Dict[str, Any], reason: str) -> None:
        nonlocal deferred_this_run, total_errors
        file_id = item["file_id"]
        if file_id in deferred_records:
            return

        deferred_records[file_id] = {
            "file_id": file_id,
            "chunk": item.get("chunk", ""),
            "rel_path": item.get("rel_path", ""),
            "worker_id": state["worker_id"],
            "gpu_id": state["gpu_id"],
            "reason": reason,
            "deferred_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        save_deferred_records(deferred_path, deferred_records)
        deferred_this_run += 1
        total_errors += 1

    interrupted = False
    try:
        for state in worker_states.values():
            spawn_worker(state)

        while any(state["process"] is not None for state in worker_states.values()):
            try:
                event = status_queue.get(timeout=1.0)
            except queue.Empty:
                event = None

            if event is not None:
                state = worker_states[event["worker_id"]]
                event_type = event.get("type")

                if event_type == "ready":
                    state["ready"] = True
                elif event_type == "batch_start":
                    state["current_batch"] = event.get("items", [])
                    state["current_started_at"] = time.time()
                elif event_type == "doc_done":
                    status = event.get("status")
                    if status in {"failed", "read_error", "empty"}:
                        total_errors += 1
                    current_batch = state.get("current_batch") or []
                    state["current_batch"] = [
                        item for item in current_batch if item["index"] != event.get("index")
                    ]
                    if not state["current_batch"]:
                        state["current_batch"] = None
                        state["current_started_at"] = None
                elif event_type == "worker_done":
                    state["current_batch"] = None
                    state["current_started_at"] = None

            now = time.time()
            for state in worker_states.values():
                proc = state.get("process")
                if proc is None:
                    continue

                if args.freeze_timeout_s > 0:
                    if state.get("current_batch") and state.get("current_started_at") is not None:
                        elapsed = now - state["current_started_at"]
                        if elapsed > args.freeze_timeout_s:
                            batch = state["current_batch"]
                            preview = ", ".join(item["file_id"] for item in batch[:3])
                            if len(batch) > 3:
                                preview += ", ..."
                            print(
                                f"[supervisor] worker {state['worker_id']} exceeded freeze timeout "
                                f"({elapsed/60:.1f}m) on {preview}. Deferring and restarting worker.",
                                flush=True,
                            )
                            for item in batch:
                                note_deferred(
                                    state,
                                    item,
                                    f"freeze timeout after {args.freeze_timeout_s}s",
                                )
                            _terminate_process(proc, state["worker_id"])
                            state["process"] = None
                            state["current_batch"] = None
                            state["current_started_at"] = None
                            spawn_worker(state)
                            continue

                    if not state.get("ready") and state.get("started_at") is not None:
                        elapsed = now - state["started_at"]
                        if elapsed > args.freeze_timeout_s:
                            print(
                                f"[supervisor] worker {state['worker_id']} exceeded startup timeout "
                                f"({elapsed/60:.1f}m). Restarting worker.",
                                flush=True,
                            )
                            _terminate_process(proc, state["worker_id"])
                            state["process"] = None
                            state["current_batch"] = None
                            state["current_started_at"] = None
                            spawn_worker(state)
                            continue

                if proc.is_alive():
                    continue

                exitcode = proc.exitcode
                active_batch = state.get("current_batch") or []
                state["process"] = None
                state["current_batch"] = None
                state["current_started_at"] = None

                if exitcode not in (0, None) and active_batch:
                    preview = ", ".join(item["file_id"] for item in active_batch[:3])
                    if len(active_batch) > 3:
                        preview += ", ..."
                    print(
                        f"[supervisor] worker {state['worker_id']} crashed on {preview}. "
                        f"Deferring active doc(s) before restart.",
                        flush=True,
                    )
                    for item in active_batch:
                        note_deferred(
                            state,
                            item,
                            f"worker exited with code {exitcode} while processing active document",
                        )

                remaining = _remaining_entries(state["entries"], ann_dir, set(deferred_records))
                if not remaining:
                    continue

                print(
                    f"[supervisor] worker {state['worker_id']} exited with code {exitcode}. "
                    f"Restarting remaining docs on GPU {state['gpu_id']}.",
                    flush=True,
                )
                spawn_worker(state)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[supervisor] KeyboardInterrupt received. Stopping all workers...", flush=True)
    finally:
        for state in worker_states.values():
            proc = state.get("process")
            if proc is not None:
                _terminate_process(proc, state["worker_id"])
                state["process"] = None
                state["current_batch"] = None
                state["current_started_at"] = None

        deferred_records = {
            file_id: record
            for file_id, record in deferred_records.items()
            if not _annotation_path(ann_dir, file_id).exists()
        }
        save_deferred_records(deferred_path, deferred_records)

        try:
            status_queue.close()
            status_queue.join_thread()
        except Exception:
            pass

    if interrupted:
        print("[supervisor] All workers stopped. Resume will skip finished docs and continue remaining ones.", flush=True)
        raise SystemExit(130)

    wall_elapsed = time.time() - wall_start

    # ── Summary ────────────────────────────────────────────────────────────
    print("\nBuilding summary ...")
    total_sentences   = 0
    total_ner_ents    = 0
    agg_ner: Counter  = Counter()
    agg_rr:  Counter  = Counter()
    doc_count         = 0
    rr_fallback_count = 0

    for json_file in ann_dir.glob("*.json"):
        try:
            rec = json.loads(json_file.read_text())
        except Exception:
            continue
        doc_count       += 1
        total_sentences += len(rec.get("sentences", []))
        total_ner_ents  += sum(rec.get("ner_by_label", {}).values())
        agg_ner.update(rec.get("ner_by_label", {}))
        agg_rr.update(rec.get("rr_by_role",   {}))
        if not rec.get("rr_available", True):
            rr_fallback_count += 1

    summary = {
        "input_dir":              str(in_dir),
        "annotations_dir":        str(ann_dir),
        "deferred_docs_path":     str(deferred_path),
        "total_files_found":      len(all_files),
        "total_processed":        doc_count,
        "total_errors":           total_errors,
        "docs_full_ner_rr":       doc_count - rr_fallback_count,
        "docs_ner_only_fallback": rr_fallback_count,
        "deferred_docs_remaining": len(deferred_records),
        "deferred_docs_this_run":  deferred_this_run,
        "total_sentences":        total_sentences,
        "total_ner_entities":     total_ner_ents,
        "ner_by_label":           dict(agg_ner.most_common()),
        "rr_by_role":             dict(agg_rr.most_common()),
        "num_workers":            n_workers,
        "use_gpu":                args.use_gpu,
        "wall_time_s":            round(wall_elapsed, 1),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  Done in {wall_elapsed/60:.1f}m  ({n_workers} worker(s))")
    print(f"  {doc_count:,} docs written  ({total_errors} errors)")
    print(f"  Full NER+RR      : {doc_count - rr_fallback_count:,}")
    print(f"  NER-only fallback: {rr_fallback_count:,}")
    print(f"  Deferred docs    : {len(deferred_records):,}")
    print(f"  Total sentences  : {total_sentences:,}")
    print(f"  Total NER ents   : {total_ner_ents:,}")
    print(f"\n  NER entity counts:")
    for label, cnt in agg_ner.most_common():
        print(f"    {label:<22} {cnt:>8,}")
    print(f"\n  RR role counts:")
    for role, cnt in agg_rr.most_common():
        print(f"    {role:<22} {cnt:>8,}")
    print(f"\n  Annotations : {ann_dir}/")
    print(f"  Summary     : {summary_path}")
    print(f"  Deferred    : {deferred_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
