#!/usr/bin/env python3
"""Add case-outcome labels to summary-enriched JSON files.

This script reads the enriched JSONs produced by
`run_opennyai_summarizer_custom.py`, extracts:

- `opennyai_summary.decision`
- all sentence texts whose `rhetorical_role` is `RPC`

It then applies the same Mistral/vLLM case-outcome classification flow used by
`add_case_outcome_labels_mistral.py` and writes one labeled JSON per document.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from add_case_outcome_labels_mistral import (  # noqa: E402
    DEFAULT_MODEL_ID,
    LABEL_TO_SCORE,
    apply_classification_to_payload,
    build_classifier,
)
from src.io_utils import write_json  # noqa: E402


DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs" / "summary_from_annotations" / "enriched_jsons"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "summary_from_annotations_labeled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", choices=["local_vllm", "remote_hf"], default="local_vllm")
    parser.add_argument("--provider", default="auto")
    parser.add_argument(
        "--hf_token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output_dir by skipping already-finished files.",
    )
    parser.add_argument("--dry_run", action="store_true")
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
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tokenizer_mode", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument(
        "--cuda_visible_devices",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value for local_vllm runs.",
    )
    return parser.parse_args()


def collect_input_texts_from_enriched(payload: dict[str, Any]) -> dict[str, Any]:
    summary_mapping = payload.get("opennyai_summary", {})
    if not isinstance(summary_mapping, dict):
        summary_mapping = {}

    decision_text = str(summary_mapping.get("decision", "")).strip()
    rpc_texts: list[str] = []
    for sentence in payload.get("sentences", []):
        if not isinstance(sentence, dict):
            continue
        role = str(sentence.get("rhetorical_role") or "NONE").strip().upper()
        if role != "RPC":
            continue
        text = str(sentence.get("text", "")).strip()
        if text:
            rpc_texts.append(text)

    return {
        "decision_text": decision_text,
        "rpc_texts": rpc_texts,
    }


def flush_local_vllm_batch(
    *,
    pending_batch: list[dict[str, Any]],
    classifier: Any,
    args: argparse.Namespace,
) -> int:
    if not pending_batch:
        return 0

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
                combined_payload=item["payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            write_json(item["output_path"], item["payload"])
        return errors
    except Exception:
        pass

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
                combined_payload=item["payload"],
                extracted=item["extracted"],
                classification=classification,
                args=args,
            )
            write_json(item["output_path"], item["payload"])
        except Exception as exc:  # pragma: no cover - runtime behavior
            item["record"]["status"] = "error"
            item["record"]["error"] = str(exc)
            errors += 1
            write_json(item["output_path"], item["payload"])

    return errors


def main() -> int:
    load_dotenv()
    args = parse_args()

    if args.cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices.strip()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    labelled_dir = output_dir / "labelled_jsons"
    manifest_path = output_dir / "case_outcomes.jsonl"
    run_summary_path = output_dir / "run_summary.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together.")

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    labelled_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.hf_token:
        raise EnvironmentError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN."
        )

    classifier = None
    if not args.dry_run:
        classifier = build_classifier(args)

    paths = sorted(input_dir.glob("*.json"))
    if args.resume:
        paths = [path for path in paths if not (labelled_dir / path.name).exists()]
    if args.max_files is not None:
        paths = paths[: args.max_files]

    skipped = 0
    errors = 0
    pending_local_vllm_batch: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    from tqdm import tqdm

    for path in tqdm(paths, desc="Labeling case outcomes"):
        output_path = labelled_dir / path.name
        payload = json.loads(path.read_text(encoding="utf-8"))
        extracted = collect_input_texts_from_enriched(payload)
        file_id = str(payload.get("file_id") or path.stem).strip() or path.stem

        record = {
            "file_id": file_id,
            "source_json": str(path),
            "labelled_json": str(output_path),
            "decision_text": extracted["decision_text"],
            "rpc_texts": extracted["rpc_texts"],
            "case_outcome_label": None,
            "case_outcome_score": None,
            "status": "pending",
        }
        records.append(record)

        if not extracted["decision_text"] and not extracted["rpc_texts"]:
            record["status"] = "skipped_missing_decision_and_rpc"
            skipped += 1
            write_json(output_path, payload)
            continue

        if args.dry_run:
            record["status"] = "dry_run_only"
            write_json(output_path, payload)
            continue

        payload = copy.deepcopy(payload)
        if args.backend == "local_vllm":
            pending_local_vllm_batch.append(
                {
                    "file_id": file_id,
                    "payload": payload,
                    "extracted": extracted,
                    "record": record,
                    "output_path": output_path,
                }
            )
            if len(pending_local_vllm_batch) >= args.generation_batch_size:
                errors += flush_local_vllm_batch(
                    pending_batch=pending_local_vllm_batch,
                    classifier=classifier,
                    args=args,
                )
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
                combined_payload=payload,
                extracted=extracted,
                classification=classification,
                args=args,
            )
        except Exception as exc:  # pragma: no cover - runtime behavior
            record["status"] = "error"
            record["error"] = str(exc)
            errors += 1

        write_json(output_path, payload)
        if args.backend == "remote_hf" and args.sleep_seconds > 0:
            import time

            time.sleep(args.sleep_seconds)

    if pending_local_vllm_batch:
        errors += flush_local_vllm_batch(
            pending_batch=pending_local_vllm_batch,
            classifier=classifier,
            args=args,
        )

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in records:
            manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "labelled_dir": str(labelled_dir),
        "backend": args.backend,
        "model_id": args.model_id,
        "provider": args.provider if args.backend == "remote_hf" else "local_gpu",
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "files_attempted_this_run": len(paths),
        "successful_files": sum(1 for item in records if item["status"] == "ok"),
        "dry_run_files": sum(1 for item in records if item["status"] == "dry_run_only"),
        "skipped_files": skipped,
        "error_files": errors,
        "label_counts": {
            label: sum(1 for item in records if item.get("case_outcome_label") == label)
            for label in LABEL_TO_SCORE
        },
    }
    write_json(run_summary_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
