#!/usr/bin/env python3
"""
Benchmark LegalSeg Hier_BiLSTM-CRF on the OpenNyAI InRhetoricalRoles test split.

This is a cross-dataset transfer benchmark. OpenNyAI labels are collapsed into the
7-label LegalSeg space before inference-time comparison.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

LOGGER = logging.getLogger("benchmark_opennyai_hier_bilstm")

OPENNYAI_TO_7 = {
    "NONE": 0,
    "PREAMBLE": 0,
    "FAC": 1,
    "RLC": 1,
    "ISSUE": 2,
    "ARG_PETITIONER": 3,
    "ARG_RESPONDENT": 4,
    "ANALYSIS": 5,
    "STA": 5,
    "PRE_RELIED": 5,
    "PRE_NOT_RELIED": 5,
    "RATIO": 5,
    "RPC": 6,
}

LABEL_ID_TO_NAME = {
    0: "None",
    1: "Facts",
    2: "Issue",
    3: "Arguments of Petitioner",
    4: "Arguments of Respondent",
    5: "Reasoning",
    6: "Decision",
}

SCRIPT_DIR = Path(__file__).resolve().parent
MY_CODE_DIR = SCRIPT_DIR.parent
PIPELINE_SCRIPT = MY_CODE_DIR / "legal_pdf_rr_pipeline" / "scripts" / "04_infer_hier_bilstm_crf.py"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("legalseg_hier_infer", PIPELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pipeline script: {PIPELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_opennyai_label(raw_label: str) -> str:
    return raw_label.strip().upper()


def load_opennyai_split(local_dataset_dir: str | None, split: str) -> list[dict[str, Any]]:
    if local_dataset_dir:
        dataset_root = Path(local_dataset_dir).resolve()
        json_path = dataset_root / f"{split}.json"
        parquet_candidates = sorted((dataset_root / "data").glob(f"{split}-*.parquet"))
        if json_path.is_file():
            LOGGER.info("Loading local JSON split from %s", json_path)
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise RuntimeError(f"Expected list payload in {json_path}, found {type(data).__name__}")
            return data
        if parquet_candidates:
            parquet_path = parquet_candidates[0]
            LOGGER.info("Loading local parquet split from %s", parquet_path)
            return pd.read_parquet(parquet_path).to_dict(orient="records")
        raise RuntimeError(
            f"Could not find {split}.json or data/{split}-*.parquet under {dataset_root}"
        )

    LOGGER.info("Loading gated HF dataset opennyaiorg/InRhetoricalRoles split=%s", split)
    try:
        dataset = load_dataset("opennyaiorg/InRhetoricalRoles", split=split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load opennyaiorg/InRhetoricalRoles. "
            "If this dataset is gated, either run `huggingface-cli login` inside LAW_PRELAB, "
            "set HF_TOKEN, or download the dataset locally and pass --local_dataset_dir."
        ) from exc
    return [dataset[idx] for idx in range(len(dataset))]


def extract_annotation_results(row: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for annotation in row.get("annotations", []):
        for result in annotation.get("result", []):
            value = result.get("value", {})
            text = str(value.get("text", "")).strip()
            labels = value.get("labels", []) or []
            if not text or not labels:
                continue
            results.append(
                {
                    "start": int(value.get("start", 0)),
                    "end": int(value.get("end", 0)),
                    "text": text,
                    "raw_label": normalize_opennyai_label(str(labels[0])),
                }
            )
    results.sort(key=lambda item: (item["start"], item["end"]))
    return results


def make_doc_id(split: str, index: int, row: dict[str, Any]) -> str:
    meta_group = ""
    meta = row.get("meta")
    if isinstance(meta, dict):
        meta_group = str(meta.get("group", "")).strip()
    if meta_group:
        return f"{split}_{index:04d}_{meta_group}"
    return f"{split}_{index:04d}"


def convert_to_benchmark_rows(split_rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inference_docs: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    unknown_labels: Counter[str] = Counter()

    for index, row in enumerate(split_rows):
        doc_id = make_doc_id(split, index, row)
        results = extract_annotation_results(row)
        if not results:
            LOGGER.warning("Skipping %s: no annotated segments found", doc_id)
            continue

        segments: list[str] = []
        for segment_id, result in enumerate(results):
            raw_label = result["raw_label"]
            mapped_label_id = OPENNYAI_TO_7.get(raw_label)
            if mapped_label_id is None:
                unknown_labels[raw_label] += 1
                continue
            clean_text = " ".join(result["text"].split())
            if not clean_text:
                continue
            segments.append(clean_text)
            gold_rows.append(
                {
                    "doc_id": doc_id,
                    "sentence_id": segment_id,
                    "start": result["start"],
                    "end": result["end"],
                    "text": clean_text,
                    "opennyai_label": raw_label,
                    "gold_label_id": mapped_label_id,
                    "gold_label": LABEL_ID_TO_NAME[mapped_label_id],
                }
            )

        if segments:
            inference_docs.append({"doc_id": doc_id, "segments": segments})

    if unknown_labels:
        raise RuntimeError(
            f"Encountered unmapped OpenNyAI labels: {dict(sorted(unknown_labels.items()))}"
        )
    return inference_docs, gold_rows


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(aligned_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(aligned_rows)
    correct = sum(1 for row in aligned_rows if row["gold_label_id"] == row["pred_label_id"])
    accuracy = safe_div(correct, total)

    per_label: dict[int, dict[str, Any]] = {}
    macro_f1_sum = 0.0
    weighted_f1_sum = 0.0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for label_id, label_name in LABEL_ID_TO_NAME.items():
        tp = sum(
            1
            for row in aligned_rows
            if row["gold_label_id"] == label_id and row["pred_label_id"] == label_id
        )
        fp = sum(
            1
            for row in aligned_rows
            if row["gold_label_id"] != label_id and row["pred_label_id"] == label_id
        )
        fn = sum(
            1
            for row in aligned_rows
            if row["gold_label_id"] == label_id and row["pred_label_id"] != label_id
        )
        support = sum(1 for row in aligned_rows if row["gold_label_id"] == label_id)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_label[label_id] = {
            "label": label_name,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        macro_f1_sum += f1
        weighted_f1_sum += f1 * support

    for row in aligned_rows:
        gold = LABEL_ID_TO_NAME[row["gold_label_id"]]
        pred = LABEL_ID_TO_NAME[row["pred_label_id"]]
        confusion[gold][pred] += 1

    return {
        "num_segments": total,
        "accuracy": accuracy,
        "macro_f1": safe_div(macro_f1_sum, len(LABEL_ID_TO_NAME)),
        "weighted_f1": safe_div(weighted_f1_sum, total),
        "per_label": per_label,
        "gold_distribution": dict(sorted(Counter(row["gold_label"] for row in aligned_rows).items())),
        "pred_distribution": dict(sorted(Counter(row["pred_label"] for row in aligned_rows).items())),
        "confusion_matrix": {gold: dict(sorted(preds.items())) for gold, preds in sorted(confusion.items())},
    }


def benchmark(
    output_root: Path,
    split: str,
    local_dataset_dir: str | None,
    device: str | None,
    batch_size: int,
    max_docs: int | None,
) -> dict[str, Any]:
    data_dir = ensure_dir(output_root / "data")
    outputs_dir = ensure_dir(output_root / "outputs")
    predictions_dir = ensure_dir(outputs_dir / "predictions")
    models_dir = ensure_dir(output_root / "models" / "LegalSeg_Hier_BiLSTM_CRF")

    split_rows = load_opennyai_split(local_dataset_dir=local_dataset_dir, split=split)
    if max_docs is not None:
        split_rows = split_rows[:max_docs]
        LOGGER.info("Limiting benchmark to first %s documents", max_docs)

    inference_docs, gold_rows = convert_to_benchmark_rows(split_rows=split_rows, split=split)
    if not inference_docs:
        raise RuntimeError("No inference documents were created from the OpenNyAI split.")

    write_json(data_dir / f"{split}_inference_input.json", inference_docs)
    write_json(data_dir / f"{split}_gold_segments.json", gold_rows)

    pipeline_module = load_pipeline_module()
    pipeline_module.run(
        input_path=str(data_dir / f"{split}_inference_input.json"),
        output_dir=str(predictions_dir),
        models_dir=str(models_dir),
        device=device,
        batch_size=batch_size,
    )

    predictions_csv = predictions_dir / "all_predictions.csv"
    if not predictions_csv.is_file():
        raise RuntimeError(f"Expected predictions CSV at {predictions_csv}")

    # Preserve the literal label string "None" instead of letting pandas coerce
    # it to NaN, which would make the benchmark report misleading.
    prediction_rows = pd.read_csv(predictions_csv, keep_default_na=False).to_dict(orient="records")
    pred_lookup = {
        (row["doc_id"], int(row["sentence_id"])): {
            "pred_label_id": int(row["label_id"]),
            "pred_label": str(row["label"]),
        }
        for row in prediction_rows
    }

    aligned_rows: list[dict[str, Any]] = []
    missing_predictions: list[dict[str, Any]] = []
    for row in gold_rows:
        key = (row["doc_id"], int(row["sentence_id"]))
        pred = pred_lookup.get(key)
        if pred is None:
            missing_predictions.append({"doc_id": row["doc_id"], "sentence_id": row["sentence_id"]})
            continue
        aligned_rows.append({**row, **pred})

    if missing_predictions:
        raise RuntimeError(
            f"Missing predictions for {len(missing_predictions)} segments. "
            f"Example: {missing_predictions[:5]}"
        )

    metrics = compute_metrics(aligned_rows)
    results = {
        "dataset": "opennyaiorg/InRhetoricalRoles",
        "split": split,
        "local_dataset_dir": str(Path(local_dataset_dir).resolve()) if local_dataset_dir else None,
        "num_documents": len(inference_docs),
        **metrics,
    }

    write_csv(outputs_dir / "aligned_gold_vs_predictions.csv", aligned_rows)
    write_json(outputs_dir / "benchmark_results.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LegalSeg Hier_BiLSTM-CRF on OpenNyAI InRhetoricalRoles."
    )
    parser.add_argument(
        "--output_root",
        default=str(SCRIPT_DIR),
        help="All benchmark files are written under this directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="OpenNyAI split to benchmark.",
    )
    parser.add_argument(
        "--local_dataset_dir",
        default=None,
        help="Optional local dataset export directory containing split JSON/parquet files.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu or cuda:0. Auto-detect if omitted.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Documents per inference batch.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional cap for quick dry-runs.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    logs_dir = ensure_dir(output_root / "logs")
    log_path = logs_dir / "benchmark_opennyai_hier_bilstm.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )

    LOGGER.info("Using benchmark root: %s", output_root)
    LOGGER.info("Pipeline script: %s", PIPELINE_SCRIPT)
    if "HF_TOKEN" in os.environ:
        LOGGER.info("HF_TOKEN detected in environment")

    results = benchmark(
        output_root=output_root,
        split=args.split,
        local_dataset_dir=args.local_dataset_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
    )
    LOGGER.info("Benchmark complete: accuracy=%.4f macro_f1=%.4f", results["accuracy"], results["macro_f1"])


if __name__ == "__main__":
    main()
