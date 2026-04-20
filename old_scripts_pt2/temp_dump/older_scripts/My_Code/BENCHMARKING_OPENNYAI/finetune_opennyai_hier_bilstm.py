#!/usr/bin/env python3
"""
Fine-tune the released LegalSeg Hier_BiLSTM-CRF checkpoint on OpenNyAI
InRhetoricalRoles train/dev after collapsing OpenNyAI labels into the 7-label
LegalSeg space.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import benchmark_opennyai_hier_bilstm as bench

LOGGER = logging.getLogger("finetune_opennyai_hier_bilstm")

SCRIPT_DIR = Path(__file__).resolve().parent
MY_CODE_DIR = SCRIPT_DIR.parent
PIPELINE_SCRIPT = MY_CODE_DIR / "legal_pdf_rr_pipeline" / "scripts" / "04_infer_hier_bilstm_crf.py"


@dataclass
class SplitBundle:
    split: str
    inference_docs: list[dict[str, Any]]
    gold_rows: list[dict[str, Any]]
    encoded_docs: list[list[list[int]]]
    gold_tag_sequences: list[list[int]]


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("legalseg_hier_infer", PIPELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load pipeline script: {PIPELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_indices(num_examples: int, batch_size: int, shuffle: bool) -> list[list[int]]:
    indices = list(range(num_examples))
    if shuffle:
        random.shuffle(indices)
    return [indices[i : i + batch_size] for i in range(0, num_examples, batch_size)]


def save_json(path: Path, payload: Any) -> None:
    bench.ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def prepare_split_bundle(
    split: str,
    local_dataset_dir: str | None,
    word2idx: dict[str, int],
    tag2idx: dict[str, int],
    infer_module,
    data_dir: Path,
) -> SplitBundle:
    split_rows = bench.load_opennyai_split(local_dataset_dir=local_dataset_dir, split=split)
    inference_docs, gold_rows = bench.convert_to_benchmark_rows(split_rows=split_rows, split=split)
    if not inference_docs:
        raise RuntimeError(f"No documents available for split={split}")

    save_json(data_dir / f"{split}_inference_input.json", inference_docs)
    save_json(data_dir / f"{split}_gold_segments.json", gold_rows)

    encoded_docs = infer_module.prepare_documents(inference_docs, word2idx)
    gold_rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in gold_rows:
        gold_rows_by_doc.setdefault(row["doc_id"], []).append(row)

    gold_tag_sequences: list[list[int]] = []
    for doc in inference_docs:
        rows = sorted(gold_rows_by_doc[doc["doc_id"]], key=lambda row: int(row["sentence_id"]))
        gold_tag_sequences.append([tag2idx[row["gold_label"]] for row in rows])

    if len(encoded_docs) != len(gold_tag_sequences):
        raise RuntimeError(f"Split={split}: encoded docs / gold docs mismatch")

    for doc, encoded, gold_tags in zip(inference_docs, encoded_docs, gold_tag_sequences):
        if len(encoded) != len(gold_tags):
            raise RuntimeError(
                f"Split={split} doc={doc['doc_id']}: encoded length {len(encoded)} != gold length {len(gold_tags)}"
            )

    return SplitBundle(
        split=split,
        inference_docs=inference_docs,
        gold_rows=gold_rows,
        encoded_docs=encoded_docs,
        gold_tag_sequences=gold_tag_sequences,
    )


def tag_seq_to_benchmark_rows(
    bundle: SplitBundle,
    pred_tag_sequences: list[list[int]],
    idx2tag: dict[int, str],
    role_to_label_id: dict[str, int],
) -> list[dict[str, Any]]:
    gold_rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in bundle.gold_rows:
        gold_rows_by_doc.setdefault(row["doc_id"], []).append(row)

    aligned_rows: list[dict[str, Any]] = []
    for doc, pred_tags in zip(bundle.inference_docs, pred_tag_sequences):
        rows = sorted(gold_rows_by_doc[doc["doc_id"]], key=lambda row: int(row["sentence_id"]))
        if len(rows) != len(pred_tags):
            raise RuntimeError(
                f"Prediction length mismatch for {doc['doc_id']}: got {len(pred_tags)}, expected {len(rows)}"
            )

        for row, pred_tag in zip(rows, pred_tags):
            pred_label_name = idx2tag[int(pred_tag)]
            pred_label_id = role_to_label_id[pred_label_name]
            aligned_rows.append(
                {
                    **row,
                    "pred_label_id": pred_label_id,
                    "pred_label": pred_label_name,
                }
            )
    return aligned_rows


def evaluate_model(
    model,
    bundle: SplitBundle,
    batch_size: int,
    idx2tag: dict[int, str],
    role_to_label_id: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    model.eval()
    predictions: list[list[int]] = []
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in batch_indices(len(bundle.encoded_docs), batch_size, shuffle=False):
            batch_x = [bundle.encoded_docs[idx] for idx in batch]
            batch_y = [bundle.gold_tag_sequences[idx] for idx in batch]
            batch_pred = model(batch_x)
            batch_loss = model._loss(batch_y)
            predictions.extend(batch_pred)
            total_loss += float(batch_loss.item())
            total_batches += 1

    aligned_rows = tag_seq_to_benchmark_rows(
        bundle=bundle,
        pred_tag_sequences=predictions,
        idx2tag=idx2tag,
        role_to_label_id=role_to_label_id,
    )
    metrics = bench.compute_metrics(aligned_rows)
    metrics["avg_loss"] = total_loss / total_batches if total_batches else 0.0
    return metrics, aligned_rows, metrics["avg_loss"]


def train_one_epoch(model, optimizer, bundle: SplitBundle, batch_size: int, grad_clip: float | None) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in batch_indices(len(bundle.encoded_docs), batch_size, shuffle=True):
        batch_x = [bundle.encoded_docs[idx] for idx in batch]
        batch_y = [bundle.gold_tag_sequences[idx] for idx in batch]

        _ = model(batch_x)
        loss = model._loss(batch_y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / total_batches if total_batches else 0.0


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict[str, Any]) -> None:
    bench.ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "name": model.__class__.__name__,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dev_macro_f1": metrics["macro_f1"],
            "best_dev_accuracy": metrics["accuracy"],
        },
        path,
    )


def build_overfitting_report(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {
            "best_dev_epoch": None,
            "best_dev_macro_f1": None,
            "best_test_macro_f1_seen_during_training": None,
            "last_epoch": None,
            "last_dev_macro_f1": None,
            "last_test_macro_f1": None,
            "dev_drop_from_best_to_last": None,
            "test_drop_from_best_seen_to_last": None,
            "overfitting_signal": False,
        }

    best_dev_row = max(history, key=lambda row: row["dev_macro_f1"])
    best_test_row = max(history, key=lambda row: row["test_macro_f1"])
    last_row = history[-1]
    dev_drop = best_dev_row["dev_macro_f1"] - last_row["dev_macro_f1"]
    test_drop = best_test_row["test_macro_f1"] - last_row["test_macro_f1"]

    return {
        "best_dev_epoch": best_dev_row["epoch"],
        "best_dev_macro_f1": best_dev_row["dev_macro_f1"],
        "best_dev_test_macro_f1": best_dev_row["test_macro_f1"],
        "best_test_epoch_seen_during_training": best_test_row["epoch"],
        "best_test_macro_f1_seen_during_training": best_test_row["test_macro_f1"],
        "last_epoch": last_row["epoch"],
        "last_dev_macro_f1": last_row["dev_macro_f1"],
        "last_test_macro_f1": last_row["test_macro_f1"],
        "dev_drop_from_best_to_last": dev_drop,
        "test_drop_from_best_seen_to_last": test_drop,
        "overfitting_signal": bool(dev_drop > 0.01 and last_row["train_loss"] < best_dev_row["train_loss"]),
    }


def run_finetune(
    output_root: Path,
    local_dataset_dir: str | None,
    device: str | None,
    train_batch_size: int,
    eval_batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float | None,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    bench.ensure_dir(output_root)

    data_dir = bench.ensure_dir(output_root / "data")
    outputs_dir = bench.ensure_dir(output_root / "outputs")
    reports_dir = bench.ensure_dir(outputs_dir / "finetune_reports")
    models_dir = bench.ensure_dir(output_root / "models" / "LegalSeg_Hier_BiLSTM_CRF_finetuned")

    infer_module = load_pipeline_module()
    resolved_device = infer_module.resolve_device(device)
    if not resolved_device.startswith("cuda"):
        raise RuntimeError(
            "Fine-tuning the official Hier_BiLSTM-CRF code path is only supported on CUDA in this setup."
        )

    model, word2idx, tag2idx, checkpoint, checkpoint_dir, repo_root = infer_module.load_model(
        models_dir=models_dir,
        device=resolved_device,
    )
    idx2tag = {value: key for key, value in tag2idx.items()}

    train_bundle = prepare_split_bundle(
        split="train",
        local_dataset_dir=local_dataset_dir,
        word2idx=word2idx,
        tag2idx=tag2idx,
        infer_module=infer_module,
        data_dir=data_dir,
    )
    dev_bundle = prepare_split_bundle(
        split="dev",
        local_dataset_dir=local_dataset_dir,
        word2idx=word2idx,
        tag2idx=tag2idx,
        infer_module=infer_module,
        data_dir=data_dir,
    )
    test_bundle = prepare_split_bundle(
        split="test",
        local_dataset_dir=local_dataset_dir,
        word2idx=word2idx,
        tag2idx=tag2idx,
        infer_module=infer_module,
        data_dir=data_dir,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    baseline_dev_metrics, baseline_dev_rows, _ = evaluate_model(
        model=model,
        bundle=dev_bundle,
        batch_size=eval_batch_size,
        idx2tag=idx2tag,
        role_to_label_id=infer_module.ROLE_TO_LABEL_ID,
    )
    baseline_test_metrics, baseline_test_rows, _ = evaluate_model(
        model=model,
        bundle=test_bundle,
        batch_size=eval_batch_size,
        idx2tag=idx2tag,
        role_to_label_id=infer_module.ROLE_TO_LABEL_ID,
    )

    best_dev_macro_f1 = baseline_dev_metrics["macro_f1"]
    history: list[dict[str, Any]] = []
    best_checkpoint_path = models_dir / "model_state_best.tar"

    save_checkpoint(best_checkpoint_path, model, optimizer, epoch=0, metrics=baseline_dev_metrics)
    bench.write_csv(reports_dir / "baseline_dev_aligned.csv", baseline_dev_rows)
    bench.write_csv(reports_dir / "baseline_test_aligned.csv", baseline_test_rows)
    save_json(reports_dir / "baseline_dev_metrics.json", baseline_dev_metrics)
    save_json(reports_dir / "baseline_test_metrics.json", baseline_test_metrics)

    LOGGER.info(
        "Baseline dev/test macro_f1: %.4f / %.4f",
        baseline_dev_metrics["macro_f1"],
        baseline_test_metrics["macro_f1"],
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            bundle=train_bundle,
            batch_size=train_batch_size,
            grad_clip=grad_clip,
        )
        dev_metrics, _, dev_loss = evaluate_model(
            model=model,
            bundle=dev_bundle,
            batch_size=eval_batch_size,
            idx2tag=idx2tag,
            role_to_label_id=infer_module.ROLE_TO_LABEL_ID,
        )
        test_metrics, _, test_loss = evaluate_model(
            model=model,
            bundle=test_bundle,
            batch_size=eval_batch_size,
            idx2tag=idx2tag,
            role_to_label_id=infer_module.ROLE_TO_LABEL_ID,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "test_loss": test_loss,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
        }
        history.append(epoch_row)
        LOGGER.info(
            "Epoch %d/%d train_loss=%.4f dev_loss=%.4f dev_acc=%.4f dev_macro_f1=%.4f test_macro_f1=%.4f",
            epoch,
            epochs,
            train_loss,
            dev_loss,
            dev_metrics["accuracy"],
            dev_metrics["macro_f1"],
            test_metrics["macro_f1"],
        )

        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            save_checkpoint(best_checkpoint_path, model, optimizer, epoch=epoch, metrics=dev_metrics)

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    missing_keys, unexpected_keys = model.load_state_dict(best_checkpoint["state_dict"], strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Best checkpoint reload mismatch. missing={missing_keys}, unexpected={unexpected_keys}"
        )

    final_dev_metrics, final_dev_rows, _ = evaluate_model(
        model=model,
        bundle=dev_bundle,
        batch_size=eval_batch_size,
        idx2tag=idx2tag,
        role_to_label_id=infer_module.ROLE_TO_LABEL_ID,
    )
    final_test_metrics, final_test_rows, _ = evaluate_model(
        model=model,
        bundle=test_bundle,
        batch_size=eval_batch_size,
        idx2tag=idx2tag,
        role_to_label_id=infer_module.ROLE_TO_LABEL_ID,
    )

    bench.write_csv(reports_dir / "final_dev_aligned.csv", final_dev_rows)
    bench.write_csv(reports_dir / "final_test_aligned.csv", final_test_rows)
    save_json(reports_dir / "final_dev_metrics.json", final_dev_metrics)
    save_json(reports_dir / "final_test_metrics.json", final_test_metrics)
    pd.DataFrame(history).to_csv(reports_dir / "training_history.csv", index=False)
    overfitting_report = build_overfitting_report(history)
    save_json(reports_dir / "overfitting_report.json", overfitting_report)

    summary = {
        "source_checkpoint_dir": str(checkpoint_dir),
        "official_legalseg_repo": str(repo_root),
        "device": resolved_device,
        "seed": seed,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip": grad_clip,
        "train_docs": len(train_bundle.inference_docs),
        "dev_docs": len(dev_bundle.inference_docs),
        "test_docs": len(test_bundle.inference_docs),
        "baseline_dev_macro_f1": baseline_dev_metrics["macro_f1"],
        "baseline_test_macro_f1": baseline_test_metrics["macro_f1"],
        "final_dev_macro_f1": final_dev_metrics["macro_f1"],
        "final_test_macro_f1": final_test_metrics["macro_f1"],
        "baseline_dev_accuracy": baseline_dev_metrics["accuracy"],
        "baseline_test_accuracy": baseline_test_metrics["accuracy"],
        "final_dev_accuracy": final_dev_metrics["accuracy"],
        "final_test_accuracy": final_test_metrics["accuracy"],
        "best_checkpoint_epoch": int(best_checkpoint["epoch"]),
        "overfitting_report": overfitting_report,
    }
    save_json(reports_dir / "finetune_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LegalSeg Hier_BiLSTM-CRF on OpenNyAI train/dev."
    )
    parser.add_argument(
        "--output_root",
        default=str(SCRIPT_DIR),
        help="Root folder for downloaded data, model artifacts, and reports.",
    )
    parser.add_argument(
        "--local_dataset_dir",
        default=None,
        help="Optional local OpenNyAI dataset export directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device. This fine-tune path expects CUDA.",
    )
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    logs_dir = bench.ensure_dir(output_root / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logs_dir / "finetune_opennyai_hier_bilstm.log", mode="w", encoding="utf-8"),
        ],
    )

    summary = run_finetune(
        output_root=output_root,
        local_dataset_dir=args.local_dataset_dir,
        device=args.device,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
    )
    LOGGER.info(
        "Fine-tuning complete: baseline test macro_f1=%.4f -> final test macro_f1=%.4f",
        summary["baseline_test_macro_f1"],
        summary["final_test_macro_f1"],
    )


if __name__ == "__main__":
    main()
