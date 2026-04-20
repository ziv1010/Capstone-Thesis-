"""Paper-faithful evaluation for the rhetorical-role pretrained model."""

from __future__ import annotations

import copy
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.common import flatten_confusion_matrix, safe_div, write_csv, write_json


RR_LABELS = [
    "NONE",
    "PREAMBLE",
    "FAC",
    "ISSUE",
    "ARG_RESPONDENT",
    "ARG_PETITIONER",
    "ANALYSIS",
    "PRE_RELIED",
    "PRE_NOT_RELIED",
    "STA",
    "RLC",
    "RPC",
    "RATIO",
]


def prepare_rr_imports(repo_root: Path) -> None:
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_test_rows(dataset_root: Path) -> list[dict[str, Any]]:
    with (dataset_root / "InRhetoricalRoles" / "test.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        prepared = dict(row)
        prepared["benchmark_doc_id"] = str(row.get("id") or f"rrdoc{index:05d}")
        rows.append(prepared)
    return rows


def write_runtime_hsln_files(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    runtime_dir: Path,
) -> None:
    from transformers import BertTokenizer

    prepare_rr_imports(repo_root)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    data_dir = runtime_dir / "pubmed-20k"
    data_dir.mkdir(parents=True, exist_ok=True)

    final_lines: list[str] = []
    for row in rows:
        final_lines.append(f"###{row['benchmark_doc_id']}")
        for annotation in row["annotations"][0]["result"]:
            sentence_text = annotation["value"]["text"].replace("\r", "")
            if not sentence_text.strip():
                continue
            token_ids = tokenizer.encode(sentence_text, add_special_tokens=True, max_length=128)
            final_lines.append("NONE\t" + " ".join(str(token_id) for token_id in token_ids))
        final_lines.append("")

    full_text = "\n".join(final_lines) + "\n"
    for split_name in ["train_scibert.txt", "dev_scibert.txt", "test_scibert.txt"]:
        (data_dir / split_name).write_text(full_text, encoding="utf-8")


def predict_labels(
    *,
    repo_root: Path,
    model_path: Path,
    rows: list[dict[str, Any]],
    runtime_dir: Path,
    device_name: str,
    logger: Any,
) -> tuple[dict[str, list[str]], str]:
    prepare_rr_imports(repo_root)
    write_runtime_hsln_files(rows, repo_root=repo_root, runtime_dir=runtime_dir)

    import models
    from eval import eval_model
    from models import BertHSLN
    from task import pubmed_task

    config = {
        "bert_model": "bert-base-uncased",
        "bert_trainable": False,
        "model": BertHSLN.__name__,
        "cacheable_tasks": [],
        "dropout": 0.5,
        "word_lstm_hs": 758,
        "att_pooling_dim_ctx": 200,
        "att_pooling_num_ctx": 15,
        "lr": 3e-05,
        "lr_epoch_decay": 0.9,
        "batch_size": 32,
        "max_seq_length": 128,
        "max_epochs": 40,
        "early_stopping": 5,
    }

    device = torch.device(device_name)
    task = pubmed_task(
        train_batch_size=config["batch_size"],
        max_docs=-1,
        data_folder=str(runtime_dir),
    )
    model = getattr(models, config["model"])(config, [task]).to(device)
    logger.info("Loading RR checkpoint from %s on %s", model_path, device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    test_batches = task.get_folds()[0].test
    _, _, labels_dict, _ = eval_model(model, test_batches, device, task)
    predictions = {
        str(doc_name): predicted_labels
        for doc_name, predicted_labels in zip(labels_dict["doc_names"], labels_dict["docwise_y_predicted"])
    }
    return predictions, str(device)


def score_predictions(
    rows: list[dict[str, Any]],
    predictions_by_doc_id: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    per_doc_rows: list[dict[str, Any]] = []
    prediction_payload = copy.deepcopy(rows)
    failures: list[dict[str, Any]] = []

    for row, prediction_row in zip(rows, prediction_payload):
        doc_id = row["benchmark_doc_id"]
        gold_annotations = row["annotations"][0]["result"]
        predicted_labels = predictions_by_doc_id.get(doc_id)
        if predicted_labels is None:
            failures.append({"doc_id": doc_id, "error": "Missing prediction for document."})
            continue
        if len(predicted_labels) != len(gold_annotations):
            failures.append(
                {
                    "doc_id": doc_id,
                    "error": f"Sentence-count mismatch: gold={len(gold_annotations)} pred={len(predicted_labels)}",
                }
            )
            continue

        doc_true: list[str] = []
        doc_pred: list[str] = []
        for annotation, prediction_annotation, predicted_label in zip(
            gold_annotations,
            prediction_row["annotations"][0]["result"],
            predicted_labels,
        ):
            gold_label = str(annotation["value"]["labels"][0]).strip().upper()
            pred_label = str(predicted_label).strip().upper()
            doc_true.append(gold_label)
            doc_pred.append(pred_label)
            prediction_annotation["value"]["labels"] = [pred_label]

        y_true.extend(doc_true)
        y_pred.extend(doc_pred)
        correct = sum(1 for gold_label, pred_label in zip(doc_true, doc_pred) if gold_label == pred_label)
        per_doc_rows.append(
            {
                "doc_id": doc_id,
                "sentences": len(doc_true),
                "correct_sentences": correct,
                "accuracy": safe_div(correct, len(doc_true)),
            }
        )

    labels = [label for label in RR_LABELS if label in set(y_true) | set(y_pred)]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    per_label_rows: list[dict[str, Any]] = []
    per_label_payload: dict[str, Any] = {}
    for label, label_precision, label_recall, label_f1, label_support in zip(labels, precision, recall, f1, support):
        payload = {
            "label": label,
            "support": int(label_support),
            "precision": float(label_precision),
            "recall": float(label_recall),
            "f1": float(label_f1),
        }
        per_label_rows.append(payload)
        per_label_payload[label] = dict(payload)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    report = {
        "task": "Rhetorical_Role",
        "metric_definition": "Sentence-level rhetorical-role classification on the public paper test split.",
        "documents_evaluated": len(per_doc_rows),
        "sentences_evaluated": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_label": per_label_payload,
        "gold_distribution": {label: int((np.array(y_true) == label).sum()) for label in labels},
        "pred_distribution": {label: int((np.array(y_pred) == label).sum()) for label in labels},
    }
    if report["weighted_f1"] > 0.95:
        report["warning"] = (
            "The available pretrained checkpoint appears to score implausibly high on the public paper split. "
            "This likely indicates the released checkpoint was trained beyond the paper's train-only setup."
        )
    return report, per_doc_rows, per_label_rows, flatten_confusion_matrix(labels, confusion.tolist()), {
        "labels": labels,
        "matrix": confusion.tolist(),
        "predictions": prediction_payload,
        "failures": failures,
    }


def run_rr_replication(
    *,
    dataset_root: Path,
    output_dir: Path,
    repo_root: Path,
    model_path: Path,
    runtime_dir: Path,
    logger: Any,
    device_name: str,
) -> dict[str, Any]:
    rows = load_test_rows(dataset_root)
    logger.info("Loaded %s InRhetoricalRoles test documents", len(rows))

    fallback_used: dict[str, Any] | None = None
    failures: list[dict[str, Any]] = []
    runtime_device = device_name
    try:
        predictions_by_doc_id, runtime_device = predict_labels(
            repo_root=repo_root,
            model_path=model_path,
            rows=rows,
            runtime_dir=runtime_dir,
            device_name=device_name,
            logger=logger,
        )
    except Exception:
        if device_name.startswith("cuda"):
            fallback_used = {
                "from": device_name,
                "to": "cpu",
                "reason": traceback.format_exc(),
            }
            logger.exception("RR GPU inference failed. Retrying on CPU.")
            predictions_by_doc_id, runtime_device = predict_labels(
                repo_root=repo_root,
                model_path=model_path,
                rows=rows,
                runtime_dir=runtime_dir,
                device_name="cpu",
                logger=logger,
            )
        else:
            raise

    report, per_doc_rows, per_label_rows, confusion_rows, artifacts = score_predictions(rows, predictions_by_doc_id)
    failures.extend(artifacts["failures"])
    report["num_failed_documents"] = len(failures)
    report["failed_documents"] = [item["doc_id"] for item in failures]
    report["runtime"] = {
        "requested_device": device_name,
        "runtime_device": runtime_device,
        "fallback_used": fallback_used,
    }

    write_json(output_dir / "test_metrics.json", report)
    write_csv(output_dir / "per_label_metrics.csv", per_label_rows)
    write_csv(output_dir / "document_metrics.csv", per_doc_rows)
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    write_json(output_dir / "predictions.json", artifacts["predictions"])
    write_json(output_dir / "failures.json", failures)
    return report
