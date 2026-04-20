#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hetero_gnn import HeteroLegalOutcomeGNN
from src.training.evaluate import evaluate_split
from src.training.metrics import save_confusion_matrix_plot
from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved GNN checkpoint on a cached graph bundle without retraining."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Checkpoint Evaluation")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for inference: auto, cpu, or cuda.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    requested = str(device_arg).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}")


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint at {checkpoint_path} does not contain a state dict.")
    if payload and all(isinstance(key, str) and key.startswith("module.") for key in payload):
        payload = {key[len("module."):]: value for key, value in payload.items()}
    return payload


def _infer_category(file_name: str) -> str:
    if "__" not in file_name:
        return "unknown"
    return file_name.split("__", 1)[0] or "unknown"


def _load_reference_bundle(checkpoint_path: Path) -> tuple[dict[str, Any] | None, Path | None]:
    checkpoint_dir = checkpoint_path.parent
    reference_cfg_path = checkpoint_dir / "run_config_snapshot.yaml"
    if not reference_cfg_path.exists():
        return None, None
    reference_cfg = load_yaml(reference_cfg_path)
    reference_graph_cache_path = (
        Path(reference_cfg.get("paths", {}).get("graph_cache_dir"))
        / str(reference_cfg.get("graph", {}).get("cache_name", "case_star_graph.pt"))
    )
    if not reference_graph_cache_path.exists():
        raise FileNotFoundError(
            f"Reference graph cache from checkpoint config does not exist: {reference_graph_cache_path}"
        )
    return (
        torch.load(reference_graph_cache_path, map_location="cpu", weights_only=False),
        reference_graph_cache_path,
    )


def _align_graph_to_reference(
    current_data: Any,
    reference_data: Any,
) -> tuple[Any, list[str], list[str]]:
    added_node_types: list[str] = []
    added_edge_types: list[str] = []

    for node_type in reference_data.node_types:
        if node_type in current_data.node_types:
            current_dim = int(current_data[node_type].x.shape[1])
            reference_dim = int(reference_data[node_type].x.shape[1])
            if current_dim != reference_dim:
                raise RuntimeError(
                    f"Feature dimension mismatch for node type '{node_type}': "
                    f"current={current_dim} reference={reference_dim}"
                )
            continue

        reference_x = reference_data[node_type].x
        current_data[node_type].x = torch.zeros(
            (0, int(reference_x.shape[1])),
            dtype=reference_x.dtype,
        )
        current_data[node_type].node_id = []
        added_node_types.append(node_type)

    for edge_type in reference_data.edge_types:
        if edge_type in current_data.edge_types:
            continue
        current_data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
        added_edge_types.append("|".join(edge_type))

    return current_data, added_node_types, added_edge_types


def _split_name_at(index: int, train_mask: Any, val_mask: Any, test_mask: Any) -> str:
    if bool(train_mask[index]):
        return "train"
    if bool(val_mask[index]):
        return "val"
    if bool(test_mask[index]):
        return "test"
    return "unassigned"


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths_cfg = cfg.get("paths", {})
    graph_cache_path = Path(
        args.graph_cache
        or (
            Path(paths_cfg.get("graph_cache_dir"))
            / str(cfg.get("graph", {}).get("cache_name", "case_star_graph.pt"))
        )
    )
    checkpoint_path = Path(args.checkpoint)
    output_dir = ensure_dir(args.output_dir)
    logger = configure_logger("evaluate_saved_model", log_dir=output_dir)
    device = _resolve_device(args.device)

    bundle = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    metadata = bundle["metadata"]
    label_names = list(metadata["label_names"])
    data = bundle["data"]
    reference_bundle, reference_graph_cache_path = _load_reference_bundle(checkpoint_path)
    reference_metadata = None
    added_node_types: list[str] = []
    added_edge_types: list[str] = []
    if reference_bundle is not None:
        data, added_node_types, added_edge_types = _align_graph_to_reference(
            current_data=data,
            reference_data=reference_bundle["data"],
        )
        reference_metadata = reference_bundle["data"].metadata()

    model_metadata = reference_metadata or data.metadata()
    input_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in model_metadata[0]}
    model = HeteroLegalOutcomeGNN(
        metadata=model_metadata,
        input_dims=input_dims,
        out_dim=len(label_names),
        cfg=cfg.get("model", {}),
    )

    state_dict = _load_state_dict(checkpoint_path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint load mismatch. "
            f"missing_keys={list(incompatible.missing_keys)} "
            f"unexpected_keys={list(incompatible.unexpected_keys)}"
        )

    model = model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        x_dict = {node_type: data[node_type].x for node_type in model_metadata[0]}
        edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in model_metadata[1]}
        logits, hidden = model(x_dict, edge_index_dict)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)

    y_true = data["case"].y.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    y_proba = probabilities.detach().cpu().numpy()
    train_mask = data["case"].train_mask.detach().cpu().numpy()
    val_mask = data["case"].val_mask.detach().cpu().numpy()
    test_mask = data["case"].test_mask.detach().cpu().numpy()
    case_ids = list(data["case"].case_id)
    file_names = list(data["case"].file_name)
    raw_labels = [str(item) for item in list(data["case"].raw_label)]
    split_names = [_split_name_at(index, train_mask, val_mask, test_mask) for index in range(len(case_ids))]
    categories = [_infer_category(file_name) for file_name in file_names]

    overall_metrics = evaluate_split(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        label_names=label_names,
    )

    bundle_split_metrics: dict[str, Any] = {}
    for split_name, mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        bundle_split_metrics[split_name] = evaluate_split(
            y_true=y_true[mask],
            y_pred=y_pred[mask],
            y_proba=y_proba[mask],
            label_names=label_names,
        )

    per_category_metrics: dict[str, Any] = {}
    for category in sorted(set(categories)):
        indices = [index for index, value in enumerate(categories) if value == category]
        per_category_metrics[category] = evaluate_split(
            y_true=y_true[indices],
            y_pred=y_pred[indices],
            y_proba=y_proba[indices],
            label_names=label_names,
        )

    target_counts = Counter(str(label_names[index]) for index in y_true.tolist())
    predicted_counts = Counter(str(label_names[index]) for index in y_pred.tolist())
    category_counts = Counter(categories)
    split_counts = Counter(split_names)
    category_label_distribution: dict[str, dict[str, int]] = defaultdict(dict)
    for category in sorted(set(categories)):
        category_indices = [idx for idx, value in enumerate(categories) if value == category]
        counts = Counter(str(label_names[index]) for index in y_true[category_indices].tolist())
        category_label_distribution[category] = dict(sorted(counts.items()))

    prediction_frame = pd.DataFrame(
        {
            "case_id": case_ids,
            "file_name": file_names,
            "category": categories,
            "raw_label": raw_labels,
            "bundle_split": split_names,
            "target_index": y_true.tolist(),
            "target_label": [label_names[index] for index in y_true.tolist()],
            "pred_index": y_pred.tolist(),
            "pred_label": [label_names[index] for index in y_pred.tolist()],
            "confidence": y_proba.max(axis=1).tolist(),
        }
    )
    for label_index, label_name in enumerate(label_names):
        prediction_frame[f"proba_{label_name}"] = y_proba[:, label_index].tolist()

    metrics_payload = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "graph_cache_path": str(graph_cache_path.resolve()),
        "reference_graph_cache_path": (
            str(reference_graph_cache_path.resolve()) if reference_graph_cache_path is not None else None
        ),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "n_cases": int(len(case_ids)),
        "label_names": label_names,
        "alignment": {
            "used_reference_graph_metadata": reference_bundle is not None,
            "added_node_types": added_node_types,
            "added_edge_types": added_edge_types,
        },
        "overall": overall_metrics,
        "bundle_split_metrics": bundle_split_metrics,
        "per_category": per_category_metrics,
        "target_label_distribution": dict(sorted(target_counts.items())),
        "predicted_label_distribution": dict(sorted(predicted_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "bundle_split_counts": dict(sorted(split_counts.items())),
        "category_label_distribution": dict(category_label_distribution),
        "graph_metadata_summary": {
            "node_counts": metadata.get("node_counts", {}),
            "case_split_counts": metadata.get("case_split_counts", {}),
            "encoder_backend": metadata.get("encoder_backend"),
            "embedding_dim": metadata.get("embedding_dim"),
            "feature_dim": metadata.get("feature_dim"),
        },
    }

    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    dump_json(metrics_payload, output_dir / "metrics.json")
    dump_yaml(cfg, output_dir / "run_config_snapshot.yaml")
    torch.save(
        {
            "case_embeddings": hidden["case"].detach().cpu(),
            "probabilities": probabilities.detach().cpu(),
            "predictions": predictions.detach().cpu(),
        },
        output_dir / "inference_outputs.pt",
    )

    confusion = overall_metrics.get("confusion_matrix", [])
    if confusion:
        save_confusion_matrix_plot(
            confusion=confusion,
            label_names=label_names,
            output_path=output_dir / "confusion_matrix_overall.png",
            title=args.title,
        )

    logger.info(
        "Evaluation complete. n_cases=%d accuracy=%.4f macro_f1=%.4f checkpoint=%s graph_cache=%s",
        len(case_ids),
        float(overall_metrics.get("accuracy", 0.0)),
        float(overall_metrics.get("macro_f1", 0.0)),
        checkpoint_path,
        graph_cache_path,
    )


if __name__ == "__main__":
    main()
