#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from explain_hgt import choose_device, load_model, load_yaml
from pattern_explainability_common import (
    DEFAULT_EXPLANATION_DIR,
    DEFAULT_PATTERN_OUTPUT_DIR,
    build_case_metadata,
    dump_json,
    infer_artifact_paths,
    load_graph,
    load_predictions,
    select_case_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen HGT case embeddings for pattern analyses.")
    parser.add_argument("--explanation-dir", type=Path, default=DEFAULT_EXPLANATION_DIR)
    parser.add_argument("--graph-cache", type=Path, default=None)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PATTERN_OUTPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--split", default="all")
    parser.add_argument("--case-limit", type=int, default=None, help="Smoke-test output row limit.")
    parser.add_argument("--device", default="cuda:0")
    return parser


def move_tensor_dict(tensors: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in tensors.items()}


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_file or (args.output_dir / "hgt_case_embeddings.npz")

    paths = infer_artifact_paths(
        args.explanation_dir,
        graph_cache=args.graph_cache,
        predictions_csv=args.predictions_csv,
        config=args.config,
        model_path=args.model_path,
    )
    cfg = load_yaml(paths["config"])
    data, metadata = load_graph(paths["graph_cache"])
    predictions = load_predictions(paths["predictions_csv"], expected_cases=int(data["case"].num_nodes))
    case_meta = build_case_metadata(data, predictions)
    selected_case_indices = select_case_indices(case_meta, args.split, args.case_limit)
    label_names = [str(label) for label in metadata["label_names"]]

    device = choose_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script is configured for GPU extraction; pass a CUDA device such as cuda:0.")
    print(f"[load] model={paths['model_path']} graph={paths['graph_cache']} device={device}", flush=True)
    model = load_model(
        data=data,
        label_names=label_names,
        cfg=cfg,
        model_path=paths["model_path"],
        device=device,
        record_attention=False,
    )
    x_dict = move_tensor_dict(data.x_dict, device)
    edge_index_dict = move_tensor_dict(data.edge_index_dict, device)
    print(f"[forward] full graph, output_cases={len(selected_case_indices)}", flush=True)
    with torch.no_grad():
        logits, hidden = model(x_dict, edge_index_dict)
        probs = torch.softmax(logits, dim=-1)
    selected_index_tensor = torch.tensor(
        selected_case_indices.copy(),
        dtype=torch.long,
        device=device,
    )
    embeddings = hidden["case"][selected_index_tensor].detach().cpu().numpy()
    selected_probs = probs[selected_index_tensor].detach().cpu().numpy()
    selected_logits = logits[selected_index_tensor].detach().cpu().numpy()

    selected_meta = case_meta.set_index("case_index").loc[selected_case_indices].reset_index()
    np.savez_compressed(
        output_file,
        case_indices=selected_case_indices.astype(np.int64),
        embeddings=embeddings.astype(np.float32),
        logits=selected_logits.astype(np.float32),
        probabilities=selected_probs.astype(np.float32),
        case_id=selected_meta["case_id"].astype(str).to_numpy(),
        file_name=selected_meta["file_name"].astype(str).to_numpy(),
        split=selected_meta["split"].astype(str).to_numpy(),
        target_label=selected_meta["target_label"].astype(str).to_numpy(),
        pred_label=selected_meta["pred_label"].astype(str).to_numpy(),
        confidence=selected_meta["confidence"].astype(float).to_numpy(),
        label_names=np.asarray(label_names, dtype=str),
    )
    dump_json(
        args.output_dir / "hgt_case_embeddings_manifest.json",
        {
            "hgt_case_embeddings": str(output_file),
            "graph_cache": str(paths["graph_cache"]),
            "model_path": str(paths["model_path"]),
            "config": str(paths["config"]),
            "predictions_csv": str(paths["predictions_csv"]),
            "split": args.split,
            "case_limit": args.case_limit,
            "device": str(device),
            "embedding_shape": list(embeddings.shape),
            "elapsed_seconds": time.time() - start,
        },
    )
    print(f"[done] embeddings={embeddings.shape} file={output_file}", flush=True)


if __name__ == "__main__":
    main()
