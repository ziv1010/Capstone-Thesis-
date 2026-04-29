#!/usr/bin/env python
"""Phase 1-2: Freeze the HGT, run full-graph inference, save artefacts,
and build a FAISS index over training-set case embeddings."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyser.loader import build_and_load_hgt, load_config, load_graph_cache  # noqa: E402
from analyser.hetero_utils import get_split_indices, load_fold_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    output_root = Path(cfg["output_root"]) / "phase1_2_inference"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[phase1_2] loading graph cache: {cfg['graph_cache']}", flush=True)
    data, metadata = load_graph_cache(cfg["graph_cache"])
    label_names = metadata.get("label_names", ["-1", "1"])

    print(f"[phase1_2] loading HGT checkpoint: {cfg['checkpoint_dir']}", flush=True)
    model, effective_cfg = build_and_load_hgt(
        data=data,
        checkpoint_dir=cfg["checkpoint_dir"],
        model_cfg=cfg.get("model"),
        section_gnn_root=cfg.get("section_gnn_root"),
        device=device,
    )
    with open(output_root / "effective_model_cfg.json", "w") as f:
        json.dump(effective_cfg, f, indent=2)

    data = data.to(device)

    print("[phase1_2] running frozen inference over entire graph...", flush=True)
    with torch.no_grad():
        logits, hidden = model(data.x_dict, data.edge_index_dict)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = probabilities.argmax(dim=-1)
        case_embeddings = hidden["case"]

    n_cases = int(case_embeddings.shape[0])
    embedding_dim = int(case_embeddings.shape[1])
    print(f"[phase1_2] n_cases={n_cases} embedding_dim={embedding_dim}", flush=True)

    y_np = data["case"].y.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()
    probs_np = probabilities.detach().cpu().numpy()
    emb_np = case_embeddings.detach().cpu().numpy().astype(np.float32)

    fold_predictions_csv = Path(cfg["checkpoint_dir"]) / "predictions.csv"
    if fold_predictions_csv.exists():
        print(f"[phase1_2] using fold-specific splits from {fold_predictions_csv}", flush=True)
        split_indices = load_fold_splits(str(fold_predictions_csv), data)
    else:
        print("[phase1_2] fold predictions.csv not found; falling back to graph-cache masks", flush=True)
        split_indices = get_split_indices(data)
    split_name_for = ["train"] * n_cases  # default
    for split_name, idxs in split_indices.items():
        for idx in idxs:
            split_name_for[idx] = split_name

    case_ids = list(data["case"].case_id)
    file_names = list(data["case"].file_name)
    raw_labels = list(data["case"].raw_label)

    print("[phase1_2] saving embeddings, predictions, metadata...", flush=True)
    np.save(output_root / "case_embeddings.npy", emb_np)
    np.save(output_root / "logits.npy", logits.detach().cpu().numpy())
    np.save(output_root / "probabilities.npy", probs_np)
    np.save(output_root / "predictions.npy", preds_np)
    np.save(output_root / "targets.npy", y_np)

    predictions_df = pd.DataFrame(
        {
            "node_index": list(range(n_cases)),
            "case_id": case_ids,
            "file_name": file_names,
            "raw_label": raw_labels,
            "split": split_name_for,
            "target_index": y_np.tolist(),
            "target_label": [label_names[i] for i in y_np.tolist()],
            "pred_index": preds_np.tolist(),
            "pred_label": [label_names[i] for i in preds_np.tolist()],
            "confidence": probs_np.max(axis=1).tolist(),
            "prob_class_0": probs_np[:, 0].tolist(),
            "prob_class_1": probs_np[:, 1].tolist() if probs_np.shape[1] > 1 else [0.0] * n_cases,
        }
    )
    predictions_df.to_csv(output_root / "predictions.csv", index=False)

    print("[phase1_2] building FAISS index over TRAIN embeddings...", flush=True)
    import faiss  # noqa: WPS433

    train_idx = np.asarray(split_indices.get("train", []), dtype=np.int64)
    if train_idx.size == 0:
        raise RuntimeError("No training indices found; cannot build retrieval index.")

    metric = str(cfg.get("faiss", {}).get("metric", "cosine")).lower()
    train_vecs = emb_np[train_idx].copy()
    if metric == "cosine":
        faiss.normalize_L2(train_vecs)
        index = faiss.IndexFlatIP(embedding_dim)
    else:
        index = faiss.IndexFlatL2(embedding_dim)
    index.add(train_vecs)

    faiss.write_index(index, str(output_root / "train_faiss.index"))
    np.save(output_root / "train_indices.npy", train_idx)
    if metric == "cosine":
        # also save normalized vectors so we can recompute similarities later if needed
        np.save(output_root / "train_embeddings_normalized.npy", train_vecs)

    summary = {
        "graph_cache": str(cfg["graph_cache"]),
        "checkpoint_dir": str(cfg["checkpoint_dir"]),
        "n_cases": n_cases,
        "embedding_dim": embedding_dim,
        "label_names": label_names,
        "effective_model_cfg": effective_cfg,
        "split_counts": {k: len(v) for k, v in split_indices.items()},
        "metric": metric,
    }
    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[phase1_2] done. artefacts written to {output_root}", flush=True)


if __name__ == "__main__":
    main()
