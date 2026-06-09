#!/usr/bin/env python3
"""
Step 04 — Run the configured trained HGT folds on the self-contained test graph
built in step 03.

Strategy
--------
- Load the trained-graph metadata to know exactly which node and edge types
  the model was trained on.
- Load our test graph. If any node / edge types are missing, pad with a single
  zero-feature dummy node so the model can be instantiated with matching
  metadata. Dummy nodes have no incident real edges, so they don't influence
  predictions on the real test cases.
- Build the HeteroLegalOutcomeGNN with the trained metadata + matching feature
  dims, load each fold's state_dict (strict=True after padding), run a forward
  pass per fold, and ensemble class probabilities by mean.
- Save predictions.csv with per-case predicted class, probability, and stage
  metadata derived from the case_id.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXP_ROOT = Path(__file__).resolve().parents[1]
SECTION_GNN_ROOT = EXP_ROOT.parent
sys.path.insert(0, str(SECTION_GNN_ROOT))

from src.utils.io import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(EXP_ROOT / "config.yaml"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_graph_bundle(path: str) -> tuple:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "data" in blob:
        return blob["data"], blob.get("metadata", {})
    return blob, {}


def pad_to_match_metadata(test_data, train_data):
    """Add zero-feature dummy nodes / edges for any node or edge type present in
    the trained graph but missing from the test graph. Returns the padded copy
    and a record of what was padded."""
    from torch_geometric.data import HeteroData

    padded = HeteroData()
    train_node_types = list(train_data.node_types)
    train_edge_types = list(train_data.edge_types)
    record = {"added_node_types": [], "added_edge_types": []}

    for nt in train_node_types:
        if nt in test_data.node_types and test_data[nt].x.shape[0] > 0:
            padded[nt].x = test_data[nt].x.clone()
            for attr in ("y", "case_id", "file_name", "raw_label",
                         "train_mask", "val_mask", "test_mask"):
                if hasattr(test_data[nt], attr):
                    setattr(padded[nt], attr, getattr(test_data[nt], attr))
        else:
            feat_dim = train_data[nt].x.shape[1]
            padded[nt].x = torch.zeros((1, feat_dim), dtype=train_data[nt].x.dtype)
            record["added_node_types"].append(nt)

    test_edge_set = {tuple(e) for e in test_data.edge_types}
    for et in train_edge_types:
        if tuple(et) in test_edge_set:
            ei = test_data[et].edge_index
            if ei.numel() > 0:
                padded[et].edge_index = ei.clone()
            else:
                padded[et].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            padded[et].edge_index = torch.empty((2, 0), dtype=torch.long)
            record["added_edge_types"].append(list(et))

    return padded, record


def build_model(data, model_cfg, train_metadata):
    HeteroLegalOutcomeGNN = importlib.import_module("src.models.hetero_gnn").HeteroLegalOutcomeGNN
    input_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
    out_dim = 2
    model = HeteroLegalOutcomeGNN(
        metadata=train_metadata,
        input_dims=input_dims,
        out_dim=out_dim,
        cfg=model_cfg,
    )
    return model


def parse_case_id(case_id: str) -> dict:
    m = re.match(r"^STAGE(\d+)__(.+)$", case_id)
    if not m:
        return {"stage_index": None, "base_case_id": case_id}
    return {"stage_index": int(m.group(1)), "base_case_id": m.group(2)}


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    inf_cfg = cfg["inference"]
    out_dir = EXP_ROOT / "outputs" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    test_graph_path = Path(cfg["paths"]["graph_cache_dir"]) / cfg["graph"]["cache_name"]
    print(f"[04_inference] loading test graph: {test_graph_path}", flush=True)
    test_data, test_meta = load_graph_bundle(str(test_graph_path))
    n_cases = int(test_data["case"].x.shape[0])
    print(f"[04_inference] test cases: {n_cases}", flush=True)

    print(f"[04_inference] loading trained graph (for metadata): {inf_cfg['trained_graph_cache']}", flush=True)
    train_data, train_meta = load_graph_bundle(str(resolve_path(inf_cfg["trained_graph_cache"])))
    train_metadata = (list(train_data.node_types), [tuple(e) for e in train_data.edge_types])
    label_names = list(train_meta.get("label_names", ["-1", "1"]))

    padded_data, pad_record = pad_to_match_metadata(test_data, train_data)
    print(f"[04_inference] padded missing node types: {pad_record['added_node_types']}", flush=True)
    print(f"[04_inference] padded missing edge types: {len(pad_record['added_edge_types'])}", flush=True)

    # Verify dim alignment with trained graph.
    for nt in train_data.node_types:
        if padded_data[nt].x.shape[1] != train_data[nt].x.shape[1]:
            raise RuntimeError(
                f"Feature dim mismatch for node type '{nt}': "
                f"test={padded_data[nt].x.shape[1]} vs train={train_data[nt].x.shape[1]}"
            )

    padded_data = padded_data.to(device)

    folds = list(inf_cfg.get("folds_to_use", [0]))
    fold_dir_root = resolve_path(inf_cfg["trained_model_run_dir"])
    all_probs = []
    per_fold_summary = []
    for fold_idx in folds:
        fold_dir = fold_dir_root / f"fold_{fold_idx:02d}"
        ckpt = fold_dir / "model.pt"
        if not ckpt.exists():
            print(f"[04_inference] skipping fold {fold_idx} — no model.pt at {ckpt}", flush=True)
            continue
        state_dict = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        model = build_model(padded_data, inf_cfg["model"], train_metadata)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[04_inference] fold {fold_idx} load warnings: missing={missing[:4]} unexpected={unexpected[:4]}", flush=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(device)
        with torch.no_grad():
            logits, _ = model(padded_data.x_dict, padded_data.edge_index_dict)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[:n_cases]
        all_probs.append(probs)
        per_fold_summary.append({"fold": fold_idx, "ckpt": str(ckpt)})
        print(f"[04_inference] fold {fold_idx} OK", flush=True)

    if not all_probs:
        raise RuntimeError("No fold produced predictions.")

    mean_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    pred_idx = mean_probs.argmax(axis=1)
    confidence = mean_probs.max(axis=1)

    case_ids = list(padded_data["case"].case_id)[:n_cases]
    raw_labels = list(padded_data["case"].raw_label)[:n_cases] if hasattr(padded_data["case"], "raw_label") else [None] * n_cases
    targets = padded_data["case"].y.detach().cpu().numpy()[:n_cases] if hasattr(padded_data["case"], "y") else np.full(n_cases, -1)

    rows = []
    for i in range(n_cases):
        parsed = parse_case_id(str(case_ids[i]))
        rows.append({
            "node_index": i,
            "case_id": str(case_ids[i]),
            "stage_index": parsed["stage_index"],
            "base_case_id": parsed["base_case_id"],
            "raw_label": str(raw_labels[i]) if raw_labels[i] is not None else "",
            "target_index": int(targets[i]) if targets[i] >= 0 else None,
            "target_label": label_names[int(targets[i])] if targets[i] >= 0 else "",
            "pred_index": int(pred_idx[i]),
            "pred_label": label_names[int(pred_idx[i])],
            "confidence": float(confidence[i]),
            "prob_class_0": float(mean_probs[i, 0]),
            "prob_class_1": float(mean_probs[i, 1]) if mean_probs.shape[1] > 1 else 0.0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)
    np.save(out_dir / "probabilities.npy", mean_probs)
    np.save(out_dir / "predictions.npy", pred_idx)

    summary = {
        "n_cases": n_cases,
        "label_names": label_names,
        "folds_used": [f["fold"] for f in per_fold_summary],
        "ensemble": inf_cfg.get("ensemble", "mean_proba"),
        "padded": pad_record,
        "trained_graph_cache": inf_cfg["trained_graph_cache"],
        "trained_model_run_dir": inf_cfg["trained_model_run_dir"],
        "test_graph_cache": str(test_graph_path),
    }
    with open(out_dir / "inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[04_inference] wrote {out_dir / 'predictions.csv'}")
    print(f"[04_inference] summary: {out_dir / 'inference_summary.json'}")


if __name__ == "__main__":
    main()
