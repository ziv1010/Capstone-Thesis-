#!/usr/bin/env python3
"""
Extract pre-GNN and post-GNN case node embeddings from a trained HeteroLegalOutcomeGNN.

Saves a compressed numpy archive (<run_name>_embeddings.npz) containing:
  case_ids    : (N,)  str  — raw case_id strings
  domains     : (N,)  str  — domain prefix parsed from case_id (e.g. "family_matrimonial")
  labels      : (N,)  int  — ground-truth class index (0 = dismissed, 1 = allowed)
  splits      : (N,)  str  — "train" / "val" / "test" for the fold used
  pre_gnn     : (N, D_in)  — raw input features (text + scalar, before input_projection)
  post_gnn    : (N, D_h)   — hidden["case"] after all HGT message-passing layers

Usage
-----
  python extract_embeddings.py \\
      --config  <path/to/config.yaml> \\
      --model   <path/to/fold_00/model.pt> \\
      --output  <path/to/output_dir>

  # optional: override graph cache
  python extract_embeddings.py --config ... --model ... --graph-cache /path/to/graph.pt

The script is intentionally standalone — it adds PROJECT_ROOT to sys.path so it
shares the same src/ tree as the main training pipeline.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hetero_gnn import HeteroLegalOutcomeGNN
from src.utils.io import ensure_dir, load_yaml


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract pre/post-GNN embeddings from a trained model.")
    p.add_argument("--config",      required=True, help="YAML config used during training")
    p.add_argument("--model",       required=True, help="Path to model.pt checkpoint (state_dict)")
    p.add_argument("--graph-cache", default=None,  help="Override graph cache .pt path")
    p.add_argument("--output",        default=None,  help="Directory to write <run_name>_embeddings.npz")
    p.add_argument("--run-name",      default=None,  help="Label for the output file (defaults to model dir name)")
    p.add_argument("--device",        default="cuda", help="'cuda' or 'cpu'")
    p.add_argument("--predictions-csv", default=None,
                   help="Path to predictions.csv from the trained fold. When provided, "
                        "split labels are read from there (train/val/test) rather than "
                        "from the graph bundle's stored masks, ensuring the embedding "
                        "analysis uses the same fold split as the trained model.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_DOMAIN_PATTERN = re.compile(r"^(.+?)__")


def _parse_domain(case_id: str) -> str:
    m = _DOMAIN_PATTERN.match(case_id)
    if not m:
        return "unknown"
    return m.group(1).replace("_timed_mistral", "").replace("_", " ")


def _build_split_array(train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> np.ndarray:
    n = len(train_mask)
    splits = np.full(n, "none", dtype=object)
    splits[train_mask.numpy()] = "train"
    splits[val_mask.numpy()] = "val"
    splits[test_mask.numpy()] = "test"
    return splits.astype(str)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    paths_cfg = cfg.get("paths", {})
    graph_cache_path = Path(
        args.graph_cache
        or (Path(paths_cfg["graph_cache_dir"]) / str(cfg["graph"]["cache_name"]))
    )
    model_path = Path(args.model)
    run_name   = args.run_name or model_path.parent.name

    output_dir = Path(args.output) if args.output else Path(paths_cfg["outputs_dir"]) / "embedding_analysis"
    ensure_dir(output_dir)
    out_path = output_dir / f"{run_name}_embeddings.npz"

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[extract] device={device}  graph={graph_cache_path.name}  model={model_path.name}")

    # ── Load graph bundle ─────────────────────────────────────
    bundle = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    data     = bundle["data"].to(device)
    metadata = bundle["metadata"]
    label_names = list(metadata["label_names"])
    num_classes = len(label_names)
    print(f"[extract] n_cases={data['case'].y.shape[0]}  labels={label_names}")

    # ── Reconstruct model — infer architecture from checkpoint ───
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Read hidden_dim and num_layers directly from the saved weights so the
    # model is rebuilt to match the checkpoint, regardless of what the config says.
    hidden_dim = int(state_dict["input_projections.case.weight"].shape[0])
    num_layers = sum(1 for k in state_dict if k.startswith("layer_norms.") and k.endswith(".case.weight"))

    model_cfg = {**cfg.get("model", {}), "hidden_dim": hidden_dim, "num_layers": num_layers}
    if model_cfg.get("hidden_dim") != cfg.get("model", {}).get("hidden_dim") or \
       model_cfg.get("num_layers") != cfg.get("model", {}).get("num_layers"):
        print(f"[extract] config says hidden_dim={cfg.get('model',{}).get('hidden_dim')} "
              f"num_layers={cfg.get('model',{}).get('num_layers')} — "
              f"overriding from checkpoint: hidden_dim={hidden_dim}  num_layers={num_layers}")

    input_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
    model = HeteroLegalOutcomeGNN(
        metadata   = data.metadata(),
        input_dims = input_dims,
        out_dim    = num_classes,
        cfg        = model_cfg,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[extract] model loaded — hidden_dim={model.hidden_dim}  num_layers={model.num_layers}")

    # ── Forward pass — extract pre and post-GNN ───────────────
    with torch.no_grad():
        pre_gnn_tensor = data["case"].x.clone()  # raw input features, shape (N, D_in)
        _, hidden      = model(data.x_dict, data.edge_index_dict)
        post_gnn_tensor = hidden["case"]          # shape (N, D_h)

    pre_gnn  = pre_gnn_tensor.detach().cpu().numpy().astype(np.float32)
    post_gnn = post_gnn_tensor.detach().cpu().numpy().astype(np.float32)

    # ── Build metadata arrays ─────────────────────────────────
    case_ids = np.array(list(data["case"].case_id), dtype=str)
    domains  = np.array([_parse_domain(cid) for cid in case_ids], dtype=str)
    labels   = data["case"].y.cpu().numpy().astype(np.int32)

    if args.predictions_csv:
        pred_df = pd.read_csv(args.predictions_csv).set_index("case_id")["split"]
        splits  = np.array([pred_df.get(cid, "none") for cid in case_ids], dtype=str)
        n_none  = (splits == "none").sum()
        if n_none:
            print(f"[extract] WARNING: {n_none} case(s) not found in predictions CSV — labelled 'none'")
        print(f"[extract] splits from predictions CSV: "
              f"train={( splits=='train').sum()}  val={(splits=='val').sum()}  test={(splits=='test').sum()}")
    else:
        print("[extract] WARNING: no --predictions-csv supplied; using graph bundle masks, "
              "which may not match the trained fold split. Pass --predictions-csv for accuracy.")
        train_mask = data["case"].train_mask.cpu()
        val_mask   = data["case"].val_mask.cpu()
        test_mask  = data["case"].test_mask.cpu()
        splits     = _build_split_array(train_mask, val_mask, test_mask)

    # ── Save ──────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        case_ids  = case_ids,
        domains   = domains,
        labels    = labels,
        splits    = splits,
        pre_gnn   = pre_gnn,
        post_gnn  = post_gnn,
        label_names = np.array(label_names, dtype=str),
    )
    print(f"[extract] saved → {out_path}")
    print(f"[extract] pre_gnn shape={pre_gnn.shape}  post_gnn shape={post_gnn.shape}")
    print(f"[extract] label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    unique_domains, domain_counts = np.unique(domains, return_counts=True)
    for d, c in zip(unique_domains, domain_counts):
        print(f"[extract]   domain '{d}': {c} cases")


if __name__ == "__main__":
    main()
