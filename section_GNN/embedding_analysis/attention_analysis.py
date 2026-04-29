#!/usr/bin/env python3
"""
Attention and saliency analysis for the trained HeteroLegalOutcomeGNN.

Produces two complementary views:

1. p_rel heatmap (HGT type-level attention priors)
   HGTConv stores a learned scalar p_rel[edge_type] of shape (1, heads) per
   relation type per layer. This acts as a relation-type prior on the softmax
   attention before any node-level adjustment. Averaging over heads and layers
   gives a per-edge-type "base attention weight" that the model has learned.

2. Gradient saliency per node type
   For each case node, we compute the gradient of the correct-class logit
   w.r.t. the input feature tensor of each node type, then sum squared
   gradients over feature dimensions and average over all case nodes. This
   quantifies how sensitive the model output is to perturbations of each
   node type's features.

Outputs (written next to the embeddings .npz or to --output-dir):
  <run_name>_prel_heatmap.png          — per-layer p_rel heatmap
  <run_name>_prel_values.json          — raw p_rel values
  <run_name>_gradient_saliency.png     — bar chart of gradient saliency per node type
  <run_name>_gradient_saliency.json    — raw saliency values

Usage
-----
  python attention_analysis.py \\
      --config  <path/to/config.yaml> \\
      --model   <path/to/fold_00/model.pt> \\
      [--output-dir <dir>] [--run-name <name>] [--n-cases 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    p = argparse.ArgumentParser(description="Attention and gradient saliency analysis.")
    p.add_argument("--config",      required=True)
    p.add_argument("--model",       required=True)
    p.add_argument("--graph-cache", default=None)
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--run-name",    default=None)
    p.add_argument("--n-cases",     type=int, default=500,
                   help="Max case nodes to use for gradient saliency (default 500)")
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# p_rel extraction
# ─────────────────────────────────────────────────────────────

def extract_prel(model: HeteroLegalOutcomeGNN) -> dict[str, list[float]]:
    """Extract mean(abs(p_rel)) per edge type per layer → {edge_type: [layer0, layer1, ...]}"""
    edge_types_seen: set[str] = set()
    for conv in model.convs:
        for k in conv.p_rel.keys():
            edge_types_seen.add(k)

    result: dict[str, list[float]] = {et: [] for et in sorted(edge_types_seen)}
    for conv in model.convs:
        for et in sorted(edge_types_seen):
            if et in conv.p_rel:
                val = float(conv.p_rel[et].abs().mean().item())
            else:
                val = 0.0
            result[et].append(val)
    return result


def plot_prel_heatmap(prel: dict[str, list[float]], out_path: Path, run_name: str) -> None:
    edge_types = list(prel.keys())
    n_layers   = len(next(iter(prel.values())))
    matrix     = np.array([prel[et] for et in edge_types])  # (n_edge_types, n_layers)

    # Sort by mean value across layers
    order = np.argsort(matrix.mean(axis=1))[::-1]
    matrix = matrix[order]
    labels = [edge_types[i].replace("__", " → ") for i in order]

    fig_h = max(5, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(max(4, n_layers * 1.5), fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"Layer {i+1}" for i in range(n_layers)], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"HGT p_rel (type-level attention prior)\n{run_name}", fontsize=10)
    plt.colorbar(im, ax=ax, label="|p_rel| mean over heads")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[attn] p_rel heatmap → {out_path.name}")


# ─────────────────────────────────────────────────────────────
# Gradient saliency
# ─────────────────────────────────────────────────────────────

def compute_gradient_saliency(
    model: HeteroLegalOutcomeGNN,
    data,
    n_cases: int,
    device: torch.device,
) -> dict[str, float]:
    """
    For each node type, compute mean squared gradient of the correct-class
    case logit w.r.t. that node type's input features.
    Averaged over n_cases case nodes.
    """
    model.eval()
    y_all = data["case"].y
    n_total = y_all.shape[0]
    case_indices = torch.randperm(n_total)[:n_cases]

    saliency: dict[str, float] = {nt: 0.0 for nt in data.node_types}
    processed = 0

    # Enable grad on all node features
    x_dict_grad: dict[str, torch.Tensor] = {}
    for nt in data.node_types:
        t = data[nt].x.detach().clone().requires_grad_(True)
        x_dict_grad[nt] = t

    # Forward once to get all logits and grads in one pass
    logits, _ = model(x_dict_grad, data.edge_index_dict)  # (N_cases, C)
    y_case    = data["case"].y[case_indices]               # (n_cases,)
    # Sum the correct-class logits over selected case nodes
    target_logits = logits[case_indices, :]                # (n_cases, C)
    score = target_logits[torch.arange(len(case_indices)), y_case].sum()
    score.backward()

    for nt in data.node_types:
        g = x_dict_grad[nt].grad
        if g is not None:
            saliency[nt] = float((g ** 2).mean().item())

    # Zero out negligibly small values (node types not connected to any case)
    max_s = max(saliency.values()) if saliency else 1.0
    saliency = {nt: v for nt, v in saliency.items() if v > 1e-12 * max_s}
    return saliency


def plot_saliency_bar(saliency: dict[str, float], out_path: Path, run_name: str) -> None:
    items = sorted(saliency.items(), key=lambda x: x[1], reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    norm   = max(values) if values else 1.0
    values_norm = [v / norm for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.45)))
    colors = ["#1f77b4" if "text" not in k and k not in ("preamble", "facts", "arguments",
              "petitioner_arguments", "respondent_arguments", "other_lawyer_arguments")
              else "#ff7f0e" for k in labels]
    ax.barh(labels[::-1], values_norm[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Gradient saliency (normalised)", fontsize=10)
    ax.set_title(f"Node-type gradient saliency\n{run_name}\n(orange = text nodes, blue = entity nodes)", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[attn] saliency bar → {out_path.name}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_yaml(args.config)

    paths_cfg        = cfg.get("paths", {})
    graph_cache_path = Path(
        args.graph_cache
        or (Path(paths_cfg["graph_cache_dir"]) / str(cfg["graph"]["cache_name"]))
    )
    model_path = Path(args.model)
    run_name   = args.run_name or model_path.parent.name
    out_dir    = Path(args.output_dir) if args.output_dir else Path(paths_cfg["outputs_dir"]) / "embedding_analysis"
    ensure_dir(out_dir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[attn] device={device}  model={model_path.name}")

    bundle      = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    data        = bundle["data"].to(device)
    label_names = list(bundle["metadata"]["label_names"])

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    hidden_dim = int(state_dict["input_projections.case.weight"].shape[0])
    num_layers = sum(1 for k in state_dict if k.startswith("layer_norms.") and k.endswith(".case.weight"))
    model_cfg  = {**cfg.get("model", {}), "hidden_dim": hidden_dim, "num_layers": num_layers}

    input_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
    model = HeteroLegalOutcomeGNN(
        metadata=data.metadata(), input_dims=input_dims,
        out_dim=len(label_names), cfg=model_cfg,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── p_rel ─────────────────────────────────────────────────
    print("[attn] extracting p_rel ...")
    prel = extract_prel(model)
    plot_prel_heatmap(prel, out_dir / f"{run_name}_prel_heatmap.png", run_name)
    (out_dir / f"{run_name}_prel_values.json").write_text(json.dumps(prel, indent=2))

    # Print top-5 by mean p_rel
    top = sorted(prel.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]
    print("[attn] top edge types by mean |p_rel|:")
    for et, vals in top:
        print(f"  {et:50s}  {np.mean(vals):.4f}")

    # ── Gradient saliency ─────────────────────────────────────
    print(f"[attn] computing gradient saliency over {args.n_cases} cases ...")
    saliency = compute_gradient_saliency(model, data, n_cases=args.n_cases, device=device)
    plot_saliency_bar(saliency, out_dir / f"{run_name}_gradient_saliency.png", run_name)
    (out_dir / f"{run_name}_gradient_saliency.json").write_text(
        json.dumps({k: float(v) for k, v in sorted(saliency.items(), key=lambda x: x[1], reverse=True)}, indent=2)
    )
    print("[attn] done.")


if __name__ == "__main__":
    main()
