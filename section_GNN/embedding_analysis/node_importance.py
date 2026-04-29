#!/usr/bin/env python3
"""
Node-type importance via feature zeroing.

For each node type in the graph, sets that node type's feature matrix to all
zeros and measures the accuracy drop relative to the full model.  No retraining
— inference only on the test split of fold 0.

A large accuracy drop = that node type is important.
A small drop         = the model can compensate from other nodes.

Particularly informative to compare across two models:
  (a) standard model  — case node carries BGE-M3 text embedding
  (b) minimised model — case node carries only 12 scalar features

In (b), zeroing section nodes (facts, arguments, preamble) should hurt more
because the model has no other source of text signal.

Usage
-----
  python node_importance.py \\
      --config  <path/to/config.yaml> \\
      --model   <path/to/kfold/fold_00/model.pt> \\
      [--graph-cache <override .pt path>] \\
      [--run-name <label>] \\
      [--output-dir <dir>] \\
      [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hetero_gnn import HeteroLegalOutcomeGNN
from src.utils.io import ensure_dir, load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Node-type importance via feature zeroing.")
    p.add_argument("--config",      required=True)
    p.add_argument("--model",       required=True, help="Path to fold_00/model.pt")
    p.add_argument("--graph-cache", default=None)
    p.add_argument("--run-name",    default=None)
    p.add_argument("--output-dir",  default=None)
    p.add_argument("--device",      default="cuda")
    return p.parse_args()


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits[mask].argmax(dim=1)
    return float((preds == labels[mask]).float().mean().item())


def _macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds  = logits[mask].argmax(dim=1).cpu().numpy()
    truths = labels[mask].cpu().numpy()
    classes = np.unique(truths)
    f1s = []
    for c in classes:
        tp = int(((preds == c) & (truths == c)).sum())
        fp = int(((preds == c) & (truths != c)).sum())
        fn = int(((preds != c) & (truths == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def main() -> None:
    args  = parse_args()
    cfg   = load_yaml(args.config)
    paths = cfg.get("paths", {})

    graph_cache_path = Path(
        args.graph_cache
        or (Path(paths["graph_cache_dir"]) / str(cfg["graph"]["cache_name"]))
    )
    model_path = Path(args.model)
    run_name   = args.run_name or model_path.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else Path(paths["outputs_dir"]) / "embedding_analysis"
    ensure_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[node_imp] device={device}  run={run_name}")
    print(f"[node_imp] graph={graph_cache_path.name}")
    print(f"[node_imp] model={model_path.name}")

    # ── Load graph ────────────────────────────────────────────────────────────
    bundle     = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    data       = bundle["data"].to(device)
    metadata   = bundle["metadata"]
    label_names = list(metadata["label_names"])
    num_classes = len(label_names)

    labels     = data["case"].y
    test_mask  = data["case"].test_mask

    n_test = int(test_mask.sum().item())
    print(f"[node_imp] n_cases={data['case'].y.shape[0]}  n_test={n_test}  labels={label_names}")

    # ── Load model ────────────────────────────────────────────────────────────
    state = torch.load(model_path, map_location=device, weights_only=True)
    hidden_dim = int(state["input_projections.case.weight"].shape[0])
    num_layers = sum(1 for k in state if k.startswith("layer_norms.") and k.endswith(".case.weight"))
    model_cfg  = {**cfg.get("model", {}), "hidden_dim": hidden_dim, "num_layers": num_layers}

    input_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
    model = HeteroLegalOutcomeGNN(
        metadata   = data.metadata(),
        input_dims = input_dims,
        out_dim    = num_classes,
        cfg        = model_cfg,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"[node_imp] hidden_dim={model.hidden_dim}  num_layers={model.num_layers}")
    print(f"[node_imp] case node input dim = {input_dims.get('case', '?')}")

    # ── Baseline (all features intact) ───────────────────────────────────────
    with torch.no_grad():
        logits_full, _ = model(data.x_dict, data.edge_index_dict)
    baseline_acc = _accuracy(logits_full, labels, test_mask)
    baseline_f1  = _macro_f1(logits_full, labels, test_mask)
    print(f"\n[node_imp] BASELINE  acc={baseline_acc:.4f}  macro_f1={baseline_f1:.4f}")

    # ── Feature-zeroing per node type ────────────────────────────────────────
    results = []
    node_types = sorted(data.node_types)

    for nt in node_types:
        orig_x = data[nt].x.clone()
        data[nt].x = torch.zeros_like(orig_x)

        with torch.no_grad():
            logits_zeroed, _ = model(data.x_dict, data.edge_index_dict)

        acc = _accuracy(logits_zeroed, labels, test_mask)
        f1  = _macro_f1(logits_zeroed, labels, test_mask)
        delta_acc = acc - baseline_acc
        delta_f1  = f1  - baseline_f1

        results.append({
            "node_type": nt,
            "zeroed_acc": round(acc, 4),
            "zeroed_f1":  round(f1,  4),
            "delta_acc":  round(delta_acc, 4),
            "delta_f1":   round(delta_f1,  4),
            "importance_acc": round(-delta_acc, 4),  # positive = important
        })

        data[nt].x = orig_x  # restore
        print(f"  zero '{nt}': acc={acc:.4f}  Δacc={delta_acc:+.4f}  Δf1={delta_f1:+.4f}")

    # ── Rank by importance ────────────────────────────────────────────────────
    results.sort(key=lambda r: r["importance_acc"], reverse=True)

    print(f"\n[node_imp] === Ranked by importance (acc drop when zeroed) ===")
    print(f"  {'Node type':<35} {'Δacc':>8}  {'Δf1':>8}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}")
    for r in results:
        print(f"  {r['node_type']:<35} {r['delta_acc']:>+8.4f}  {r['delta_f1']:>+8.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out = {
        "run_name":     run_name,
        "baseline_acc": round(baseline_acc, 4),
        "baseline_f1":  round(baseline_f1,  4),
        "n_test":       n_test,
        "case_node_input_dim": input_dims.get("case"),
        "node_types_ranked": results,
    }
    json_path = output_dir / f"{run_name}_node_importance.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n[node_imp] saved → {json_path}")

    # ── Save bar chart ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels_plot = [r["node_type"].replace("_", "\n") for r in results]
        drops       = [-r["delta_acc"] for r in results]  # positive = bad when zeroed
        colors      = ["#d62728" if d > 0.005 else "#aec7e8" for d in drops]

        fig, ax = plt.subplots(figsize=(max(8, len(labels_plot) * 0.7), 5))
        ax.bar(range(len(labels_plot)), drops, color=colors, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, fontsize=7)
        ax.set_ylabel("Accuracy drop when node type is zeroed")
        ax.set_title(f"Node-type importance — {run_name}\n"
                     f"(baseline acc={baseline_acc:.4f}, case node dim={input_dims.get('case')})")
        plt.tight_layout()
        png_path = output_dir / f"{run_name}_node_importance.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[node_imp] chart → {png_path}")
    except Exception as e:
        print(f"[node_imp] chart skipped: {e}")


if __name__ == "__main__":
    main()
