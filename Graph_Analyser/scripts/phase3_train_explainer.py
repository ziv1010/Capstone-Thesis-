#!/usr/bin/env python
"""Phase 3: Train a heterogeneous PGExplainer on top of the frozen HGT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analyser.loader import build_and_load_hgt, load_config, load_graph_cache  # noqa: E402
from analyser.hetero_pg_explainer import HeteroEdgeMaskerConfig, HeteroPGExplainer  # noqa: E402
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

    output_root = Path(cfg["output_root"]) / "phase3_explainer"
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[phase3] loading graph cache: {cfg['graph_cache']}", flush=True)
    data, _metadata = load_graph_cache(cfg["graph_cache"])
    model, _ = build_and_load_hgt(
        data=data,
        checkpoint_dir=cfg["checkpoint_dir"],
        model_cfg=cfg.get("model"),
        section_gnn_root=cfg.get("section_gnn_root"),
        device=device,
    )
    data = data.to(device)

    explainer_cfg = cfg.get("explainer", {})
    masker_cfg = HeteroEdgeMaskerConfig(
        hidden_dim=int(explainer_cfg.get("hidden_dim", 64)),
        temperature_start=float(explainer_cfg.get("temperature_start", 5.0)),
        temperature_end=float(explainer_cfg.get("temperature_end", 1.0)),
        coeff_size=float(explainer_cfg.get("coeff_size", 0.01)),
        coeff_ent=float(explainer_cfg.get("coeff_ent", 1.0e-4)),
    )

    explainer = HeteroPGExplainer(model=model, data=data, masker_cfg=masker_cfg, device=device)

    fold_predictions_csv = Path(cfg["checkpoint_dir"]) / "predictions.csv"
    if fold_predictions_csv.exists():
        split_indices = load_fold_splits(str(fold_predictions_csv), data)
    else:
        split_indices = get_split_indices(data)
    train_idx = split_indices.get("train", [])
    if not train_idx:
        raise RuntimeError("Missing train split indices; cannot train explainer.")
    print(f"[phase3] training on {len(train_idx)} train case nodes", flush=True)

    history = explainer.train(
        target_case_indices=train_idx,
        epochs=int(explainer_cfg.get("epochs", 30)),
        lr=float(explainer_cfg.get("lr", 3.0e-3)),
        cases_per_epoch=int(explainer_cfg.get("max_train_cases_per_epoch", 512)),
        log_every=1,
    )

    explainer.save(str(output_root / "explainer.pt"))
    with open(output_root / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_root / "explainer_cfg.json", "w") as f:
        json.dump(masker_cfg.__dict__, f, indent=2)

    print(f"[phase3] done. explainer saved to {output_root}", flush=True)


if __name__ == "__main__":
    main()
