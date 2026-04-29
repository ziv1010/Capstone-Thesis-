from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml

SECTION_GNN_ROOT_DEFAULT = "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN"


def ensure_section_gnn_on_path(section_gnn_root: str | Path) -> None:
    root = str(section_gnn_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_graph_cache(graph_cache_path: str | Path) -> tuple[Any, dict[str, Any]]:
    blob = torch.load(str(graph_cache_path), map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        data = blob["data"]
        metadata = blob.get("metadata", {})
    else:
        data = blob
        metadata = {}
    return data, metadata


def infer_model_cfg_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    hidden_dim = None
    num_layers = 0
    num_heads = 4
    for key, value in state_dict.items():
        if key.startswith("input_projections.") and key.endswith(".weight"):
            hidden_dim = int(value.shape[0])
        if key.startswith("convs."):
            idx = int(key.split(".")[1])
            num_layers = max(num_layers, idx + 1)
    cfg = {
        "architecture": "hgt",
        "hidden_dim": hidden_dim or 128,
        "num_layers": num_layers or 2,
        "num_heads": num_heads,
        "dropout": 0.2,
        "mlp_hidden_dim": hidden_dim or 128,
    }
    return cfg


def build_and_load_hgt(
    data: Any,
    checkpoint_dir: str | Path,
    model_cfg: dict[str, Any] | None,
    section_gnn_root: str | Path = SECTION_GNN_ROOT_DEFAULT,
    device: torch.device | str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    ensure_section_gnn_on_path(section_gnn_root)
    # section_GNN exposes its own `src` package on sys.path (not this project's).
    import importlib

    HeteroLegalOutcomeGNN = importlib.import_module("src.models.hetero_gnn").HeteroLegalOutcomeGNN  # noqa: WPS433

    checkpoint_dir = Path(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)

    cfg = dict(model_cfg or {})
    inferred = infer_model_cfg_from_state_dict(state_dict)
    for key, value in inferred.items():
        cfg.setdefault(key, value)

    input_dims = {node_type: int(data[node_type].x.shape[1]) for node_type in data.node_types}
    out_dim = 2
    y = getattr(data["case"], "y", None)
    if y is not None:
        uniq = y.unique().tolist()
        out_dim = max(2, int(max(uniq)) + 1)

    model = HeteroLegalOutcomeGNN(
        metadata=data.metadata(),
        input_dims=input_dims,
        out_dim=out_dim,
        cfg=cfg,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading HGT state_dict: {missing[:8]}")
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in state_dict (model cfg mismatch?): {unexpected[:8]}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(device)
    return model, cfg
