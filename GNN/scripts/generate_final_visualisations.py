#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_json, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.pipeline import cleaned_case_from_dict
from src.visualization.final_visualizer import (
    save_final_edge_storage_view,
    save_final_layer1_view,
    save_final_layer2_view,
    save_final_node_storage_view,
    save_final_training_view,
    save_final_visualisations_readme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final explanatory visualisations for the current GNN.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn_case_star.yaml"))
    parser.add_argument(
        "--graph-cache",
        default=str(PROJECT_ROOT / "data" / "graph_cache" / "case_star_global_graph.pt"),
    )
    parser.add_argument(
        "--case-id",
        default="Abhijeet_Suryakant_Maske_And_Anr_vs_The_State_Of_Maharashtra_on_1_March_2022",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "visualizations" / "final_visualisations"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    logger = configure_logger(
        "generate_final_visualisations",
        log_dir=Path(cfg.get("paths", {}).get("outputs_dir", PROJECT_ROOT / "outputs")) / "logs",
    )

    bundle = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    data = bundle["data"]
    graph_metadata = bundle["metadata"]

    case_id = args.case_id
    cleaned_case_path = Path(cfg["paths"]["cleaned_case_dir"]) / f"{case_id}.json"
    if not cleaned_case_path.exists():
        fallback_case_id = str(data["case"].case_id[0])
        logger.warning("Requested case %s is not available in the graph. Falling back to %s", case_id, fallback_case_id)
        case_id = fallback_case_id
        cleaned_case_path = Path(cfg["paths"]["cleaned_case_dir"]) / f"{case_id}.json"

    cleaned_case = cleaned_case_from_dict(load_json(cleaned_case_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_storage_info = save_final_node_storage_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        cfg=cfg,
        graph_metadata=graph_metadata,
        output_path=output_dir / "01_what_is_stored_in_vertices.png",
    )
    save_final_edge_storage_view(
        output_path=output_dir / "02_what_is_stored_on_edges.png",
    )
    layer1_info = save_final_layer1_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        cfg=cfg,
        output_path=output_dir / "03_how_layer_1_updates_case_A.png",
    )
    layer2_info = save_final_layer2_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        data=data,
        case_id=case_id,
        output_path=output_dir / "04_how_layer_2_updates_case_A.png",
    )
    save_final_training_view(
        graph_metadata=graph_metadata,
        label_names=list(graph_metadata.get("label_names", [])),
        output_path=output_dir / "05_how_training_uses_labels.png",
    )
    save_final_visualisations_readme(
        output_path=output_dir / "README.md",
        selected_case_id=case_id,
        node_storage_info=node_storage_info,
        layer1_info=layer1_info,
        layer2_info=layer2_info,
        graph_metadata=graph_metadata,
    )
    logger.info("Saved final visualisations to %s", output_dir)


if __name__ == "__main__":
    main()
