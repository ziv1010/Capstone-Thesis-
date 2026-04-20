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
from src.visualization.graph_visualizer import (
    save_case_representation_story_view,
    save_case_star_aggregated_view,
    save_cross_case_connection_story_view,
    save_layer_connection_story_view,
    save_local_case_star_view,
    save_message_passing_simple_view,
    save_model_flow_view,
    save_schema_view,
    save_schema_simple_view,
    save_shared_bridges_simple_view,
    save_two_layer_receptive_field_view,
    save_visualization_summary,
    save_story_readme,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the current heterogeneous case graph and GNN flow.")
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
        default=str(PROJECT_ROOT / "outputs" / "visualizations" / "current_graph"),
    )
    parser.add_argument(
        "--simplified-output-dir",
        default=str(PROJECT_ROOT / "outputs" / "visualizations" / "current_graph_simplified"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    logger = configure_logger(
        "visualize_graph",
        log_dir=Path(cfg.get("paths", {}).get("outputs_dir", PROJECT_ROOT / "outputs")) / "logs",
    )

    bundle = torch.load(args.graph_cache, map_location="cpu", weights_only=False)
    data = bundle["data"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    simplified_output_dir = Path(args.simplified_output_dir)
    simplified_output_dir.mkdir(parents=True, exist_ok=True)

    case_id = args.case_id
    cleaned_case_path = Path(cfg["paths"]["cleaned_case_dir"]) / f"{case_id}.json"
    if not cleaned_case_path.exists():
        fallback_case_id = str(data["case"].case_id[0])
        logger.warning("Requested case %s is not available in the current graph. Falling back to %s", case_id, fallback_case_id)
        case_id = fallback_case_id
        cleaned_case_path = Path(cfg["paths"]["cleaned_case_dir"]) / f"{case_id}.json"

    cleaned_case = cleaned_case_from_dict(load_json(cleaned_case_path))

    schema_path = output_dir / "graph_schema.png"
    case_star_path = output_dir / f"case_star_{case_id}.png"
    receptive_field_path = output_dir / f"receptive_field_2layer_{case_id}.png"
    model_flow_path = output_dir / "model_flow.png"
    summary_path = output_dir / "graph_visualization_summary.md"

    simple_schema_path = simplified_output_dir / "01_schema_simple.png"
    simple_case_star_path = simplified_output_dir / f"02_case_star_simple_{case_id}.png"
    simple_bridges_path = simplified_output_dir / f"03_shared_bridges_{case_id}.png"
    simple_message_path = simplified_output_dir / "04_message_passing_steps.png"
    story_case_path = simplified_output_dir / "01_one_case_representation.png"
    story_layer_path = simplified_output_dir / "02_how_two_hgt_layers_work.png"
    story_bridge_path = simplified_output_dir / "03_how_this_case_connects_to_other_cases.png"
    simple_readme_path = simplified_output_dir / "README.md"

    save_schema_view(data=data, output_path=schema_path)
    local_case_counts = save_local_case_star_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        output_path=case_star_path,
    )
    receptive_field_stats = save_two_layer_receptive_field_view(
        data=data,
        case_id=case_id,
        output_path=receptive_field_path,
    )
    save_model_flow_view(model_cfg=cfg.get("model", {}), output_path=model_flow_path)
    save_visualization_summary(
        output_path=summary_path,
        selected_case_id=case_id,
        node_counts={node_type: int(data[node_type].num_nodes) for node_type in data.node_types},
        local_case_counts=local_case_counts,
        receptive_field_stats=receptive_field_stats,
        model_cfg=cfg.get("model", {}),
    )
    save_schema_simple_view(data=data, output_path=simple_schema_path)
    _ = save_case_star_aggregated_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        output_path=simple_case_star_path,
    )
    bridge_stats = save_shared_bridges_simple_view(data=data, case_id=case_id, output_path=simple_bridges_path)
    save_message_passing_simple_view(case_id=case_id, bridge_stats=bridge_stats, output_path=simple_message_path)
    story_case_summary = save_case_representation_story_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        output_path=story_case_path,
    )
    story_layer_summary = save_layer_connection_story_view(
        cleaned_case=cleaned_case,
        graph_cfg=cfg.get("graph", {}),
        data=data,
        case_id=case_id,
        output_path=story_layer_path,
    )
    story_bridge_rows = save_cross_case_connection_story_view(
        data=data,
        case_id=case_id,
        output_path=story_bridge_path,
    )
    save_story_readme(
        output_path=simple_readme_path,
        selected_case_id=case_id,
        case_summary=story_case_summary,
        layer_summary=story_layer_summary,
        bridge_rows=story_bridge_rows,
        model_cfg=cfg.get("model", {}),
    )
    logger.info("Saved graph visualizations to %s", output_dir)
    logger.info("Saved simplified graph guide to %s", simplified_output_dir)


if __name__ == "__main__":
    main()
