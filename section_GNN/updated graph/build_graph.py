#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

UPDATED_GRAPH_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UPDATED_GRAPH_ROOT.parent
for path in (UPDATED_GRAPH_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed
from updated_graph.pipeline import build_graph_bundle, load_cleaned_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reasoning-focused case graph variant without shortcut context edges."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn_case_star_augmented_jsons_3way.yaml"))
    parser.add_argument("--cleaned-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_global_seed(int(cfg.get("project", {}).get("seed", 42)))

    paths_cfg = cfg.get("paths", {})
    cleaned_dir = Path(args.cleaned_dir or paths_cfg.get("cleaned_case_dir"))
    graph_cache_dir = ensure_dir(paths_cfg.get("graph_cache_dir"))
    output_dir = ensure_dir(paths_cfg.get("outputs_dir"))
    log_dir = ensure_dir(output_dir / "logs")
    logger = configure_logger("build_reasoning_graph", log_dir=log_dir)

    cleaned_cases = load_cleaned_cases(cleaned_dir, limit=args.limit)
    logger.info("Loaded %d cleaned cases from %s", len(cleaned_cases), cleaned_dir)
    bundle = build_graph_bundle(cleaned_cases, cfg, logger=logger)

    effective_graph_cfg = bundle["metadata"].get("effective_graph_config", {})
    graph_cache_path = graph_cache_dir / str(effective_graph_cfg.get("cache_name", "reasoning_focused_case_star_graph.pt"))
    torch.save(bundle, graph_cache_path)
    dump_json(bundle["metadata"], graph_cache_dir / "graph_metadata.reasoning_focused.json")
    dump_json(bundle["metadata"].get("node_mappings", {}), graph_cache_dir / "node_mappings.reasoning_focused.json")
    dump_json(bundle["metadata"].get("relation_mappings", []), graph_cache_dir / "relation_mappings.reasoning_focused.json")
    dump_json(bundle["metadata"].get("split_assignments", {}), graph_cache_dir / "split_assignments.reasoning_focused.json")
    dump_json(bundle["metadata"].get("debug_samples", []), graph_cache_dir / "graph_debug_samples.reasoning_focused.json")
    dump_yaml(cfg, graph_cache_dir / "graph_config_snapshot.reasoning_focused.yaml")
    dump_yaml(effective_graph_cfg, graph_cache_dir / "effective_graph_config.reasoning_focused.yaml")
    logger.info("Saved reasoning-focused graph bundle to %s", graph_cache_path)


if __name__ == "__main__":
    main()
