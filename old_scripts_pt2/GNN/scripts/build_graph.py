#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.pipeline import build_graph_bundle, load_cleaned_cases
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a leakage-safe case star graph in PyG HeteroData format.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gnn_case_star.yaml"))
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
    logger = configure_logger("build_graph", log_dir=log_dir)

    cleaned_cases = load_cleaned_cases(cleaned_dir, limit=args.limit)
    logger.info("Loaded %d cleaned cases from %s", len(cleaned_cases), cleaned_dir)
    bundle = build_graph_bundle(cleaned_cases, cfg, logger=logger)

    graph_cache_path = graph_cache_dir / str(cfg.get("graph", {}).get("cache_name", "case_star_graph.pt"))
    torch.save(bundle, graph_cache_path)
    dump_json(bundle["metadata"], graph_cache_dir / "graph_metadata.json")
    dump_json(bundle["metadata"].get("node_mappings", {}), graph_cache_dir / "node_mappings.json")
    dump_json(bundle["metadata"].get("relation_mappings", []), graph_cache_dir / "relation_mappings.json")
    dump_json(bundle["metadata"].get("split_assignments", {}), graph_cache_dir / "split_assignments.json")
    dump_json(bundle["metadata"].get("debug_samples", []), graph_cache_dir / "graph_debug_samples.json")
    dump_yaml(cfg, graph_cache_dir / "graph_config_snapshot.yaml")
    logger.info("Saved graph bundle to %s", graph_cache_path)


if __name__ == "__main__":
    main()
