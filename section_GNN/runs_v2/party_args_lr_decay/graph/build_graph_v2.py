#!/usr/bin/env python3
"""
Build graph with party-argument sections included in the case-node text embedding.
Patches updated_graph.pipeline to use case_star_builder_v2 before building.
Does NOT modify any existing file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_SECTION_GNN = _THIS_DIR.parents[2]
_FINAL_GRAPH = _SECTION_GNN / "final_graph"
_UPDATED_GRAPH = _FINAL_GRAPH / "updated_graph"

for _p in (str(_THIS_DIR), str(_UPDATED_GRAPH), str(_FINAL_GRAPH), str(_SECTION_GNN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Patch the pipeline BEFORE it gets used anywhere
import updated_graph.pipeline as _pipeline
from case_star_builder_v2 import build_case_star_graph as _fixed_builder
_pipeline.build_case_star_graph = _fixed_builder

import torch
from src.utils.io import dump_json, dump_yaml, ensure_dir, load_yaml
from src.utils.logging_utils import configure_logger
from src.utils.seed import set_global_seed
from updated_graph.pipeline import build_graph_bundle, load_cleaned_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph with party-argument case-node text (v2).")
    parser.add_argument("--config", required=True)
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
    logger = configure_logger("build_graph_v2", log_dir=log_dir)

    cleaned_cases = load_cleaned_cases(cleaned_dir, limit=args.limit)
    logger.info("Loaded %d cleaned cases from %s", len(cleaned_cases), cleaned_dir)
    bundle = build_graph_bundle(cleaned_cases, cfg, logger=logger)

    effective_graph_cfg = bundle["metadata"].get("effective_graph_config", {})
    graph_cache_path = graph_cache_dir / str(effective_graph_cfg.get("cache_name", "party_args_graph.pt"))
    torch.save(bundle, graph_cache_path)
    logger.info("Graph saved to %s", graph_cache_path)

    audit_dir = ensure_dir(paths_cfg.get("audits_dir", output_dir / "audits"))
    dump_json(bundle["metadata"], audit_dir / "graph_metadata_v2.json")
    dump_yaml(effective_graph_cfg, audit_dir / "effective_graph_config_v2.yaml")
    logger.info("Done.")


if __name__ == "__main__":
    main()
