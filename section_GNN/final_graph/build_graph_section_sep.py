#!/usr/bin/env python3
"""Build a graph bundle using section-separated case node embeddings.

Case node text feature = concat(preamble_emb, facts_emb, args_emb) → 3×D dims.
Graph structure and entity nodes are identical to the standard build pipeline.

Usage:
    python final_graph/build_graph_section_sep.py --config ablations/section_sep_enc/<bucket>/config.yaml
"""
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
from updated_graph.pipeline_section_sep import build_graph_bundle_section_sep
from src.utils.pipeline import load_cleaned_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a case graph with section-separated case node embeddings."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "ablations" / "section_sep_enc" / "fin_fraud_timed_mistral" / "config.yaml"),
    )
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
    logger = configure_logger("build_graph_section_sep", log_dir=log_dir)

    cleaned_cases = load_cleaned_cases(cleaned_dir, limit=args.limit)
    logger.info("Loaded %d cleaned cases from %s", len(cleaned_cases), cleaned_dir)

    bundle = build_graph_bundle_section_sep(cleaned_cases, cfg, logger=logger)

    graph_cache_path = graph_cache_dir / str(cfg.get("graph", {}).get("cache_name", "case_star_section_sep.pt"))
    torch.save(bundle, graph_cache_path)
    dump_json(bundle["metadata"], graph_cache_dir / "graph_metadata_section_sep.json")
    dump_json(bundle["metadata"].get("node_mappings", {}), graph_cache_dir / "node_mappings_section_sep.json")
    dump_json(bundle["metadata"].get("relation_mappings", []), graph_cache_dir / "relation_mappings_section_sep.json")
    dump_json(bundle["metadata"].get("split_assignments", {}), graph_cache_dir / "split_assignments_section_sep.json")
    dump_json(bundle["metadata"].get("debug_samples", []), graph_cache_dir / "graph_debug_samples_section_sep.json")
    dump_yaml(cfg, graph_cache_dir / "graph_config_snapshot_section_sep.yaml")
    logger.info(
        "Saved graph bundle to %s | case_feature_dim=%d | non_case_feature_dim=%d",
        graph_cache_path,
        bundle["metadata"].get("case_feature_dim", "?"),
        bundle["metadata"].get("feature_dim", "?"),
    )


if __name__ == "__main__":
    main()
