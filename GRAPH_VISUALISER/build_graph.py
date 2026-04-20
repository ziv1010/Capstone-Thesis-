#!/usr/bin/env python3
"""
build_graph.py — Legal Case Graph Builder (GNN-Replicated)
===========================================================
Builds a heterogeneous NetworkX graph that replicates the GNN's star-graph
architecture from section_GNN/updated_graph/case_star_builder.py:

  Node types (matching GNN UPDATED_ENTITY_NODE_TYPES):
    case                    — one per case file
    statute, provision,
    precedent, court,
    judge, lawyer           — SHARED across cases (same entity = same node)
    petitioner, respondent  — LOCAL to each case

  Edge attributes:
    relation       — e.g. "has_statute", "has_court", "has_judge"
    in_arguments   — True if this entity appeared in an ARGUMENTS sentence
    bucket         — source bucket name

  New vs v1:
    • entity.seen_in_arguments  — True if entity cited in any ARGUMENTS section
    • entity.argument_case_count — # cases that cite it inside ARGUMENTS
    • case.connectivity_score   — Σ degree(shared_entity_neighbor)
    • case.connectivity_rank    — rank within bucket by connectivity score (1=best)
    • Outputs case_connections.pkl  — {(case_a, case_b): shared_entity_count}
    • Outputs case_layout.pkl       — spring layout of case-only network

Outputs written to output_dir (config.yaml):
  graph_full.pkl        Full NetworkX graph (all loaded cases)
  graph_sample.pkl      Connectivity-ranked display subgraph (≤max_display_nodes)
  graph_full.gexf       Gephi format
  layout.pkl            Spring layout for sample graph (entity + case nodes)
  case_layout.pkl       Spring layout for case-only network
  case_connections.pkl  Case-case connection weights {(key_a, key_b): count}
  stats.json            Hub analysis, bridge nodes, connectivity stats

Usage:
  python build_graph.py [--config config.yaml] [--limit N] [--skip-layout]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from tqdm import tqdm


# ── helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Uppercase, strip, collapse whitespace — matches GNN canonical form."""
    return re.sub(r"\s+", " ", text.strip().replace("\n", " ").replace("\r", "")).upper()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_case(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not load {path.name}: {e}", file=sys.stderr)
        return None


def is_case_json(path: Path) -> bool:
    """Skip directory-level metadata files that are not individual case payloads."""
    return path.name.lower() != "report.json"


def extract_entities(
    data: dict,
    shared_types: set,
    local_types: set,
    skip_types: set,
    normalize: bool,
):
    """
    Yield (entity_type, canonical_name, is_shared, seen_in_arguments) for every
    unique entity in the case JSON.

    seen_in_arguments = True if the entity appeared in at least one sentence
    whose rhetorical_role is ARGUMENTS.  Mirrors GNN entity.seen_in_arguments.
    """
    # (label, text) -> seen_in_arguments  (accumulate across sentences)
    seen: dict[tuple[str, str], bool] = {}

    for sent in data.get("sentences", []):
        role = sent.get("rhetorical_role", "").upper()
        in_args = role == "ARGUMENTS"

        for ent in sent.get("entities", []):
            label: str = ent.get("label", "")
            if label in skip_types:
                continue
            text: str = ent.get("text", "").replace("\n", " ").strip()
            if normalize:
                text = _norm(text)
            if not text or len(text) < 2:
                continue
            key = (label, text)
            # Mark True if this entity ever appears in an ARGUMENTS sentence
            seen[key] = seen.get(key, False) or in_args

    for (label, text), in_args in seen.items():
        if label not in shared_types and label not in local_types:
            continue
        yield label, text, (label in shared_types), in_args


# ── graph builder ─────────────────────────────────────────────────────────────

def build_graph(config: dict) -> nx.Graph:
    shared_types = set(config["graph"]["shared_node_types"])
    local_types  = set(config["graph"].get("local_node_types", []))
    skip_types   = set(config["graph"].get("skip_node_types", []))
    normalize    = bool(config["graph"].get("normalize_entities", True))

    G = nx.Graph()

    for bucket_cfg in config["buckets"]:
        bucket   = bucket_cfg["name"]
        color    = bucket_cfg["color"]
        data_dir = Path(bucket_cfg["data_dir"])
        limit    = bucket_cfg.get("limit")

        if not data_dir.exists():
            print(f"  [SKIP] {data_dir} not found — check config.yaml", file=sys.stderr)
            continue

        files = [fp for fp in sorted(data_dir.glob("*.json")) if is_case_json(fp)]
        if limit:
            files = files[:limit]

        print(f"\nBucket '{bucket}':  {len(files):,} files")

        for fp in tqdm(files, desc=f"  {bucket}", unit="case"):
            data = load_case(fp)
            if data is None:
                continue
            if "sentences" not in data:
                print(
                    f"  Warning: skipping non-case JSON without sentences: {fp.name}",
                    file=sys.stderr,
                )
                continue

            case_id  = data.get("file_id", fp.stem)
            case_key = f"case::{case_id}"

            G.add_node(
                case_key,
                node_type          = "case",
                bucket             = bucket,
                bucket_color       = color,
                label              = case_id[:80],
                outcome            = data.get("case_outcome_label", "unknown"),
                outcome_score      = data.get("case_outcome_score", None),
                # filled in by sample_for_display after scoring
                connectivity_score = 0,
                connectivity_rank  = 0,
                bucket_case_count  = 0,
            )

            for entity_type, canonical, is_shared, in_args in extract_entities(
                data, shared_types, local_types, skip_types, normalize
            ):
                ekey = (
                    f"{entity_type}::{canonical}"
                    if is_shared
                    else f"case::{case_id}::{entity_type}::{canonical}"
                )

                if ekey not in G:
                    G.add_node(
                        ekey,
                        node_type            = entity_type.lower(),
                        bucket               = "entity",
                        label                = canonical[:80],
                        is_shared            = is_shared,
                        entity_type          = entity_type,
                        seen_in_arguments    = in_args,
                        argument_case_count  = 0,   # updated below
                    )
                elif in_args and G.nodes[ekey].get("is_shared", False):
                    # propagate: if ANY case cites this entity in arguments, flag it
                    G.nodes[ekey]["seen_in_arguments"] = True

                G.add_edge(
                    case_key, ekey,
                    relation     = f"has_{entity_type.lower()}",
                    in_arguments = in_args,
                    bucket       = bucket,
                )

    # ── count argument citations per shared entity ─────────────────────────
    for node, d in G.nodes(data=True):
        if not d.get("is_shared", False):
            continue
        G.nodes[node]["argument_case_count"] = sum(
            1 for nb in G.neighbors(node)
            if G.nodes[nb].get("node_type") == "case"
            and G[nb][node].get("in_arguments", False)
        )

    print(f"\nGraph complete — {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


# ── connectivity scoring ──────────────────────────────────────────────────────

def compute_connectivity_scores(G: nx.Graph) -> dict[str, float]:
    """
    Score each case by Σ degree(shared_entity_neighbor).

    Cases connected to many high-degree hub entities score highest — they are
    the most "structurally central" cases for understanding the graph.
    Requires degree attributes already set on G (call after compute_stats).
    """
    scores: dict[str, float] = {}
    for node, d in G.nodes(data=True):
        if d.get("node_type") != "case":
            continue
        scores[node] = sum(
            G.nodes[nb].get("degree", G.degree(nb))
            for nb in G.neighbors(node)
            if G.nodes[nb].get("is_shared", False)
        )
    return scores


# ── statistics ────────────────────────────────────────────────────────────────

def compute_stats(G: nx.Graph) -> dict:
    print("Computing statistics …")

    degree_dict = dict(G.degree())
    nx.set_node_attributes(G, degree_dict, "degree")

    # node-type and bucket counts
    node_type_counts: dict[str, int] = defaultdict(int)
    bucket_counts:    dict[str, int] = defaultdict(int)
    for n, d in G.nodes(data=True):
        node_type_counts[d.get("node_type", "unknown")] += 1
        if d.get("node_type") == "case":
            bucket_counts[d.get("bucket", "unknown")] += 1

    # top-100 hub nodes with richer metadata
    top_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:100]
    top_hubs_info = []
    for k, deg in top_hubs:
        d = G.nodes[k]
        # buckets this entity bridges (case neighbors only)
        nbr_buckets: set[str] = set()
        bucket_case_counts: dict[str, int] = defaultdict(int)
        if d.get("is_shared", False):
            for nb in G.neighbors(k):
                if G.nodes[nb].get("node_type") == "case":
                    b = G.nodes[nb].get("bucket", "")
                    nbr_buckets.add(b)
                    bucket_case_counts[b] += 1
        top_hubs_info.append({
            "node"               : k,
            "label"              : d.get("label", k)[:80],
            "type"               : d.get("node_type", ""),
            "bucket"             : d.get("bucket", ""),
            "degree"             : deg,
            "argument_case_count": d.get("argument_case_count", 0),
            "buckets_bridged"    : sorted(nbr_buckets),
            "bucket_count"       : len(nbr_buckets),
            "bucket_case_counts" : dict(bucket_case_counts),
        })

    # cross-bucket bridge nodes: shared entities touching >= 2 different buckets
    bridge_nodes = []
    for node, d in G.nodes(data=True):
        if not d.get("is_shared", False):
            continue
        bucket_case_counts: dict[str, int] = defaultdict(int)
        for nb in G.neighbors(node):
            if G.nodes[nb].get("node_type") == "case":
                bucket_case_counts[G.nodes[nb].get("bucket", "")] += 1
        if len(bucket_case_counts) > 1:
            bridge_nodes.append({
                "node"               : node,
                "label"              : d.get("label", node)[:80],
                "entity_type"        : d.get("entity_type", ""),
                "buckets"            : sorted(bucket_case_counts.keys()),
                "degree"             : degree_dict.get(node, 0),
                "argument_case_count": d.get("argument_case_count", 0),
                "bucket_case_counts" : dict(bucket_case_counts),
            })
    bridge_nodes.sort(key=lambda x: x["degree"], reverse=True)

    degrees = list(degree_dict.values())
    hist, bin_edges = np.histogram(degrees, bins=20)

    return {
        "total_nodes"      : G.number_of_nodes(),
        "total_edges"      : G.number_of_edges(),
        "node_type_counts" : dict(node_type_counts),
        "bucket_counts"    : dict(bucket_counts),
        "avg_degree"       : round(float(np.mean(degrees)), 3) if degrees else 0,
        "max_degree"       : int(max(degrees)) if degrees else 0,
        "top_hubs"         : top_hubs_info,
        "bridge_nodes"     : bridge_nodes[:100],
        "bridge_node_count": len(bridge_nodes),
        "degree_histogram" : {
            "counts"   : hist.tolist(),
            "bin_edges": [round(x, 2) for x in bin_edges.tolist()],
        },
    }


# ── case-case connections ─────────────────────────────────────────────────────

def compute_case_connections(G: nx.Graph) -> dict[tuple[str, str], int]:
    """
    Compute pairwise case-case connection counts in G (sampled graph).

    Two cases are connected by +1 for each shared entity they both link to.
    Very high-degree entities (> 120 case neighbors) are skipped — they are
    too generic (e.g. "Supreme Court of India") to be informative bridges.

    Returns {(case_key_a, case_key_b): shared_entity_count}  (key_a < key_b).
    """
    print("Computing case-case connections …")
    connections: dict[tuple[str, str], int] = {}

    for entity_node, d in G.nodes(data=True):
        if not d.get("is_shared", False):
            continue
        case_nbrs = [
            n for n in G.neighbors(entity_node)
            if G.nodes[n].get("node_type") == "case"
        ]
        n_cases = len(case_nbrs)
        if n_cases < 2 or n_cases > 120:
            continue
        for i, ca in enumerate(case_nbrs):
            for cb in case_nbrs[i + 1:]:
                key = (min(ca, cb), max(ca, cb))
                connections[key] = connections.get(key, 0) + 1

    print(f"  Found {len(connections):,} case-case connection pairs")
    return connections


def get_top_case_connections(
    G: nx.Graph,
    connections: dict[tuple[str, str], int],
    n: int = 50,
) -> list[dict]:
    """Return top-N cross-bucket case pairs sorted by shared entity count."""
    result = []
    for (ca, cb), count in sorted(connections.items(), key=lambda x: x[1], reverse=True):
        if ca not in G or cb not in G:
            continue
        ba = G.nodes[ca].get("bucket", "")
        bb = G.nodes[cb].get("bucket", "")
        if ba != bb:
            result.append({
                "case_a"         : G.nodes[ca].get("label", ca)[:65],
                "case_b"         : G.nodes[cb].get("label", cb)[:65],
                "bucket_a"       : ba,
                "bucket_b"       : bb,
                "shared_entities": count,
            })
        if len(result) >= n:
            break
    return result


# ── sampling ──────────────────────────────────────────────────────────────────

def sample_for_display(G: nx.Graph, cfg: dict) -> nx.Graph:
    """
    Build a display subgraph:
      1. All hub nodes (degree >= min_degree_for_hub) are always included.
      2. Cases are ranked by connectivity_score within their bucket.
         Top cases_per_bucket cases per bucket are included.
      3. All shared-entity neighbours of included cases are pulled in.
      4. Hard cap at max_display_nodes (hubs first, then by degree).

    Sets connectivity_score, connectivity_rank, bucket_case_count on case nodes.
    (Requires degree attributes already set on G via compute_stats.)
    """
    sc             = cfg.get("sampling", {})
    max_nodes      = sc.get("max_display_nodes", 6000)
    min_hub_degree = sc.get("min_degree_for_hub", 8)
    cases_per_bkt  = sc.get("cases_per_bucket", 300)

    degree_dict = dict(G.degree())

    # ── connectivity scores ────────────────────────────────────────────────
    conn_scores = compute_connectivity_scores(G)

    # ── always-keep hub nodes ──────────────────────────────────────────────
    hub_nodes: set[str] = {n for n, deg in degree_dict.items() if deg >= min_hub_degree}

    # ── per-bucket connectivity ranking ───────────────────────────────────
    bucket_cases: dict[str, list[str]] = defaultdict(list)
    for n, d in G.nodes(data=True):
        if d.get("node_type") == "case":
            bucket_cases[d.get("bucket", "")].append(n)

    sampled_cases: set[str] = set()
    for bkt, cases in bucket_cases.items():
        ranked = sorted(cases, key=lambda n: conn_scores.get(n, 0), reverse=True)
        for rank, case_node in enumerate(ranked, 1):
            G.nodes[case_node]["connectivity_score"] = conn_scores.get(case_node, 0)
            G.nodes[case_node]["connectivity_rank"]  = rank
            G.nodes[case_node]["bucket_case_count"]  = len(ranked)
        k = min(cases_per_bkt, len(ranked))
        sampled_cases.update(ranked[:k])

    # ── include shared-entity neighbours of sampled cases ─────────────────
    included = hub_nodes | sampled_cases
    for cn in sampled_cases:
        for nb in G.neighbors(cn):
            if G.nodes[nb].get("is_shared", False):
                included.add(nb)

    # ── hard cap (hubs → shared entities → cases by degree) ───────────────
    if len(included) > max_nodes:
        hub_part = sorted(
            hub_nodes, key=lambda n: degree_dict.get(n, 0), reverse=True
        )[: max_nodes // 3]
        rest_part = sorted(
            included - hub_nodes, key=lambda n: degree_dict.get(n, 0), reverse=True
        )[: max_nodes - max_nodes // 3]
        included = set(hub_part + rest_part)

    sub = G.subgraph(included).copy()
    print(f"Sampled subgraph — {sub.number_of_nodes():,} nodes, {sub.number_of_edges():,} edges")
    return sub


# ── case-only network ─────────────────────────────────────────────────────────

def build_case_network(
    G: nx.Graph,
    connections: dict[tuple[str, str], int],
) -> nx.Graph:
    """
    Case-only graph where edges are weighted by shared entity count.
    Used to compute case_layout.pkl.
    """
    cG = nx.Graph()
    for node, d in G.nodes(data=True):
        if d.get("node_type") != "case":
            continue
        cG.add_node(
            node,
            label              = d.get("label", node),
            bucket             = d.get("bucket", ""),
            bucket_color       = d.get("bucket_color", "#BDC3C7"),
            connectivity_score = d.get("connectivity_score", 0),
            connectivity_rank  = d.get("connectivity_rank", 0),
            bucket_case_count  = d.get("bucket_case_count", 0),
            outcome            = d.get("outcome", ""),
            degree             = d.get("degree", 0),
        )
    for (ca, cb), count in connections.items():
        if ca in cG and cb in cG:
            cG.add_edge(ca, cb, weight=count, shared_entity_count=count)
    return cG


# ── layouts ───────────────────────────────────────────────────────────────────

def compute_layout(G: nx.Graph) -> dict[str, list[float]]:
    """Spring layout for the full sample (entity + case nodes)."""
    print(f"Computing spring layout for {G.number_of_nodes():,} nodes …")
    k = 2.0 / max(1, np.sqrt(G.number_of_nodes()))
    pos = nx.spring_layout(G, k=k, iterations=60, seed=42)
    return {node: list(map(float, xy)) for node, xy in pos.items()}


def compute_case_layout(cG: nx.Graph) -> dict[str, list[float]]:
    """Weighted spring layout for the case-only network."""
    n = cG.number_of_nodes()
    print(f"Computing case-network layout for {n:,} case nodes …")
    if n == 0:
        return {}
    k = 1.5 / max(1, np.sqrt(n))
    pos = nx.spring_layout(cG, k=k, weight="weight", iterations=60, seed=42)
    return {node: list(map(float, xy)) for node, xy in pos.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build legal case graph")
    p.add_argument("--config",      default="config.yaml",  help="Path to config.yaml")
    p.add_argument("--limit",       type=int, default=None, help="Override per-bucket file limit")
    p.add_argument("--skip-layout", action="store_true",    help="Skip layout computation")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    if args.limit:
        for b in config["buckets"]:
            b["limit"] = args.limit

    out_dir = Path(config.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build full graph ────────────────────────────────────────────────────
    G = build_graph(config)

    # 2. Stats on full graph (also sets degree attributes on G) ──────────────
    stats = compute_stats(G)

    # 3. GEXF (Gephi) ────────────────────────────────────────────────────────
    G_gexf = G.copy()
    for _, d in G_gexf.nodes(data=True):
        for k in list(d.keys()):
            if d[k] is None:
                d[k] = ""
    nx.write_gexf(G_gexf, str(out_dir / "graph_full.gexf"))
    print(f"Saved GEXF       → {out_dir / 'graph_full.gexf'}")

    # 4. Full graph pickle ───────────────────────────────────────────────────
    with open(out_dir / "graph_full.pkl", "wb") as f:
        pickle.dump(G, f)
    print(f"Saved full pkl   → {out_dir / 'graph_full.pkl'}")

    # 5. Sampled subgraph (connectivity-ranked) ──────────────────────────────
    G_sample = sample_for_display(G, config)
    with open(out_dir / "graph_sample.pkl", "wb") as f:
        pickle.dump(G_sample, f)
    print(f"Saved sample     → {out_dir / 'graph_sample.pkl'}")

    # 6. Case-case connections ───────────────────────────────────────────────
    connections = compute_case_connections(G_sample)
    with open(out_dir / "case_connections.pkl", "wb") as f:
        pickle.dump(connections, f)
    print(f"Saved case conn  → {out_dir / 'case_connections.pkl'}")

    # Enrich stats with case-connection data
    stats["case_connection_count"]   = len(connections)
    stats["cross_bucket_case_pairs"] = get_top_case_connections(G_sample, connections, n=50)

    # Top cases by connectivity score (from sample)
    case_nodes = [
        (n, d) for n, d in G_sample.nodes(data=True)
        if d.get("node_type") == "case"
    ]
    stats["top_cases_by_connectivity"] = [
        {
            "node"              : n,
            "label"             : d.get("label", n)[:80],
            "bucket"            : d.get("bucket", ""),
            "connectivity_score": d.get("connectivity_score", 0),
            "connectivity_rank" : d.get("connectivity_rank", 0),
            "degree"            : d.get("degree", 0),
        }
        for n, d in sorted(
            case_nodes,
            key=lambda x: x[1].get("connectivity_score", 0),
            reverse=True,
        )[:50]
    ]

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats      → {out_dir / 'stats.json'}")

    # 7. Layouts ─────────────────────────────────────────────────────────────
    if not args.skip_layout:
        layout = compute_layout(G_sample)
        with open(out_dir / "layout.pkl", "wb") as f:
            pickle.dump(layout, f)
        print(f"Saved layout     → {out_dir / 'layout.pkl'}")

        cG         = build_case_network(G_sample, connections)
        case_layout = compute_case_layout(cG)
        with open(out_dir / "case_layout.pkl", "wb") as f:
            pickle.dump(case_layout, f)
        print(f"Saved case layout→ {out_dir / 'case_layout.pkl'}")

    # 8. Summary ─────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Total nodes          : {stats['total_nodes']:,}")
    print(f"  Total edges          : {stats['total_edges']:,}")
    print(f"  Avg degree           : {stats['avg_degree']}")
    print(f"  Max degree           : {stats['max_degree']}")
    print(f"  Node types           : {stats['node_type_counts']}")
    print(f"  Buckets              : {stats['bucket_counts']}")
    print(f"  Cross-bucket bridges : {stats['bridge_node_count']}")
    print(f"  Case-case conn pairs : {stats['case_connection_count']:,}")
    print(f"\nTop-5 hubs:")
    for h in stats["top_hubs"][:5]:
        print(
            f"  [{h['type']:>10}] deg={h['degree']:>5}  "
            f"arg_citations={h['argument_case_count']:>4}  "
            f"buckets={h['bucket_count']}  {h['label'][:55]}"
        )
    print(f"\nTop-5 cases by connectivity:")
    for c in stats["top_cases_by_connectivity"][:5]:
        print(
            f"  [{c['bucket']:>20}] score={c['connectivity_score']:>6.0f}  "
            f"rank={c['connectivity_rank']:>3}  {c['label'][:50]}"
        )


if __name__ == "__main__":
    main()
