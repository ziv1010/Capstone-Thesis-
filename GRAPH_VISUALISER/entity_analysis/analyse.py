"""
Entity Co-occurrence Network Analysis
Thesis: Within-bucket and Cross-bucket analysis of legal case entities.

Builds co-occurrence graphs at the case level for legally meaningful entity types,
then computes centrality metrics and outcome correlations.
"""

import json
import os
import glob
import sys
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import networkx as nx

# ── config ────────────────────────────────────────────────────────────────────

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parent
BASE_DIR = REPO_ROOT / "DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

BUCKETS = [
    "family_matrimonial",
    "land_property",
    "motor_accidents",
    "sexual_offences",
    "fin_fraud",
]

# Entity types that carry legal signal (exclude DATE, OTHER_PERSON as too noisy)
FOCUS_TYPES = {"STATUTE", "PROVISION", "JUDGE", "COURT", "LAWYER", "PRECEDENT", "GPE"}

OUTCOME_SCORE = {
    "appellant_won": 1,
    "appellant_lost": -1,
    "partially_allowed": 0,
    "unknown": None,
    None: None,
}

MAX_FILES = None  # set to e.g. 500 per bucket for a quick test run


# ── helpers ───────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_case(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_entities(case: dict) -> list[tuple[str, str]]:
    """Return list of (normalised_text, label) for focus entity types."""
    seen = set()
    result = []
    for sent in case.get("sentences", []):
        for ent in sent.get("entities", []):
            lbl = ent.get("label", "")
            if lbl not in FOCUS_TYPES:
                continue
            key = (normalise(ent["text"]), lbl)
            if key not in seen:
                seen.add(key)
                result.append(key)
    return result


def outcome_label(case: dict) -> str | None:
    return case.get("case_outcome_label")


# ── graph builder ─────────────────────────────────────────────────────────────

def build_graph(files: list[str], bucket_tag: str | None = None):
    """
    Process all files and return a NetworkX graph plus per-node stats.

    Each node  = (entity_text, entity_label)
    Each edge  = entities co-occur in the same case
    Edge/node weights accumulate across cases.
    """
    G = nx.Graph()

    # node extras: outcome lists, bucket sets
    node_outcomes: dict[tuple, list] = defaultdict(list)
    node_buckets:  dict[tuple, set]  = defaultdict(set)
    node_cases:    dict[tuple, int]  = defaultdict(int)

    total = len(files)
    for i, path in enumerate(files):
        if i % 2000 == 0:
            print(f"  [{i}/{total}] processing…", flush=True)

        case = load_case(path)
        if case is None:
            continue

        ents = extract_entities(case)
        if not ents:
            continue

        # determine source bucket
        fname = os.path.basename(path)
        bkt = bucket_tag
        if bkt is None:
            # cross-bucket mode: infer from filename prefix (bucket__casename.json)
            bkt = fname.split("__")[0] if "__" in fname else "unknown"

        oc = outcome_label(case)

        # register nodes
        for key in ents:
            if not G.has_node(key):
                G.add_node(key, text=key[0], label=key[1], freq=0, buckets=[])
            G.nodes[key]["freq"] += 1
            node_outcomes[key].append(oc)
            node_buckets[key].add(bkt)
            node_cases[key] += 1

        # register co-occurrence edges
        for a, b in combinations(ents, 2):
            if a == b:
                continue
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    # attach per-node outcome stats
    for node in G.nodes:
        oc_list = [OUTCOME_SCORE.get(o) for o in node_outcomes[node]]
        oc_vals = [v for v in oc_list if v is not None]
        G.nodes[node]["outcome_win_rate"]  = (
            sum(1 for v in oc_vals if v > 0) / len(oc_vals) if oc_vals else None
        )
        G.nodes[node]["outcome_mean_score"] = (
            sum(oc_vals) / len(oc_vals) if oc_vals else None
        )
        G.nodes[node]["bucket_count"]  = len(node_buckets[node])
        G.nodes[node]["buckets"]       = sorted(node_buckets[node])
        G.nodes[node]["case_count"]    = node_cases[node]

    return G


# ── centrality ────────────────────────────────────────────────────────────────

def compute_centrality(G: nx.Graph) -> dict:
    if len(G) == 0:
        return {}

    print("  computing degree centrality…", flush=True)
    degree = nx.degree_centrality(G)

    print("  computing weighted degree (strength)…", flush=True)
    strength = {n: sum(d["weight"] for _, d in G[n].items()) for n in G.nodes}

    print("  computing PageRank…", flush=True)
    pagerank = nx.pagerank(G, weight="weight", max_iter=300)

    # betweenness is O(VE) — skip for very large graphs
    betweenness = {}
    if len(G) <= 5000:
        print("  computing betweenness centrality…", flush=True)
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    return {
        "degree":      degree,
        "strength":    strength,
        "pagerank":    pagerank,
        "betweenness": betweenness,
    }


# ── serialise ─────────────────────────────────────────────────────────────────

def node_key(node: tuple) -> str:
    return f"{node[0]}  [{node[1]}]"


def top_n(scores: dict, n: int = 50) -> list[dict]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"entity": node_key(k), "text": k[0], "type": k[1], "score": v}
            for k, v in ranked]


def serialise_graph(G: nx.Graph, centrality: dict, bucket_name: str) -> dict:
    nodes_out = []
    for node, data in G.nodes(data=True):
        entry = {
            "entity":             node_key(node),
            "text":               data["text"],
            "type":               data["label"],
            "frequency":          data.get("freq", 0),
            "case_count":         data.get("case_count", 0),
            "degree_centrality":  centrality["degree"].get(node, 0),
            "strength":           centrality["strength"].get(node, 0),
            "pagerank":           centrality["pagerank"].get(node, 0),
            "betweenness":        centrality["betweenness"].get(node, 0),
            "outcome_win_rate":   data.get("outcome_win_rate"),
            "outcome_mean_score": data.get("outcome_mean_score"),
            "bucket_count":       data.get("bucket_count", 1),
            "buckets":            data.get("buckets", []),
        }
        nodes_out.append(entry)

    # sort by pagerank for default view
    nodes_out.sort(key=lambda x: x["pagerank"], reverse=True)

    edges_out = [
        {"source": node_key(u), "target": node_key(v), "weight": d["weight"]}
        for u, v, d in G.edges(data=True)
    ]
    edges_out.sort(key=lambda x: x["weight"], reverse=True)

    top50_pr       = top_n(centrality["pagerank"], 50)
    top50_strength = top_n(centrality["strength"], 50)
    top50_degree   = top_n(centrality["degree"], 50)
    top50_btwn     = top_n(centrality["betweenness"], 50) if centrality["betweenness"] else []

    # cross-bucket bridging: entities appearing in most buckets
    bridge_scores = {n: G.nodes[n].get("bucket_count", 1) for n in G.nodes}
    top50_bridge  = top_n(bridge_scores, 50)

    return {
        "bucket":          bucket_name,
        "node_count":      G.number_of_nodes(),
        "edge_count":      G.number_of_edges(),
        "top_by_pagerank": top50_pr,
        "top_by_strength": top50_strength,
        "top_by_degree":   top50_degree,
        "top_by_betweenness": top50_btwn,
        "top_by_bridge":   top50_bridge,
        "nodes":           nodes_out,
        "edges":           edges_out[:5000],   # cap edge list to keep file manageable
    }


# ── main ──────────────────────────────────────────────────────────────────────

def run_within_bucket():
    print("\n═══ WITHIN-BUCKET ANALYSIS ═══\n")
    (OUT_DIR / "within_bucket").mkdir(parents=True, exist_ok=True)
    summaries = []

    for bucket in BUCKETS:
        print(f"── {bucket}")
        files = sorted(glob.glob(f"{BASE_DIR}/{bucket}/*.json"))
        if MAX_FILES:
            files = files[:MAX_FILES]
        print(f"  {len(files)} files")

        G = build_graph(files, bucket_tag=bucket)
        print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        centrality = compute_centrality(G)
        result     = serialise_graph(G, centrality, bucket)

        out_path = OUT_DIR / "within_bucket" / f"{bucket}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved → {out_path}")

        summaries.append({
            "bucket":     bucket,
            "files":      len(files),
            "nodes":      result["node_count"],
            "edges":      result["edge_count"],
            "top5_pagerank": result["top_by_pagerank"][:5],
        })

    summary_path = OUT_DIR / "within_bucket" / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWithin-bucket summary → {summary_path}")


def run_cross_bucket():
    print("\n═══ CROSS-BUCKET ANALYSIS ═══\n")
    (OUT_DIR / "cross_bucket").mkdir(parents=True, exist_ok=True)

    # use the pre-merged combined dataset directory
    cross_dir = BASE_DIR / "combined_dataset_without_food_safety"
    files = sorted(glob.glob(f"{cross_dir}/*.json"))
    if MAX_FILES:
        files = files[:MAX_FILES * len(BUCKETS)]
    print(f"  {len(files)} files total")

    G = build_graph(files, bucket_tag=None)  # bucket inferred from filename prefix
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    centrality = compute_centrality(G)
    result     = serialise_graph(G, centrality, "cross_bucket")

    out_path = OUT_DIR / "cross_bucket" / "cross_bucket_analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved → {out_path}")

    # also build a per-entity-type breakdown
    by_type: dict[str, list] = defaultdict(list)
    for n in result["nodes"]:
        by_type[n["type"]].append(n)
    for t in by_type:
        by_type[t].sort(key=lambda x: x["pagerank"], reverse=True)

    breakdown_path = OUT_DIR / "cross_bucket" / "by_entity_type.json"
    with open(breakdown_path, "w") as f:
        json.dump({t: v[:50] for t, v in by_type.items()}, f, indent=2)
    print(f"  type breakdown → {breakdown_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode in ("within", "both"):
        run_within_bucket()

    if mode in ("cross", "both"):
        run_cross_bucket()

    print("\nDone.")
