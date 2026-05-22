#!/usr/bin/env python3
"""
graph_neo4j_exporter/export_to_neo4j.py
========================================
Standalone tool — reads a saved .pt graph bundle from section_GNN and:
  1. Exports  nodes.csv  +  edges.csv  (Gephi / Neo4j admin-import ready)
  2. Optionally bulk-loads them into a running Neo4j instance via the
     official neo4j Python driver (no py2neo dependency required).

Nothing inside section_GNN is ever written to or modified.

Usage examples
--------------
# Interactive picker (lists all .pt files under the data dir)
python export_to_neo4j.py

# Non-interactive — just export CSVs, skip Neo4j upload
python export_to_neo4j.py \\
    --pt-file /path/to/case_star_fin_fraud_party_args.reasoning_focused.pt \\
    --out-dir ./exports/fin_fraud_party_args \\
    --no-upload

# Export + upload to Neo4j
python export_to_neo4j.py \\
    --pt-file /path/to/graph.pt \\
    --out-dir ./exports/my_run \\
    --neo4j-uri bolt://localhost:7687 \\
    --neo4j-user neo4j \\
    --neo4j-password yourpassword
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# ── ensure we can import torch without the section_GNN src on the path ──────
try:
    import torch
except ImportError:
    sys.exit("torch is not installed in this environment. Run:  pip install torch")

# ── optional: neo4j driver ───────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase  # type: ignore
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# ── paths ────────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_GRAPH_CACHE_ROOT = (
    _THIS_DIR.parent
    / "section_GNN"
    / "data"
    / "timed_bucket_runs"
)

BUCKET_DIRS = {
    "fin_fraud":             _GRAPH_CACHE_ROOT / "fin_fraud_timed_mistral"         / "graph_cache",
    "family_matrimonial":    _GRAPH_CACHE_ROOT / "family_matrimonial_timed_mistral" / "graph_cache",
    "land_property":         _GRAPH_CACHE_ROOT / "land_property_timed_mistral"      / "graph_cache",
    "motor_accidents":       _GRAPH_CACHE_ROOT / "motor_accidents_timed_mistral"    / "graph_cache",
    "sexual_offences":       _GRAPH_CACHE_ROOT / "sexual_offences_timed_mistral"    / "graph_cache",
    "cross_bucket":          _GRAPH_CACHE_ROOT / "cross_bucket_total_dataset"       / "graph_cache",
}

# ── label maps (match your config's label_names ordering) ────────────────────
DEFAULT_LABEL_MAP = {0: "allowed", 1: "dismissed", 2: "neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE PICKER
# ─────────────────────────────────────────────────────────────────────────────

def pick_pt_file_interactive() -> Path:
    """Walk BUCKET_DIRS, let the user choose bucket then specific .pt file."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║        Graph → Neo4j Exporter  (picker)      ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ── step 1: bucket ───────────────────────────────────────────────────────
    available_buckets = {k: v for k, v in BUCKET_DIRS.items() if v.exists()}
    if not available_buckets:
        sys.exit("ERROR: No bucket graph_cache directories found under:\n  " + str(_GRAPH_CACHE_ROOT))

    bucket_list = sorted(available_buckets.keys())
    print("Available buckets:")
    for i, name in enumerate(bucket_list):
        print(f"  [{i+1}] {name}")
    print(f"  [{len(bucket_list)+1}] Enter a custom path manually")

    while True:
        raw = input("\nSelect bucket number: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if idx == len(bucket_list):          # custom path
                custom = input("Paste full path to .pt file: ").strip()
                return Path(custom)
            if 0 <= idx < len(bucket_list):
                chosen_bucket = bucket_list[idx]
                cache_dir = available_buckets[chosen_bucket]
                break
        print("  (invalid — try again)")

    # ── step 2: model / ablation ─────────────────────────────────────────────
    pt_files = sorted(cache_dir.glob("*.pt"))
    if not pt_files:
        sys.exit(f"ERROR: No .pt files found in {cache_dir}")

    print(f"\nGraphs available in  {chosen_bucket}:")
    for i, p in enumerate(pt_files):
        size_mb = p.stat().st_size / 1_048_576
        print(f"  [{i+1}] {p.name}   ({size_mb:.1f} MB)")

    while True:
        raw = input("\nSelect graph number: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(pt_files):
                return pt_files[idx]
        print("  (invalid — try again)")


# ─────────────────────────────────────────────────────────────────────────────
# BUNDLE LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle(pt_path: Path) -> dict:
    """Load the torch bundle. Maps storage to CPU even if saved on GPU."""
    print(f"\nLoading bundle from:\n  {pt_path}")
    bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "data" not in bundle:
        sys.exit("ERROR: .pt file does not look like a graph bundle (missing 'data' key).")
    print("  Bundle loaded ✓")
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csvs(
    bundle: dict,
    out_dir: Path,
    label_map: dict[int, str] | None = None,
) -> tuple[Path, Path]:
    """
    Write  nodes.csv  and  edges.csv  into out_dir.
    Returns (nodes_csv_path, edges_csv_path).

    nodes.csv schema
    ----------------
    node_uid, node_type, label (for cases), split, display_name, feat_dim

    edges.csv schema
    ----------------
    src_uid, dst_uid, edge_type, src_node_type, dst_node_type
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data     = bundle["data"]
    metadata = bundle.get("metadata", {})
    label_map = label_map or DEFAULT_LABEL_MAP
    label_names = metadata.get("label_names", list(label_map.values()))

    # ── build a global  node_uid → (node_type, local_idx)  reverse-map ───────
    node_mappings: dict[str, dict[str, int]] = metadata.get("node_mappings", {})
    # invert: (node_type, local_idx) → node_key
    inv_map: dict[tuple[str, int], str] = {}
    for ntype, key2idx in node_mappings.items():
        for key, idx in key2idx.items():
            inv_map[(ntype, idx)] = key

    split_assignments: dict[str, str] = metadata.get("split_assignments", {})

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"

    # ── NODES ────────────────────────────────────────────────────────────────
    node_uid_registry: dict[tuple[str, int], str] = {}   # (ntype, local_idx) → uid

    with open(nodes_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "node_uid:ID",
            "node_type:LABEL",
            "local_idx:int",
            "display_name",
            "outcome_label",
            "split",
            "feat_dim:int",
        ])

        for ntype in data.node_types:
            store = data[ntype]
            n = store.num_nodes

            # node_id list (string keys) — stored by build_pyg_heterodata
            node_ids = list(store.node_id) if hasattr(store, "node_id") else [
                inv_map.get((ntype, i), f"{ntype}_{i}") for i in range(n)
            ]

            feat_dim = int(store.x.shape[1]) if hasattr(store, "x") else 0

            # label / split only meaningful for case nodes
            y_list = store.y.tolist() if hasattr(store, "y") else []
            case_ids_list = list(store.case_id) if hasattr(store, "case_id") else []

            # build split lookup by local idx
            split_by_local: dict[int, str] = {}
            if hasattr(store, "train_mask"):
                for i, v in enumerate(store.train_mask.tolist()):
                    if v:
                        split_by_local[i] = "train"
            if hasattr(store, "val_mask"):
                for i, v in enumerate(store.val_mask.tolist()):
                    if v:
                        split_by_local[i] = "val"
            if hasattr(store, "test_mask"):
                for i, v in enumerate(store.test_mask.tolist()):
                    if v:
                        split_by_local[i] = "test"

            for local_idx in range(n):
                node_key = node_ids[local_idx] if local_idx < len(node_ids) else f"{ntype}_{local_idx}"
                # uid format:  <node_type>__<key>  (double-underscore avoids ambiguity)
                uid = f"{ntype}__{node_key}"
                node_uid_registry[(ntype, local_idx)] = uid

                # label
                outcome = ""
                if y_list and local_idx < len(y_list):
                    raw_y = y_list[local_idx]
                    if 0 <= raw_y < len(label_names):
                        outcome = label_names[raw_y]
                    else:
                        outcome = str(raw_y)

                split = split_by_local.get(local_idx, "")

                # display name: shorten to ≤ 80 chars
                display = node_key[:80]

                writer.writerow([uid, ntype, local_idx, display, outcome, split, feat_dim])

    print(f"  Nodes written → {nodes_path}  ({_count_lines(nodes_path)-1:,} rows)")

    # ── EDGES ────────────────────────────────────────────────────────────────
    with open(edges_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            ":START_ID",
            ":END_ID",
            ":TYPE",
            "src_node_type",
            "dst_node_type",
            "relation",
        ])

        for edge_type in data.edge_types:
            src_type, relation, dst_type = edge_type
            edge_store = data[edge_type]
            if not hasattr(edge_store, "edge_index"):
                continue
            edge_index = edge_store.edge_index  # shape [2, E]
            E = edge_index.shape[1]
            src_arr = edge_index[0].tolist()
            dst_arr = edge_index[1].tolist()
            neo4j_rel = relation.upper().replace("-", "_").replace(" ", "_")

            for i in range(E):
                src_uid = node_uid_registry.get((src_type, src_arr[i]), f"{src_type}_{src_arr[i]}")
                dst_uid = node_uid_registry.get((dst_type, dst_arr[i]), f"{dst_type}_{dst_arr[i]}")
                writer.writerow([src_uid, dst_uid, neo4j_rel, src_type, dst_type, relation])

    print(f"  Edges written → {edges_path}  ({_count_lines(edges_path)-1:,} rows)")

    # ── also dump a metadata snapshot ────────────────────────────────────────
    meta_snap = {
        "source_pt": str(bundle.get("_pt_path", "unknown")),
        "node_types": list(data.node_types),
        "edge_types": ["|".join(et) for et in data.edge_types],
        "node_counts": {ntype: int(data[ntype].num_nodes) for ntype in data.node_types},
        "label_names": label_names,
        "split_counts": metadata.get("case_split_counts", {}),
    }
    meta_path = out_dir / "export_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_snap, f, indent=2)
    print(f"  Metadata written → {meta_path}")

    return nodes_path, edges_path


def _count_lines(p: Path) -> int:
    with open(p, "rb") as f:
        return sum(1 for _ in f)


# ─────────────────────────────────────────────────────────────────────────────
# NEO4J UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_neo4j(
    nodes_path: Path,
    edges_path: Path,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
    batch_size: int = 500,
    clear_first: bool = False,
) -> None:
    if not NEO4J_AVAILABLE:
        print(
            "\n[SKIP] neo4j Python driver not installed.\n"
            "  Run:  pip install neo4j\n"
            "  Then re-run with --upload flag.\n"
        )
        return

    print(f"\nConnecting to Neo4j at {uri} …")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session(database=database) as session:
        if clear_first:
            print("  Clearing existing graph …")
            session.run("MATCH (n) DETACH DELETE n")

        # ── ensure uniqueness constraints for node UIDs ───────────────────
        print("  Creating constraints …")
        session.run(
            "CREATE CONSTRAINT graph_node_uid IF NOT EXISTS "
            "FOR (n:_GraphNode) REQUIRE n.node_uid IS UNIQUE"
        )

        # ── load nodes in batches ─────────────────────────────────────────
        print("  Uploading nodes …")
        _upload_nodes_batched(session, nodes_path, batch_size)

        # ── load edges in batches ─────────────────────────────────────────
        print("  Uploading edges …")
        _upload_edges_batched(session, edges_path, batch_size)

    driver.close()
    print("Neo4j upload complete ✓")


def _upload_nodes_batched(session, nodes_path: Path, batch_size: int) -> None:
    """Read nodes CSV and MERGE into Neo4j, setting labels dynamically."""
    rows = _read_csv_dicts(nodes_path)
    total = 0
    for chunk in _chunked(rows, batch_size):
        # We use APOC if available; fall back to a plain parameterised query.
        # Plain approach: one MERGE per node (works without APOC)
        for row in chunk:
            uid       = row["node_uid:ID"]
            ntype     = row["node_type:LABEL"]
            local_idx = int(row["local_idx:int"])
            display   = row["display_name"]
            outcome   = row["outcome_label"]
            split     = row["split"]
            feat_dim  = int(row["feat_dim:int"])

            # Cypher: merge on uid, set type-specific label + props
            session.run(
                f"MERGE (n:_GraphNode:{_safe_label(ntype)} {{node_uid: $uid}}) "
                "SET n.node_type = $ntype, "
                "    n.local_idx = $local_idx, "
                "    n.display_name = $display, "
                "    n.outcome_label = $outcome, "
                "    n.split = $split, "
                "    n.feat_dim = $feat_dim",
                uid=uid, ntype=ntype, local_idx=local_idx,
                display=display, outcome=outcome, split=split, feat_dim=feat_dim,
            )
        total += len(chunk)
        print(f"    … {total:,} nodes committed", end="\r")
    print(f"    {total:,} nodes committed ✓         ")


def _upload_edges_batched(session, edges_path: Path, batch_size: int) -> None:
    """Read edges CSV and CREATE relationships in Neo4j."""
    rows = _read_csv_dicts(edges_path)
    total = 0
    # group by relationship type so we can issue typed MATCH/MERGE
    from collections import defaultdict
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_rel[row[":TYPE"]].append(row)

    for rel_type, rel_rows in by_rel.items():
        for chunk in _chunked(rel_rows, batch_size):
            pairs = [{"src": r[":START_ID"], "dst": r[":END_ID"]} for r in chunk]
            session.run(
                f"UNWIND $pairs AS pair "
                "MATCH (a:_GraphNode {node_uid: pair.src}), "
                "      (b:_GraphNode {node_uid: pair.dst}) "
                f"MERGE (a)-[:{rel_type}]->(b)",
                pairs=pairs,
            )
            total += len(chunk)
            print(f"    … {total:,} edges committed", end="\r")
    print(f"    {total:,} edges committed ✓         ")


def _safe_label(s: str) -> str:
    """Turn a string into a valid Neo4j label (alphanumeric + underscores)."""
    import re
    return re.sub(r"[^a-zA-Z0-9_]", "_", s)


def _read_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a .pt graph bundle → CSVs → (optionally) Neo4j",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pt-file", metavar="PATH",
        help="Path to the .pt graph bundle.  Omit to use the interactive picker.",
    )
    p.add_argument(
        "--out-dir", metavar="DIR", default=None,
        help="Directory for exported CSVs.  Defaults to ./exports/<graph_stem>",
    )
    p.add_argument(
        "--no-upload", action="store_true",
        help="Only generate CSVs; skip Neo4j upload.",
    )
    p.add_argument("--neo4j-uri",      default="bolt://localhost:7687")
    p.add_argument("--neo4j-user",     default="neo4j")
    p.add_argument("--neo4j-password", default=None,
                   help="Neo4j password.  Prompted interactively if omitted and upload is requested.")
    p.add_argument("--neo4j-database", default="neo4j")
    p.add_argument("--batch-size", type=int, default=500,
                   help="Rows per Neo4j write transaction (default: 500).")
    p.add_argument("--clear-first", action="store_true",
                   help="DETACH DELETE all nodes before uploading.  Use with caution.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── resolve .pt file ─────────────────────────────────────────────────────
    if args.pt_file:
        pt_path = Path(args.pt_file)
        if not pt_path.exists():
            sys.exit(f"ERROR: file not found: {pt_path}")
    else:
        pt_path = pick_pt_file_interactive()
        if not pt_path.exists():
            sys.exit(f"ERROR: selected file not found: {pt_path}")

    # ── resolve output dir ───────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stem = pt_path.stem.replace(".reasoning_focused", "").replace(".", "_")
        out_dir = _THIS_DIR / "exports" / stem

    # ── load + export ─────────────────────────────────────────────────────────
    bundle = load_bundle(pt_path)
    bundle["_pt_path"] = str(pt_path)
    nodes_path, edges_path = export_csvs(bundle, out_dir)

    print(f"\n✔ CSVs ready in:  {out_dir}")
    print( "  ├── nodes.csv")
    print( "  ├── edges.csv")
    print( "  └── export_metadata.json")

    # ── optional Neo4j upload ─────────────────────────────────────────────────
    if args.no_upload:
        print("\n[--no-upload set] Skipping Neo4j upload.")
        _print_import_hints(out_dir, args)
        return

    if not NEO4J_AVAILABLE:
        print(
            "\n[INFO] neo4j Python driver not found — skipping auto-upload.\n"
            "  Install with:  pip install neo4j\n"
            "  Then re-run without  --no-upload\n"
        )
        _print_import_hints(out_dir, args)
        return

    password = args.neo4j_password
    if not password:
        import getpass
        password = getpass.getpass(f"Neo4j password for {args.neo4j_user}@{args.neo4j_uri}: ")

    upload_to_neo4j(
        nodes_path=nodes_path,
        edges_path=edges_path,
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=password,
        database=args.neo4j_database,
        batch_size=args.batch_size,
        clear_first=args.clear_first,
    )

    _print_import_hints(out_dir, args)


def _print_import_hints(out_dir: Path, args) -> None:
    print("\n" + "─" * 60)
    print("NEXT STEPS")
    print("─" * 60)
    print(f"""
Option A — Neo4j Admin bulk import (fastest for large graphs):
  Requires a blank database.  From your Neo4j server:

  neo4j-admin database import full \\
    --nodes="{out_dir}/nodes.csv" \\
    --relationships="{out_dir}/edges.csv" \\
    --database=graph_thesis \\
    --overwrite-destination=true

  Then in neo4j.conf set:  dbms.default_database=graph_thesis

Option B — Python driver (already done if --upload was set):
  python export_to_neo4j.py \\
    --pt-file <your.pt> \\
    --out-dir {out_dir} \\
    --neo4j-uri {args.neo4j_uri} \\
    --neo4j-user {args.neo4j_user}

Option C — Gephi import:
  Open Gephi → Import Spreadsheet
    Nodes: {out_dir}/nodes.csv   (node_uid = ID column)
    Edges: {out_dir}/edges.csv   (:START_ID / :END_ID)
  Colour by  node_type;  size by  feat_dim;  partition by  split.

Neo4j Browser quick-start queries:
  // count nodes by type
  MATCH (n) RETURN n.node_type, count(n) ORDER BY count(n) DESC

  // inspect a case neighbourhood
  MATCH p=(c:case {{outcome_label: 'allowed'}})-[*1..2]-(nb)
  RETURN p LIMIT 50

  // statute authority nodes
  MATCH (s:statute)
  RETURN s.display_name, s.outcome_label ORDER BY s.display_name LIMIT 30
""")


if __name__ == "__main__":
    main()
