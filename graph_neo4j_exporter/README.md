# graph_neo4j_exporter

Standalone toolkit that converts a saved `.pt` graph bundle from `section_GNN`
into Neo4j-importable CSVs (and optionally uploads them live).

**Nothing inside `section_GNN` is ever written to or modified.**

---

## Directory layout

```
graph_neo4j_exporter/
├── export_to_neo4j.py      ← main script
├── requirements.txt
├── README.md               ← this file
└── exports/                ← generated at runtime, git-ignored
    └── <graph_stem>/
        ├── nodes.csv
        ├── edges.csv
        └── export_metadata.json
```

---

## Setup

```bash
cd graph_neo4j_exporter
pip install -r requirements.txt
```

---

## Usage

### 1 — Interactive picker (recommended first time)

```bash
python export_to_neo4j.py
```

You will be shown all available buckets and `.pt` files and can pick
interactively.  The CSVs are written to `exports/<graph_stem>/`.

### 2 — Non-interactive, CSV only

```bash
python export_to_neo4j.py \
    --pt-file /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/data/timed_bucket_runs/fin_fraud_timed_mistral/graph_cache/case_star_global_graph_fin_fraud_party_args.reasoning_focused.pt \
    --out-dir ./exports/fin_fraud_party_args \
    --no-upload
```

### 3 — Export + upload to a running Neo4j instance

```bash
python export_to_neo4j.py \
    --pt-file <your.pt> \
    --out-dir ./exports/my_run \
    --neo4j-uri bolt://localhost:7687 \
    --neo4j-user neo4j \
    --neo4j-password yourpassword
```

The script will prompt for the password if you omit `--neo4j-password`.

### 4 — Fastest bulk import (neo4j-admin, blank DB)

After generating the CSVs, run on the server hosting Neo4j:

```bash
neo4j-admin database import full \
  --nodes="exports/my_run/nodes.csv" \
  --relationships="exports/my_run/edges.csv" \
  --database=graph_thesis \
  --overwrite-destination=true
```

Then in `neo4j.conf` set:

```
dbms.default_database=graph_thesis
```

---

## CSV schemas

### nodes.csv

| Column | Description |
|--------|-------------|
| `node_uid:ID` | Globally unique node identifier (`<type>__<key>`) |
| `node_type:LABEL` | Neo4j node label (e.g. `case`, `statute`, `judge`) |
| `local_idx:int` | Index within that node type |
| `display_name` | Short human-readable name (≤ 80 chars) |
| `outcome_label` | `allowed` / `dismissed` / `neutral` (case nodes only) |
| `split` | `train` / `val` / `test` (case nodes only) |
| `feat_dim:int` | Feature vector dimension |

### edges.csv

| Column | Description |
|--------|-------------|
| `:START_ID` | Source `node_uid` |
| `:END_ID` | Destination `node_uid` |
| `:TYPE` | Neo4j relationship type (e.g. `CITES_STATUTE`) |
| `src_node_type` | Source node type string |
| `dst_node_type` | Destination node type string |
| `relation` | Raw relation name from the graph bundle |

---

## Neo4j Browser queries to get started

```cypher
// Count nodes by type
MATCH (n) RETURN n.node_type, count(n) ORDER BY count(n) DESC

// All node types in the graph
CALL db.labels()

// Inspect a case neighbourhood (2 hops)
MATCH p=(c:case {outcome_label: 'allowed'})-[*1..2]-(nb)
RETURN p LIMIT 50

// Statute citation authority
MATCH (s:statute)<-[:CITES_STATUTE]-(c:case)
RETURN s.display_name, count(c) AS citations ORDER BY citations DESC LIMIT 20

// Train/val/test split overview
MATCH (c:case)
RETURN c.split, c.outcome_label, count(c)
ORDER BY c.split, c.outcome_label
```

---

## Gephi import

1. Open Gephi → **File → Import Spreadsheet**
2. Nodes file: `exports/<run>/nodes.csv`  — ID column = `node_uid:ID`
3. Edges file: `exports/<run>/edges.csv`  — Source = `:START_ID`, Target = `:END_ID`
4. Layout: **Force Atlas 2**
5. Colour partition by `node_type`; size ranking by `feat_dim` or degree.

---

## Flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--pt-file` | *(interactive)* | Path to `.pt` bundle |
| `--out-dir` | `exports/<stem>` | Output directory for CSVs |
| `--no-upload` | `False` | Skip Neo4j upload |
| `--neo4j-uri` | `bolt://localhost:7687` | Neo4j Bolt URI |
| `--neo4j-user` | `neo4j` | Neo4j username |
| `--neo4j-password` | *(prompted)* | Neo4j password |
| `--neo4j-database` | `neo4j` | Target database name |
| `--batch-size` | `500` | Rows per write transaction |
| `--clear-first` | `False` | DETACH DELETE all nodes before import |
