# Legal GNN — Visualiser

Interactive browser-based visualisation suite for the **Legal GNN** pipeline (`section_GNN`).  
No Python environment needed to *view* — just a static HTTP server.

---

## Quick Start

```bash
# from the visualiser/ directory
python3 -m http.server 8080
```

Then open in your browser:

| Page | URL |
|------|-----|
| Graph Visualiser | http://localhost:8080/ |
| Entity Stats Dashboard | http://localhost:8080/stats/ |
| Layers Explorer | http://localhost:8080/layers/ |

---

## Contents

```
visualiser/
├── README.md               ← you are here
├── data/
│   ├── cases_catalog.js    ← fallback sampled catalog
│   └── training_graph/     ← exact per-case training graph export
├── index.html              ← interactive case graph visualiser
├── layers/
│   └── index.html          ← graph/model layer explainer
└── stats/
    └── index.html          ← entity statistics dashboard
```

---

## Page 1 — Graph Visualiser (`index.html`)

A **D3.js force-directed graph** rendering the heterogeneous legal knowledge graph used by the pipeline.

When `visualiser/data/training_graph/index.js` is present, the page now prefers an **exact training-graph export** built from:

- already-filtered `cleaned_cases`
- the current graph snapshot config
- the same `build_case_star_graph()` logic used by the pipeline

That means the graph view reflects the real local case star used during training, including role-specific text nodes such as `petitioner_arguments`, `respondent_arguments`, and `other_lawyer_arguments`, plus the exact edge relations that survive preprocessing and graph construction.

### Exact Training Graph Behaviour

- The left sidebar lists all loaded cases and shows how many other loaded cases each one links to.
- The search box filters by title, case id, parties, court, judge, lawyers, statutes, provisions, precedents, and case numbers.
- The central graph now stays faithful to the **local training subgraph** for the selected case.
- The connected-case sidebar summarizes the broader global neighborhood through truly shared training nodes, instead of drawing synthetic case-to-case edges for hundreds of neighbors.
- If the exact export is missing, the page falls back to the older sampled catalog.

### Graph Node Types

| Colour | Node Type | Description |
|--------|-----------|-------------|
| 🟠 Orange | `case` | Root node — one per case |
| 🔵 Blue | `preamble` | Header section text |
| 🔵 Light Blue | `facts` | Factual background section |
| 🟢 Green | `arguments` | Combined arguments section |
| 🟢 Dark Green | `petitioner_arguments` | Arguments made by petitioner side |
| 🟣 Purple | `respondent_arguments` | Arguments made by respondent side |
| ⚪ Slate | `other_lawyer_arguments` | Non-party lawyer argument bucket |
| 🔴 Red | `petitioner` | Named petitioner entity |
| 🟠 Amber | `respondent` | Named respondent entity |
| 🟡 Yellow | `judge` | Presiding judge(s) |
| 🩵 Cyan | `court` | Court/bench |
| 🩷 Pink | `petitioner_lawyer` | Petitioner's counsel |
| 🟣 Lavender | `defence_lawyer` | Respondent's counsel |
| 🟢 Lime | `statute` | Cited legislation |
| 🟢 Dark | `provision` | Specific section/rule |
| 🔵 Navy | `precedent` | Cited prior case |
| 🟣 Violet | `org` | Organisation mention |
| 🔵 Sky | `gpe` | Place / geopolitical entity |
| 🟢 Mint | `date` | Date mention |
| ⚪ Grey | `case_number` | FIR/case reference number |

### Controls

| Control | Action |
|---------|--------|
| Click case card (sidebar) | Switch to that case's graph |
| Type in the search box | Filter the loaded catalog |
| Click any node | Inspect node type, label and metadata in sidebar |
| Click a connected-case card | Jump focus to that related case |
| Drag node | Reposition node (simulation continues) |
| Scroll | Zoom in/out |
| `＋` / `－` buttons | Zoom in / out |
| `⊡` button | Reset zoom to fit screen |
| `🏷` button | Toggle node labels on/off |

### Rebuild The Exact Training Graph Export

If you want the browser view to mirror the current training graph snapshot:

```bash
python3 ../scripts/export_training_graph_visualiser.py
```

That writes:

```text
visualiser/data/training_graph/index.js
visualiser/data/training_graph/cases/*.json
```

### Rebuild The Fallback Catalog

If you want to regenerate the older sampled catalog instead:

```bash
python3 ../scripts/export_visualiser_catalog.py
```

That writes:

```text
visualiser/data/cases_catalog.js
```

The default exporter settings currently select food-law cases, keep a mix of connected and isolated examples, and suppress very high-frequency shared descriptors so the visual overlap remains readable.

---

## Page 1.5 — Layers Explorer (`layers/index.html`)

This page explains:

- the **graph layers**: case anchor, text sections, parties/forum, counsel/authorities, and the global shared layer
- the **model layers**: input projections, `HGTConv × N`, residual + LayerNorm + ReLU, and the final `MLPHead`
- how the graph tab’s dashed related-case links should be interpreted

---

## Page 2 — Entity Stats Dashboard (`stats/index.html`)

Statistical breakdown of entity mentions extracted from **300 augmented case JSON files**.

### Dataset Summary (300 cases)

| Outcome Class | Count | Share |
|---------------|-------|-------|
| Appellant Won | 143 | 47.7 % |
| Procedural / Pending | 99 | 33.0 % |
| Appellant Lost | 58 | 19.3 % |

### Top Judges (by caseload)

| Judge | Cases | Petitioner Wins |
|-------|-------|----------------|
| S.Q. Pathan | 165 | — |
| Sikar | 65 | 65 |
| Chittorgarh | 45 | 45 |
| M.M. Shrivastava | 41 | 40 |
| Munnuri Laxman | 28 | 28 |

### Top Lawyers (by caseload)

| Lawyer | Cases | Wins (pet) |
|--------|-------|-----------|
| Suraj Samdarshi | 123 | — |
| Naresh Dixit | 123 | — |
| Gyan Prakash Ojha | 114 | — |
| Govind Ram | 77 | 77 |
| K.SanjaiGandhi | 11 | 1 (defence) |

### Top Respondents

| Respondent | Mentions |
|------------|----------|
| Union Of India | 9,279 |
| State Of Rajasthan | 6,888 |
| Commissioner Of Transport | 729 |

### Top Statutes Cited

| Statute | Cases |
|---------|-------|
| Motor Vehicles Act, 1988 | 240 |
| Food Safety and Standards Act, 2006 | 68 |
| IPC | 53 |
| Cr.P.C | 49 |

---

## GNN Architecture Reference

The underlying model is a **Heterogeneous Graph Transformer (HGT)**.

```
src/models/hetero_gnn.py  →  class HeteroLegalOutcomeGNN
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `architecture` | `"hgt"` | `"hgt"` → `HGTConv` · `"heteroconv"` → `SAGEConv` per edge type |
| `hidden_dim` | `128` | Shared hidden size for all node types |
| `num_layers` | `2` | Number of message-passing layers |
| `num_heads` | `4` | Attention heads (HGT only) |
| `dropout` | `0.2` | Applied after each conv layer |

**Forward pass summary:**

```
x_dict  → Linear projections + type embeddings
        → [HGTConv × num_layers]  (residual + LayerNorm + ReLU each layer)
        → hidden["case"]
        → MLPHead (2-layer)
        → logits
```

**Why HGT?**  
HGT uses separate key/query/value matrices per *(source type, relation, target type)* triple, making it well-suited to legal graphs where the same structural role (e.g., "cited by arguments") carries different semantic weight depending on which node types are involved.

---

## Data Source

All graphs are built from the augmented JSON files in:

```
Capstone-Thesis-/Manual_Check/augmented_jsons/
```

Each JSON contains:
- `annotations` — sentence-level spans with entity labels (`JUDGE`, `LAWYER`, `PETITIONER`, `RESPONDENT`, `STATUTE`, `PROVISION`, `PRECEDENT`, …)
- `raw_result.summary` — section text (`PREAMBLE`, `facts`, `arguments`, `decision`)
- `case_outcome_label` — ground truth label (`appellant_won` / `respondent_won` / `postponed_or_procedural`)
- `llm_case_outcome` — LLM-generated outcome confidence + explanation

Graph construction logic lives in:

```
section_GNN/src/graph/
├── schema.py            ← node types, edge relations, leakage rules
├── case_star_builder.py ← builds per-case heterogeneous graph
├── global_graph_builder.py ← merges cases into global graph
└── pyg_builder.py       ← converts to PyG HeteroData tensors
```
