# 🕸️ src/graph — Heterogeneous Graph Construction

> Part of [`section_GNN/src/`](../README.md).

Builds the heterogeneous legal graphs consumed by the GNN models.

## 📄 Files

| File | Role |
|------|------|
| `schema.py` | Canonical node and relation names shared by all graph builders. |
| `case_star_builder.py` | Builds a local **case-star graph** per cleaned case. |
| `global_graph_builder.py` | Merges case-star graphs into one global graph, sharing canonical authority nodes across cases. |
| `pyg_builder.py` | Converts the global graph into PyTorch Geometric `HeteroData` (encodes text features, caches embeddings). |
| `pyg_builder_section_sep.py` | PyG conversion variant with **section-separated** case features. |

## 🌟 Graph Shape

Each case is a star centred on a `case` node. Depending on the config, nodes include:

- **Text sections:** `preamble`, `facts`, `arguments`, `petitioner_arguments`,
  `respondent_arguments`, `other_lawyer_arguments`
- **Participants:** `petitioner`, `respondent`, `court`, `judge`, `lawyer`,
  `petitioner_lawyer`, `defence_lawyer`
- **Legal authorities (shared across cases):** `statute`, `provision`, `precedent`

Graph variants (see [`../../ablations/`](../../ablations/README.md)) remove node types,
prevent cross-case sharing, or change text encoding.

## 🧮 Embeddings

`pyg_builder.py` calls the configured text encoder via `src.utils.text_encoder`
(BGE-M3, InLegalBERT, hashing fallback, …). Embeddings are cached under
`paths.embeddings_cache_dir` (usually `data/.../embeddings_cache/`), so rebuilding a graph
with unchanged text is fast.

---

⬆️ Back to [`src/`](../README.md)
