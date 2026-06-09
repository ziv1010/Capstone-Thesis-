# src/graph

This package builds the heterogeneous legal graphs used by the GNN models.

## Files

- `schema.py`: node and relation names shared across graph builders.
- `case_star_builder.py`: builds per-case star graphs from cleaned cases.
- `global_graph_builder.py`: merges case-star graphs into a single global graph.
- `pyg_builder.py`: converts the global graph into PyTorch Geometric `HeteroData`.
- `pyg_builder_section_sep.py`: PyG conversion variant with section-separated case features.

## Graph Shape

The baseline graph centers each case around a `case` node connected to text,
party, court, lawyer, and legal-authority nodes. Depending on the config, graph
nodes can include:

- `case`
- `preamble`, `facts`, `arguments`
- `petitioner_arguments`, `respondent_arguments`, `other_lawyer_arguments`
- `petitioner`, `respondent`
- `court`, `judge`, `lawyer`, `petitioner_lawyer`, `defence_lawyer`
- `statute`, `provision`, `precedent`

Graph variants may remove node types, prevent cross-case sharing, or change how
case text is encoded.

## Embeddings

`pyg_builder.py` calls the configured text encoder through `src.utils.text_encoder`.
Embeddings are cached in `paths.embeddings_cache_dir`, usually under
`data/.../embeddings_cache/`.
