# Dataset Builder and Explorer

Utilities for converting raw legal case material into bucketed, merged, and entity-normalized datasets used by the thesis experiments.

## Main Areas

- `ENCODING_CLASSIFICATION/`: embedding-space analysis and bucket clustering visualizations.
- `Nyaya/`: bucket creation and financial-fraud classification helpers.
- `Timeline_Maker/`: merged-case construction, cross-bucket dataset builders, timeline visualizers, and entity resolution.

## Data Policy

Generated corpora, LLM cache directories, extracted text dumps, and resolved output trees are intentionally ignored by Git. Keep scripts, configs, and documentation in the repository; regenerate or copy large datasets locally when running experiments.

## Typical Flow

1. Build or classify bucket datasets using the scripts under `Nyaya/`.
2. Merge per-case JSON outputs with `Timeline_Maker/merge_cases_v3.py`.
3. Optionally canonicalize statutes, provisions, and precedents using `Timeline_Maker/entity_resolver/resolve_entities.py`.
4. Pass the resulting local dataset into `section_GNN` or downstream analysis pipelines.
