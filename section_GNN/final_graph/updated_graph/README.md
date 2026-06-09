# updated_graph

Implementation package for the reasoning-focused graph builder in
`final_graph/`.

## Files

- `reasoning_graph_policy.py`: defines which node/edge types are kept or
  removed in the reasoning-focused graph.
- `case_star_builder.py`: builds local case-star records under that policy.
- `pipeline.py`: assembles cleaned cases, labels, splits, and PyG conversion
  into a graph bundle.

## Entry Points

Use the scripts one level up:

```bash
python final_graph/build_graph.py --config runs/cross_bucket_total_dataset/config.yaml
python final_graph/build_graph_section_sep.py --config ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml
```

Do not import this package from shell scripts directly; use the build entry
points so config loading, logging, and cache snapshots are handled consistently.
