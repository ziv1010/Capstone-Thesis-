# final_graph

`final_graph` contains reasoning-focused graph builders used by the later
experiments. These builders remove some noisy context nodes and shortcut edges
from the older case-star graph while keeping the parts needed for legal outcome
prediction.

## Files

- `build_graph.py`: reasoning-focused graph build entry point.
- `build_graph_section_sep.py`: section-separated graph build entry point.
- `graph_config_template.yaml`: graph-section template for dataset configs.
- `visualize_graph_structure.py`: produces graph-structure visualisations.
- `updated_graph/`: implementation modules for the reasoning-focused builder.

## Reasoning-Focused Policy

This graph family keeps:

- case-to-section edges
- case-to-party/court/judge/lawyer edges
- argument-to-statute/provision/precedent citation edges
- provision-to-statute membership edges
- party/lawyer-to-party-argument edges

It removes noisy context nodes such as generic `org`, `gpe`, `date`, and
`case_number`, plus shortcut edges that can over-connect argument nodes.

## Build Example

From `section_GNN`:

```bash
CUDA_VISIBLE_DEVICES=0,1 micromamba run -n thesis_work python final_graph/build_graph.py \
  --config runs/cross_bucket_total_dataset/config.yaml
```

For section-separated features:

```bash
CUDA_VISIBLE_DEVICES=0,1 micromamba run -n thesis_work python final_graph/build_graph_section_sep.py \
  --config ablations/section_sep_enc/cross_bucket_total_dataset/config.yaml
```

Graph caches and metadata are written under the config's `paths.graph_cache_dir`.
