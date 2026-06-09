# food_safety

This folder contains the food-safety cross-domain evaluation.

## Main Files

- `run_cross_domain_food_safety.py`: end-to-end cross-domain runner.
- `food_safety_cross_domain_config.yaml`: generated/checked config for the
  food-safety graph.
- `cross_domain_summary.json`: aggregate evaluation summary.

## Generated Folders

- `processed/`: cleaned cases and preprocessing summary.
- `audits/`: leakage and preprocessing audit JSONs.
- `embeddings_cache/`: cached text embeddings.
- `graph_cache/`: food-safety graph bundle and metadata.
- `logs/`: per-fold evaluation logs.

## Run

From `section_GNN`:

```bash
micromamba run -n thesis_work python cross_domain_test/food_safety/run_cross_domain_food_safety.py \
  --cuda 0,1,2,3
```

The runner derives repository paths from its own location and uses relative
config paths where possible.
