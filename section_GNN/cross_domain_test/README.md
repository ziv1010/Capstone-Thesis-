# Cross-Domain Tests

Scripts and configs for evaluating trained graph models on held-out or out-of-domain buckets.

## Food Safety Holdout

`food_safety/run_cross_domain_food_safety.py` runs the food-safety cross-domain evaluation using the local graph pipeline outputs and matching config.

Generated graph caches, embeddings, processed cases, audits, and per-fold evaluation outputs are ignored because they contain large model/data artifacts.
