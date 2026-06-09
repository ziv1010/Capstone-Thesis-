# cross_domain_test

Cross-domain tests evaluate a trained model on a held-out legal domain that was
not part of the original training distribution.

## Subfolders

- `food_safety/`: evaluates cross-bucket legal-domain checkpoints on food-safety
  cases.

## Workflow

The general pattern is:

1. Create or reuse a domain-specific config.
2. Preprocess that domain's JSON files.
3. Build a graph using the same graph assumptions as training.
4. Evaluate trained checkpoints on the new graph.
5. Aggregate fold metrics into a cross-domain summary.

See `food_safety/README.md` for the concrete workflow.
