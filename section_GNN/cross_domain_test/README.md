# 🌍 cross_domain_test — Held-Out Domain Evaluation

> Part of [`section_GNN/`](../README.md) · evaluates trained models on a legal domain that
> was **never part of the training distribution**.

## 🗂️ Subfolders

| Folder | Purpose |
|--------|---------|
| [`food_safety/`](food_safety/README.md) | Evaluates cross-bucket checkpoints on food-safety cases. |

## 🔄 Workflow Pattern

1. Create or reuse a domain-specific config.
2. Preprocess the new domain's JSON files.
3. Build a graph under the **same graph assumptions as training**.
4. Evaluate the trained checkpoints on the new graph.
5. Aggregate fold metrics into a cross-domain summary.

See [`food_safety/README.md`](food_safety/README.md) for the concrete workflow.

---

⬆️ Back to [`section_GNN/`](../README.md)
