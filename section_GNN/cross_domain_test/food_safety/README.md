# 🍽️ food_safety — Cross-Domain Evaluation

> Part of [`cross_domain_test/`](../README.md) · tests whether the five-domain cross-bucket
> model generalizes to the **held-out food-safety domain**.

## 📄 Main Files

| File | Role |
|------|------|
| `run_cross_domain_food_safety.py` | End-to-end cross-domain runner (preprocess → graph → evaluate). |
| `food_safety_cross_domain_config.yaml` | Config for the food-safety graph. |
| `cross_domain_summary.json` | Aggregate evaluation summary. |

## 📤 Generated Folders

`processed/` (cleaned cases + summary) · `audits/` (leakage/preprocessing audits) ·
`embeddings_cache/` · `graph_cache/` (food-safety graph bundle) · `logs/` (per-fold logs).

## ▶️ Run

From `section_GNN/`:

```bash
micromamba run -n thesis_work python cross_domain_test/food_safety/run_cross_domain_food_safety.py \
  --cuda 0,1,2,3
```

The runner derives repository paths from its own location, so no path editing is needed.
Input text originates from `INPUT_DATA/food_safety_text*` via the Stage ② extraction.

---

⬆️ Back to [`cross_domain_test/`](../README.md)
