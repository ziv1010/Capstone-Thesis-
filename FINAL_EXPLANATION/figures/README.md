# 🖼️ figures — Paper-Facing Static Figures

> Part of [`FINAL_EXPLANATION/`](../README.md) · generated from the tables in
> [`../outputs/`](../outputs/README.md).

## 📊 Current Figures

| Figure | Contents |
|--------|----------|
| `fig1_faithfulness_curves.*` | Sufficiency & comprehensiveness curves comparing counterfactual, attention, and random rankings. |
| `fig2_community_cluster_sankey.*` | Relationship between structural communities and embedding clusters. |
| `fig3_contrastive_subgraph_51419_15962.*` | Contrastive subgraph for one query/opposite-case pair. |

## ♻️ Regeneration

From the repository root (writes both PNG and PDF):

```bash
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_paper_figures.py
```

---

⬆️ Back to [`FINAL_EXPLANATION/`](../README.md)
