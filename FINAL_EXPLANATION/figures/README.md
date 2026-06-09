# Figures

This folder stores paper-facing static figures generated from
`FINAL_EXPLANATION/outputs`.

## Current Figures

- `fig1_faithfulness_curves.*`
  - Sufficiency and comprehensiveness curves comparing counterfactual,
    attention, and random rankings.

- `fig2_community_cluster_sankey.*`
  - Relationship between structural communities and embedding clusters.

- `fig3_contrastive_subgraph_51419_15962.*`
  - Contrastive subgraph figure for one query/opposite-case pair.

## Regeneration

From the repository root:

```bash
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_paper_figures.py
```

The script writes both PNG and PDF by default.
