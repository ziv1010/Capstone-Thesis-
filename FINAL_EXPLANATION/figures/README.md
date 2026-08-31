# 🖼️ figures — Paper-Facing Static Figures

> Part of [`FINAL_EXPLANATION/`](../README.md) · generated from the tables in
> [`../outputs/`](../outputs/README.md).

## 📊 Current Figures

| Figure | Contents |
|--------|----------|
| `fig1_faithfulness_curves.*` | Sufficiency & comprehensiveness curves comparing counterfactual, attention, and random rankings. |
| `fig2_community_cluster_sankey.*` | Relationship between structural communities and embedding clusters. |
| `fig3_contrastive_subgraph_51419_15962.*` | Contrastive subgraph for one query/opposite-case pair. |

## 🎤 Presentation Figures

Per-case slide figures, written by `generate_presentation_figures.py` and named after the
cases they show. They mirror the Exp-6 panels of the visualizer exactly.

| Figure | Contents |
|--------|----------|
| `ego_similarity_<case>.*` | The case among its 3 closest same-label and 3 closest opposite-label cases; edges labelled with shared evidence by type (`41P, 19V, 10C`) and cosine. |
| `contrast_same_<case>_<neighbour>.*` | Query case vs its closest **same-label** case. |
| `contrast_opposite_<case>_<neighbour>.*` | Query case vs its closest **opposite-label** case. |

Every evidence box carries its counterfactual rank as a `#3 ▲` pill: **▲** = masking that
evidence *lowers* the model's confidence, so it drives the decision; **▼** = masking it *raises*
confidence, so it argues against the decision. A ribbon marks a top-3 driving factor and a
`FLIPS` chip marks evidence whose removal alone changes the predicted label. Under the diagram,
**both** cases get a line naming their three strongest factors — including types (parties,
arguments, lawyers) that get no box, so the diagram doesn't imply the top driver is whichever
box ranks highest. A case outside the test split says so instead of showing a blank, since the
masking experiment was only ever run on test cases.

Each case circle shows **both** its true label and the model's prediction, flagged when they
disagree. This matters: the nearest opposite-*true*-label case gets the **same** prediction from
the model about 93% of the time, so a figure showing ground truth alone would hide the fact that
the model never separated the two cases. The banner above the columns says which it was.

`--match pred` pairs cases on the model's **prediction** instead of the true label — the only
setting where the two cases genuinely got different verdicts, and therefore the only one where
"what drove the difference" is a real question. `--match target` (default) reproduces the
published nearest-opposite-label analysis.

Pass `--detail` for the denser layout used by the paper figures: idf readings plus the full
`CF #3 ▲ Δ0.011` badge on its own row.

## ♻️ Regeneration

From the repository root (writes both PNG and PDF):

```bash
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_paper_figures.py

# Presentation figures for one case (PNG + PDF + SVG):
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_presentation_figures.py \
  --case-index 51419

# Pair on what the model predicted, not on ground truth:
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_presentation_figures.py \
  --case-index 51419 --match pred

# Same figures with idf and Δ values on every box:
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_presentation_figures.py \
  --case-index 51419 --detail

# Not sure which case to show? Rank cases by how legible their figures are:
micromamba run -n thesis_work python FINAL_EXPLANATION/generate_presentation_figures.py --suggest 20
```

Counterfactual badges need `case_counterfactual_factor_index.csv`; build it once with
`build_counterfactual_factor_index.py` (see [`../README.md`](../README.md)).

---

⬆️ Back to [`FINAL_EXPLANATION/`](../README.md)
