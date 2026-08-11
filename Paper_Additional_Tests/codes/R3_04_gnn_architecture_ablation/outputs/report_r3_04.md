# R3-04 — HGT vs. simpler GNN architectures

> *Reviewer 3: "The paper uses HGT [8] but does not compare against simpler GNNs (GCN,
> GraphSAGE, GAT) on the same graph. It is unclear whether the complexity of heterogeneous
> attention is necessary or a simpler model would suffice."*

## Protocol

Every row uses the **same cached graph tensor**, the **same five stratified folds**
(`StratifiedKFold(5, shuffle=True, random_state=42)`, 72/8/20, seeds 42–46) and the **same
trainer** (`train_v2.train_model`: AdamW 1e-3/1e-5, class-weighted cross-entropy, 90 epochs,
early stopping patience 20 on validation macro-F1, `ReduceLROnPlateau`). Hidden size 64,
2 layers, 4 heads, dropout 0.25, `MLPHead(64,128,2)` are fixed. **Only the convolution
operator changes.**

GCN / GraphSAGE / GAT are applied to the *type-collapsed* view of the same graph: after the
per-node-type input projection, all 17 node types are concatenated and all 42 typed relations
are merged into one untyped edge set. That is the honest reading of "a simpler model on the
same graph" — no relation-specific parameters, no type-aware attention. It is also the only
way to run a true `GCNConv`, which cannot consume PyG's bipartite `(src, dst)` form.

**The HGT row is the published run, read from its own `kfold_summary.json`. It was not re-run.**
Harness equivalence was instead proven statically by `check_harness_equivalence.py`, which
verifies that the folds reproduce `predictions.csv` exactly, that each config differs from the
paper config in only `model.architecture` and `paths.outputs_dir`, and that the injected model
class reproduces the saved checkpoint's 381 tensors with identical shapes.

## Results (5-fold mean ± population s.d.)

| Architecture | Acc (%) | Macro-F1 | HGT − this (acc pts) | HGT − this (F1) | McNemar p | paired-t p |
|---|---:|---:|---:|---:|---:|---:|
| MLP (no graph) | 77.69 | 0.7698 | +2.94 | +0.0304 | 3.30e-103 | 8.78e-05 |
| GCN | 80.27 | 0.7965 | +0.36 | +0.0036 | 0.001 | 0.150 |
| GraphSAGE | 79.40 | 0.7862 | +1.23 | +0.0140 | 5.28e-27 | 0.002 |
| GAT | 80.59 | 0.7997 | +0.04 | +0.0004 | 0.733 | 0.835 |
| R-GCN (relational SAGE) | 80.12 | 0.7960 | +0.51 | +0.0041 | 5.18e-06 | 0.223 |
| Relational GAT | 81.07 | 0.8032 | -0.44 | -0.0031 | 1.45e-04 | 0.154 |
| **HGT (published)** | **80.63** | **0.8002** | — | — | — | — |

McNemar is the exact two-sided test over the 71,813 paired held-out predictions obtained by
pooling each run's five disjoint test folds (every case is held out exactly once). The paired
t-test is over the five per-fold macro-F1 values.

## Verdict

- **The graph is doing real work.** Removing message passing entirely (MLP on the same case features, no edges) costs 2.94 accuracy points and 0.0304 macro-F1 (McNemar p = 3.30e-103). This is the largest single effect in the table.
- **Heterogeneous attention is not what carries the accuracy.** The best simple GNN (GAT on the type-collapsed graph — no relation-specific parameters, no node-type awareness, 1,281,922 parameters vs. HGT's 2,011,508) is not statistically distinguishable from HGT: +0.04 accuracy points, McNemar p = 0.733.
- **Statistically tied with HGT at p > 0.05:** GAT.
- **One configuration outperforms the published HGT:** Relational GAT (81.07%, +0.44 pts, McNemar p = 1.45e-04). We report this rather than suppress it. It does not change the paper's substantive claim — the graph representation, not the specific operator, is what produces the ~80% result — but the accuracy-based justification for choosing HGT specifically should be dropped.

## Suggested change to the paper

The ablation supports a *representation* claim, not an *operator* claim. Concretely:

1. Keep the headline result — every graph-based operator lands in the 79.4–81.1% band, so the
   ~80% number is a property of the case-graph representation, not of HGT.
2. Replace the accuracy-based justification for HGT in Section 3 ("We use HGT because the graph
   contains distinct node and relation types...") with the explainability justification: typed
   attention is what makes the relation-level counterfactual interventions of Section 4.2
   expressible at all. A type-collapsed GCN or GAT cannot produce a per-relation attribution.
3. Delete the Limitations sentence "We do not compare against ... simpler GNNs" and cite this
   table instead.
4. State plainly that a simpler untyped GAT matches HGT on accuracy. A reviewer who runs this
   ablation themselves will find the same thing, and reporting it first is the stronger position.
