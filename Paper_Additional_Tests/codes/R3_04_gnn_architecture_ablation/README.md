# R3-04 — HGT vs. simpler GNN architectures

> **Reviewer 3:** *"The paper uses HGT [8] but does not compare against simpler GNNs (GCN,
> GraphSAGE, GAT) on the same graph. It is unclear whether the complexity of heterogeneous
> attention is necessary or a simpler model would suffice."*

## What this does

Re-runs the message-passing operator — and nothing else — on the paper's own graph, folds and
trainer. Seven rungs, so the answer separates *graph* from *relation-awareness* from *attention*:

| Config | Message passing | Graph | Rel.-aware | Attention |
|---|---|---|---|---|
| `mlp` | per-type `Linear` (no message passing) | no | no | no |
| `gcn` | `GCNConv` on the type-collapsed graph | yes | no | no |
| `sage` | `SAGEConv` on the type-collapsed graph | yes | no | no |
| `gat` | `GATConv` on the type-collapsed graph | yes | no | untyped |
| `rgcn` | `HeteroConv{relation: SAGEConv}` | yes | yes | no |
| `hgat` | `HeteroConv{relation: GATConv}` | yes | yes | per-relation |
| `hgt` | `HGTConv` — **the published run, read from disk, not re-run** | yes | yes | typed |

"Type-collapsed" means: after the per-node-type input projection, all 17 node types are
concatenated and all 42 typed relations are merged into one untyped edge set — the same graph
with the type information removed. It is also the only way to run a true `GCNConv`, which
cannot consume PyG's bipartite `(src, dst)` channel form and so cannot sit inside `HeteroConv`.

Held fixed across every row: the 6.3 GB cached graph tensor, the five stratified folds
(`StratifiedKFold(5, shuffle=True, random_state=42)`, 72/8/20, seeds 42–46), hidden 64,
2 layers, 4 heads, dropout 0.25, `MLPHead(64,128,2)`, AdamW 1e-3/1e-5, class-weighted
cross-entropy, 90 epochs, early stopping patience 20 on validation macro-F1,
`ReduceLROnPlateau(0.5, patience 8, min 1e-6)`.

## Nothing in `section_GNN/` is modified

`kfold_arch_cv.py` rebinds `train_v2.HeteroLegalOutcomeGNN` to `ArchLegalOutcomeGNN` and then
calls `kfold_cv_v2.main()`. `train_v2.train_model` resolves that name as a module global at
call time, so the whole split / training / metric / aggregation stack runs unmodified. For
`architecture: hgt`, `ArchLegalOutcomeGNN` delegates to the original class, so it is provably
a superset of the paper model rather than a re-implementation.

## Run it

```bash
# static equivalence gate (no training, ~20 s)
micromamba run -n thesis_work python check_harness_equivalence.py

# full sweep: 6 architectures x 5 folds, one fold per GPU (~1 h)
bash run_arch_ablation.sh --gpus 1,2,4,5,6

# subset / smoke test
bash run_arch_ablation.sh --only gcn --folds 0

# tables + report
micromamba run -n thesis_work python make_tables.py
```

`check_harness_equivalence.py` is how the harness is validated *without* re-running HGT:

1. recomputing the folds reproduces every `fold_XX/predictions.csv` exactly;
2. each `configs/arch_*.yaml` differs from the paper config in only `model.architecture` and
   `paths.outputs_dir`;
3. `ArchLegalOutcomeGNN(architecture="hgt")` reproduces the saved `fold_00/model.pt` — all 381
   tensors, identical shapes, 2,011,508 parameters;
4. the graph cache is the same file the reference run recorded.

## Files

| File | Purpose |
|---|---|
| `arch_gnn.py` | `ArchLegalOutcomeGNN` — the paper scaffold with a swappable conv operator |
| `kfold_arch_cv.py` | injects the model into `train_v2`, delegates to `kfold_cv_v2.main()` |
| `make_configs.py` | generates `configs/arch_*.yaml` from the paper config |
| `check_harness_equivalence.py` | the four static checks above |
| `run_arch_ablation.sh` | architecture × fold sweep, one fold per GPU |
| `make_tables.py` | LaTeX table, markdown report, McNemar tests vs. HGT |
| `outputs/models/arch_*_kfold/kfold/` | per-fold `metrics.json`, `predictions.csv`, `model.pt` |
| `outputs/table_r3_04_architectures.tex` | drop-in LaTeX table |
| `outputs/report_r3_04.md` | results, significance tests, suggested paper changes |

## Significance testing

Because all runs share the same five folds and those folds partition the corpus, pooling each
run's test-split rows gives exactly one held-out prediction per case — 71,813 paired
predictions per comparison. `make_tables.py` reports an exact McNemar test on those pairs,
plus a paired t-test over the five per-fold macro-F1 values.

Environment: `thesis_work` (torch 2.5.1+cu124, PyG 2.6.1) — the env that produced the paper numbers.
