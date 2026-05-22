# Saved patch — preamble/facts mention edges

These two files are **patched copies** of the section_GNN graph builder that
add new edge types from `preamble` and `facts` to `statute`/`provision`/`precedent`,
distinct from the existing `arguments --[cites_*]--> *` edges.

Without this patch, cases that mention statutes/provisions/precedents only in
their preamble or facts (rather than in arguments text) end up with **zero**
legal citation edges in the graph. Example: case_69 (498A/304B Anadi Pandey)
had `statute_count=1` ("indian penal code") in its raw JSON metadata but zero
graph edges to any legal node.

## What changed

### `case_star_builder.PATCHED.py`

1. Two new tracked section keys near the top of the function:
   ```python
   preamble_node_key: str | None = None
   facts_node_key: str | None = None
   ```
2. They are populated alongside `arguments_node_key` when those sections are
   created.
3. A new edge-emission block runs for every `statute`/`provision`/`precedent`
   entity, independent of the existing arguments-only branch:
   - `preamble --[mentions_statute]--> statute` (and provision / precedent)
   - `facts --[mentions_statute]--> statute` (and provision / precedent)

   Section presence is derived from `entity.mentions[].section` (lowercased)
   plus the existing `entity.seen_in_preamble` flag.

### `schema.PATCHED.py`

Six new entries added to `RELATION_DEFINITIONS` to register the new relations.

## Applying the patch

```bash
cp saved_for_later/case_star_builder.PATCHED.py \
   ../section_GNN/src/graph/case_star_builder.py
cp saved_for_later/schema.PATCHED.py \
   ../section_GNN/src/graph/schema.py
```

## Required follow-up after applying

The HGT model checkpoint at
`section_GNN/outputs/.../cross_bucket_party_args_lr_decay_kfold/kfold/fold_00/`
does **not** have parameters for the new edge types. Without retraining, the
loader at `Graph_Analyser/analyser/loader.py:91-92` will raise
`RuntimeError: Missing keys when loading HGT state_dict`.

Required full sequence:

1. Apply the two patches above.
2. Rebuild the graph cache:
   ```bash
   cd /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-
   micromamba run -n graph_explainer python -m section_GNN.src.scripts.build_graph \
     --config section_GNN/runs_v2/party_args_lr_decay/cross_bucket_total_dataset/config.yaml
   ```
3. Retrain the GNN (single fold suffices for Graph_Analyser, fold 0 only):
   ```bash
   cd /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN
   CUDA_VISIBLE_DEVICES=0 micromamba run -n graph_explainer python src/scripts/kfold_cv.py \
     --config runs_v2/party_args_lr_decay/cross_bucket_total_dataset/config.yaml \
     --run-name cross_bucket_party_args_lr_decay_kfold --fold 0
   ```
4. Re-run Graph_Analyser pipeline.

## Verified behaviour (before reverting)

case_69 went from **0** legal edges → **1** new edge:
`preamble --[mentions_statute]--> statute (indian penal code)`
