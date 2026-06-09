# section_sep_enc

The section-separated encoding ablation keeps separate embeddings for major
case sections instead of collapsing them into one case text representation.

## Purpose

This tests whether the model benefits from preserving distinctions between
preamble, facts, arguments, and party-specific argument sections.

## Main Builder

The section-separated graph path uses:

```text
final_graph/build_graph_section_sep.py
src/graph/pyg_builder_section_sep.py
```

## Run Example

```bash
bash ablations/section_sep_enc/cross_bucket_total_dataset/run.sh
```

or all buckets:

```bash
bash runs/run_section_sep_enc_all_buckets.sh
```
