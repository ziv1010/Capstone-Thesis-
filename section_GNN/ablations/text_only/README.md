# text_only

The text-only ablation removes legal-entity and authority-node structure so the
model mainly sees case and text-section information.

## Purpose

This tests whether performance comes from text encodings alone or from the
heterogeneous legal graph structure.

## Typical Nodes

The configs restrict the graph to nodes such as:

- `case`
- `preamble`
- `facts`
- `arguments`
- party-specific argument sections where configured

Entity nodes such as statutes, provisions, precedent, judge, lawyer, petitioner,
and respondent are removed or disabled.

## Run Example

```bash
bash ablations/text_only/cross_bucket_total_dataset/run.sh
```

Each bucket folder has a `config.yaml`; some have a `config_hashing.yaml` for
hashing-encoder tests.
