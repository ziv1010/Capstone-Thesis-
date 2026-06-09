# case_node_minimised

The case-node-minimised ablation reduces how much text or scalar information is
placed directly on the `case` node.

## Purpose

This tests whether the model is relying on the central case node feature vector
instead of learning through section and entity graph structure.

## Run All Buckets

```bash
bash ablations/case_node_minimised/run_case_node_minimised.sh
```

Per-bucket configs live in:

```text
ablations/case_node_minimised/<bucket>/config.yaml
```
