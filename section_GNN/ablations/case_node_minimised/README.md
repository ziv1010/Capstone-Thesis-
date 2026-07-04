# 🎯 case_node_minimised — Minimal Case-Node Ablation

> Part of [`ablations/`](../README.md).

Reduces how much text/scalar information sits directly on the central `case` node — testing
whether the model relies on the case-node feature vector instead of **learning through the
section and entity graph structure**.

## ▶️ Run

```bash
bash ablations/case_node_minimised/run_case_node_minimised.sh
```

Per-bucket configs: `ablations/case_node_minimised/<bucket>/config.yaml`.

---

⬆️ Back to [`ablations/`](../README.md)
