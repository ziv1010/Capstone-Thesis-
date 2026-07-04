# 🪞 depth — GNN Depth Ablation

> Part of [`ablations/`](../README.md).

Varies the number of GNN layers — measuring whether performance depends on shallow local
aggregation or deeper **multi-hop message passing**.

## 🗂️ Layout

Each bucket folder contains e.g. `config_depth1.yaml`, `config_depth2.yaml`,
`config_depth3.yaml`, and a `run.sh`.

## ▶️ Run

```bash
bash ablations/depth/fin_fraud_timed_mistral/run.sh
```

The run script reuses the matching baseline graph cache and changes only the model
depth/training config.

---

⬆️ Back to [`ablations/`](../README.md)
