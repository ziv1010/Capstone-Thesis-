# 🕶️ no_names — Identity-Masking Ablation

> Part of [`ablations/`](../README.md).

Removes or masks name-bearing information so the model cannot rely on party, lawyer, judge,
or other identity strings — testing whether performance is driven by **reusable legal
structure and case text** rather than memorized named entities.

## ▶️ Run

```bash
bash ablations/no_names/fin_fraud_timed_mistral/run.sh
```

Outputs stay under each bucket's configured `outputs_dir`. The v2 LR-decay version lives at
[`runs_v2/no_names_lr_decay/`](../../runs_v2/no_names_lr_decay/README.md).

---

⬆️ Back to [`ablations/`](../README.md)
