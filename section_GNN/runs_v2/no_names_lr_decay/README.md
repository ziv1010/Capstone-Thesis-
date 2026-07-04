# 🕶️ no_names_lr_decay — V2 No-Names Ablation

> Part of [`runs_v2/`](../README.md) · removes name-bearing features/nodes under LR-decay training.

Uses the shared v2 builder/trainer from
[`../party_args_lr_decay/`](../party_args_lr_decay/README.md) but swaps in configs that strip
identity information (party, lawyer, judge names), testing whether the v2 gains survive
without identity signals.

## ▶️ Run

```bash
bash runs_v2/no_names_lr_decay/run_all_buckets.sh
```

Outputs follow each per-bucket config's `paths.outputs_dir`.

---

⬆️ Back to [`runs_v2/`](../README.md)
