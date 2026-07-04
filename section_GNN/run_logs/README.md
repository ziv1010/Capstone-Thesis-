# 🧾 run_logs — Experiment Launcher Logs

> Part of [`section_GNN/`](../README.md) · 📤 **generated artifacts** — deletable and
> regenerable.

Logs from long-running experiment launchers. Recommended pattern (from `section_GNN/`):

```bash
nohup bash run_scripts/<script>.sh > run_logs/<script>.log 2>&1 &
tail -f run_logs/<script>.log
```

Name logs after the launcher or experiment family, e.g.
`run_logs/run_inlegalbert_experiments.log`,
`run_logs/remove_central_authorities_fin_fraud.log`.

Logs help diagnose failed folds, GPU-assignment issues, and missing graph caches — but the
**authoritative metrics** are the `kfold_summary.json` files under
[`outputs/`](../outputs/README.md).

---

⬆️ Back to [`section_GNN/`](../README.md)
