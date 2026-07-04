# 🏁 runs_v2 — LR-Control & Party-Text Run Families

> Part of [`section_GNN/`](../README.md) · later BGE-M3 run families sharing the main
> preprocessing/training code but with alternate graph builders or case-node text policies.

## 🗂️ Subfolders

| Folder | Purpose |
|--------|---------|
| `baseline_lr_decay/` | Baseline graph with LR-decay training settings. |
| `party_args_no_lr/` | Party-argument case-node text, no LR decay. |
| [`party_args_lr_decay/`](party_args_lr_decay/README.md) | ⭐ Party-argument case-node text **with** LR decay — hosts the shared v2 builder/trainer. |
| `party_args_preamble_lr_decay/` | Party-argument + preamble case-node text with LR decay. |
| [`no_names_lr_decay/`](no_names_lr_decay/README.md) | No-names v2 ablation with LR decay. |

## 🔗 Shared Scripts

`party_args_lr_decay/` provides the shared v2 machinery reused by the sibling folders:
`graph/build_graph_v2.py`, `scripts/kfold_cv_v2.py`, `03_kfold_v2.sh`, `run_all_buckets.sh`.

## ▶️ Run

```bash
bash runs_v2/party_args_lr_decay/run_all_buckets.sh
bash runs_v2/no_names_lr_decay/run_all_buckets.sh
```

## 📤 Outputs

Generated data and models usually land under `data/timed_bucket_runs/<bucket>/` and
`outputs/timed_bucket_runs/<bucket>/`; some orchestration scripts retarget into
`outputs/ablations/` or `outputs/inlegalbert_*` depending on the thesis table being filled.

---

⬆️ Back to [`section_GNN/`](../README.md)
