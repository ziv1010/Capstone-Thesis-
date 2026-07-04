# ✂️ no_cross_case — Cross-Case Sharing Ablation

> Part of [`ablations/`](../README.md).

The baseline graph shares canonical authority/context nodes **across cases** (the same
statute cited by two cases is one node). This ablation disables that sharing, asking whether
global graph connectivity improves prediction or whether local case-star structure suffices.

## ▶️ Run

```bash
bash ablations/no_cross_case/cross_bucket_total_dataset/run.sh
```

Compare against the matching baseline bucket in `runs/<bucket>/`.

---

⬆️ Back to [`ablations/`](../README.md)
