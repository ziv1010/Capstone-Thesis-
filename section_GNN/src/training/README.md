# 🏋️ src/training — Training & Evaluation Logic

> Part of [`section_GNN/src/`](../README.md).

| File | Role |
|------|------|
| `dataset.py` | Graph label/split validation and dataset helpers. |
| `train.py` | Training loop with early stopping, optimizer, and LR-scheduler handling. |
| `evaluate.py` | Split evaluation and prediction collection. |
| `metrics.py` | Metric computation + plots (training history, split bars, confusion matrices). |

## 📜 Output Contract

Training functions return a dictionary with the trained model state, a predictions
DataFrame, per-split metric dictionaries (train/validation/test), and the training history.
Script wrappers persist all of it under `outputs/.../models/<run_name>/`.

---

⬆️ Back to [`src/`](../README.md)
