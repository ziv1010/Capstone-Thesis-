# 🧹 remove_central_authorities — Hub-Filtering Ablation

> Part of [`ablations/`](../README.md).

Very common statutes, provisions, or precedents become **high-degree hubs** in the global
graph. This ablation filters those central authority nodes before training — testing whether
performance depends on hubs or whether they inject noise/shortcut behaviour.

## 📄 Main Files

| File | Role |
|------|------|
| `analyze_central_authorities.py` | Identifies high-centrality authority nodes. |
| `filter_cleaned_cases.py` | Removes selected central authorities from cleaned cases. |
| `prepare_configs.py` | Generates configs for the filtered runs. |
| `run_remove_central_authorities_ablation.sh` | Main launcher. |

## 🗂️ Configs & Outputs

Configs: `configs/` and `configs_no_lr/`.
Outputs: `outputs/ablations/remove_central_authorities/` and
`data/ablations/remove_central_authorities/`.

---

⬆️ Back to [`ablations/`](../README.md)
