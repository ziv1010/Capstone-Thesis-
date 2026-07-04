# 🔭 embedding_analysis — Post-Hoc Embedding & Probing Tools

> Part of [`section_GNN/`](../README.md) · diagnostic tools for trained graphs, embeddings,
> and predictions.

## 📄 Scripts

| Script | Purpose |
|--------|---------|
| `extract_embeddings.py` | Exports learned or cached embeddings for analysis. |
| `tsne_visualise.py` | t-SNE projections of case embeddings. |
| `probing_classifier.py` | Trains simple probes on embedding features. |
| `shap_analysis.py` | SHAP-style feature attribution. |
| `attention_analysis.py` | Attention/importance analysis helpers. |
| `node_importance.py` | Node-level importance analysis. |
| `error_analysis.py` | Error slicing and inspection. |
| `run_analysis.sh` | Shell wrapper for a standard analysis pass. |

## 📥 Inputs

These scripts need artifacts from a completed run:

- a graph cache from `data/.../graph_cache/`
- a trained checkpoint from `outputs/.../models/.../`
- predictions/metadata from the same run

## ▶️ Run

From `section_GNN/`:

```bash
bash embedding_analysis/run_analysis.sh
```

⚠️ Check and edit the config paths inside the script before pointing it at a different
bucket or model. For the final counterfactual explanation pipeline, use
[`FINAL_EXPLANATION/`](../../FINAL_EXPLANATION/README.md) instead.

---

⬆️ Back to [`section_GNN/`](../README.md)
