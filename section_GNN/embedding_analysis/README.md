# embedding_analysis

Post-hoc analysis tools for trained graphs, embeddings, and predictions.

## Scripts

- `extract_embeddings.py`: exports learned or cached embeddings for analysis.
- `tsne_visualise.py`: produces t-SNE visualisations.
- `probing_classifier.py`: trains simple probes on embedding features.
- `shap_analysis.py`: SHAP-style feature analysis.
- `attention_analysis.py`: attention/importance analysis helpers.
- `node_importance.py`: node-level importance analysis.
- `error_analysis.py`: error slicing and inspection.
- `run_analysis.sh`: shell wrapper for a standard analysis pass.

## Inputs

These scripts usually need:

- a graph cache from `data/.../graph_cache/`
- a trained model checkpoint from `outputs/.../models/.../`
- predictions or metadata from the same run

## Run

From `section_GNN`:

```bash
bash embedding_analysis/run_analysis.sh
```

Check and edit the config paths in the script before running on a different
bucket or model.
