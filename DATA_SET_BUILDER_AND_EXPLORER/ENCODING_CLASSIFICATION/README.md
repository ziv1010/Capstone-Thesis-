# Case Embedding and Cluster Explorer

`cluster_case_buckets.py` reads the five legal case text buckets, creates one embedding per case, clusters the full corpus, and writes figures plus CSVs that show where buckets are cleanly separated and where they overlap.

Default embedding model:
- `mixedbread-ai/mxbai-embed-large-v1`

Why this default:
- It is available directly through `sentence-transformers`.
- Its Hugging Face model card reports stronger overall and clustering benchmark scores than `BAAI/bge-large-en-v1.5`.
- It supports Matryoshka truncation, so the script defaults to `--truncate-dim 512` to keep storage and clustering lighter.

## Run

Use the existing micromamba environment for this repo:

```bash
bash /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/run_cluster_case_buckets.sh --device cuda
```

Equivalent explicit command:

```bash
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/old_scripts_pt2/GNN/.micromamba/gnn_case_star \
  python /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/cluster_case_buckets.py \
  --device cuda
```

Multi-GPU behavior:
- `--multi-gpu auto` is the default.
- If `--device cuda` and more than one CUDA device is visible, the embedder fans out across all visible GPUs with `sentence-transformers` multi-process encoding.
- To force a subset, pass `--gpu-devices`, for example `--gpu-devices cuda:0 cuda:1 cuda:2 cuda:3`.
- To stay on one GPU, use `--multi-gpu off --device cuda:0`.

If you want a faster smoke test first:

```bash
bash /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/run_cluster_case_buckets.sh \
  --max-files-per-bucket 200 \
  --model-name sentence-transformers/all-MiniLM-L6-v2 \
  --truncate-dim 0 \
  --device cpu
```

If the default model is too heavy, use:

```bash
--model-name BAAI/bge-base-en-v1.5 --truncate-dim 0
```

## Outputs

The script writes into `ENCODING_CLASSIFICATION/outputs/` by default:

- `cases_with_clusters.csv`: one row per case with bucket, cluster, 2D projection, and nearest cross-bucket match
- `case_embeddings.npy`: one embedding per case
- `cluster_bucket_counts.csv`: raw cluster composition by bucket
- `cluster_bucket_proportions.csv`: normalized cluster composition by bucket
- `cluster_exemplars.csv`: the most centroid-like filenames inside each cluster
- `bucket_centroid_similarity.csv`: bucket-to-bucket cosine similarity using centroid embeddings
- `bucket_neighbor_overlap_fraction.csv`: how often a case's closest cross-bucket neighbor comes from each other bucket
- `bucket_neighbor_overlap_similarity.csv`: mean similarity of those cross-bucket matches
- `top_cross_bucket_pairs.csv`: strongest cross-bucket case pairs
- `run_report.json`: run configuration and summary
- `figures/*.png`: scatter plots and heatmaps for quick inspection

## Extra Visualization Scripts

These scripts run in the same micromamba environment and use the saved outputs from the main clustering run.

3D PCA views:

```bash
bash /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/run_plot_case_embeddings_3d.sh \
  --input-dir /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/outputs
```

This writes:
- `figures_3d/case_projection_3d.csv`
- `figures_3d/scatter3d_by_bucket_isometric.png`
- `figures_3d/scatter3d_by_bucket_top.png`
- `figures_3d/scatter3d_by_bucket_side.png`
- `figures_3d/scatter3d_by_cluster_isometric.png`
- `figures_3d/scatter3d_by_cluster_top.png`
- `figures_3d/scatter3d_by_cluster_side.png`

Interactive 3D HTML:

```bash
bash /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/run_build_interactive_3d_plot.sh \
  --input-dir /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/outputs
```

This writes:
- `figures_interactive/interactive_3d_pca.html`
- `figures_interactive/interactive_3d_pca_report.json`
- `figures_interactive/interactive_3d_projection_pca.csv`

Useful options:
- `--method pca` for a faster global view
- `--method tsne --max-points 8000` for stronger local visual separation
- `--max-points 0` to plot every point, if your browser can handle it

Sampled t-SNE views:

```bash
bash /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/run_plot_case_embeddings_tsne.sh \
  --input-dir /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/outputs \
  --sample-size 8000
```

This writes:
- `figures_tsne/tsne_projection_2d.csv`
- `figures_tsne/tsne_projection_3d.csv`
- `figures_tsne/tsne_2d_by_bucket.png`
- `figures_tsne/tsne_2d_by_cluster.png`
- `figures_tsne/tsne_3d_by_bucket_*.png`
- `figures_tsne/tsne_3d_by_cluster_*.png`

When to use which:
- 3D PCA is useful when you want a faithful, global view of the embedding geometry.
- Interactive 3D HTML is useful when you want to rotate, zoom, and inspect hover labels for individual cases.
- t-SNE is better when you want clusters to separate visually, but it is local and sample-based, so do not treat distances there as globally reliable.
- For this dataset, t-SNE 2D is usually more useful than a static 3D scatter for interpretation.

## Notes

- The script embeds each case by chunking long texts, embedding each chunk, then mean-pooling the chunk embeddings.
- `--reuse-embeddings` lets you skip re-encoding and rerun only the clustering/plotting stage from saved embeddings.
- If `faiss` is installed, nearest-neighbor overlap uses it automatically. Otherwise small runs use scikit-learn brute-force cosine search, and large runs fall back to a stratified sample controlled by `--neighbor-analysis-sample-size`.
- On this machine, invoking the wrapper through `bash .../run_cluster_case_buckets.sh` is more reliable than executing it directly.
- On this machine, `micromamba` sees 8 CUDA devices, so the default `--device cuda` path now uses all 8 for embedding unless you disable multi-GPU mode.
