# Fixed-Open Dataset Size Sweep

This folder contains a runner for a three-way dataset-size sweep built on the `updated graph` pipeline.

What it does:

- loads the cleaned cases from the config
- skips integrity-failing cleaned cases by default and records them
- keeps the last 1000 usable cases as the fixed test set
- runs three experiments with requested train sizes `2500`, `5000`, and `all`
- builds a separate reasoning-focused graph bundle for each experiment
- trains the GNN on each graph and writes a combined summary

Default command:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 \
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/dataset_size_sweep/run_fixed_open_dataset_size_sweep.py" \
  --config "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/fixed_open_reasoning_config.yaml"
```

Dry-run only:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 \
micromamba run -p /scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/GNN/.micromamba/gnn_case_star \
  python "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/dataset_size_sweep/run_fixed_open_dataset_size_sweep.py" \
  --config "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/fixed_open_pipeline/fixed_open_reasoning_config.yaml" \
  --dry-run
```

Outputs:

- per-run graph caches: `section_GNN/data/fixed_open_binary/graph_cache/dataset_size_sweep/<experiment_name>/<run_key>/`
- per-run training outputs: `section_GNN/outputs/fixed_open_binary/dataset_size_sweep/<experiment_name>/<run_key>/`
- combined sweep summary: `section_GNN/outputs/fixed_open_binary/dataset_size_sweep/<experiment_name>/summary.csv`
