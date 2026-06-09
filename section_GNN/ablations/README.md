# ablations

`ablations` contains controlled variants of the baseline graph/model pipeline.
The goal is to isolate which information sources and graph design choices drive
performance.

## Common Layout

Most ablation families use one folder per bucket:

```text
ablations/<variant>/<bucket>/
  config.yaml
  run.sh
```

Some generated/config-only variants omit `run.sh` and are launched through
top-level orchestration scripts.

## Main Variants

| Folder | Question Tested |
| --- | --- |
| `text_only/` | What happens if the graph uses only case/text-section nodes? |
| `no_names/` | How much do identity/name-bearing nodes contribute? |
| `no_cross_case/` | How much does sharing nodes across cases help? |
| `hierarchical_enc/` | Does hierarchical text encoding improve graph features? |
| `section_sep_enc/` | Does separating section embeddings help? |
| `section_sep_enc_lr_decay/` | Section-separated graph with LR-decay training settings. |
| `case_node_minimised/` | How much can case-node text/features be reduced? |
| `depth/` | Sensitivity to GNN depth/layer count. |
| `entity_resolved_data/` | Uses externally resolved entity data. |
| `remove_central_authorities/` | Filters overly central authority nodes and reruns selected variants. |

## Running Ablations

Run an individual bucket:

```bash
bash ablations/text_only/cross_bucket_total_dataset/run.sh
```

Run larger groups through top-level launchers:

```bash
bash run_scripts/run_complete_ablation_matrix.sh
bash run_scripts/run_remaining_non_cross_bucket_ablations.sh
```

## Outputs

Most ablation outputs are written under:

```text
outputs/timed_bucket_runs/<bucket>/
outputs/ablations/<variant>/
```

The exact path is controlled by each `config.yaml`.

## Adding a New Ablation

1. Copy the closest existing variant config.
2. Keep `paths.*` relative to `section_GNN`.
3. Change only the config fields needed for the ablation.
4. Add a short README or update this file if the variant introduces a new
   assumption.
5. Run one small bucket or `--limit` test before launching the full matrix.
