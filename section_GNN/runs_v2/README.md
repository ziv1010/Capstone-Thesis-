# runs_v2

`runs_v2` contains later BGE-M3 run families used for LR-control and party-text
experiments. These runs share the main preprocessing and training code but use
alternate graph builders or case-node text definitions.

## Subfolders

| Folder | Purpose |
| --- | --- |
| `baseline_lr_decay/` | Baseline graph with LR-decay training settings. |
| `party_args_no_lr/` | Party-argument case-node text without LR decay. |
| `party_args_lr_decay/` | Party-argument case-node text with LR decay. |
| `party_args_preamble_lr_decay/` | Party-argument plus preamble case-node text with LR decay. |
| `no_names_lr_decay/` | No-name v2 ablation with LR decay. |

## Shared Scripts

`party_args_lr_decay/` contains shared v2 scripts:

- `graph/build_graph_v2.py`
- `scripts/kfold_cv_v2.py`
- `03_kfold_v2.sh`
- `run_all_buckets.sh`

Other v2 folders often call these shared scripts with their own configs.

## Run All Buckets

```bash
bash runs_v2/party_args_lr_decay/run_all_buckets.sh
bash runs_v2/no_names_lr_decay/run_all_buckets.sh
```

## Output Layout

Generated data and outputs usually land under:

```text
data/timed_bucket_runs/<bucket>/
outputs/timed_bucket_runs/<bucket>/
```

Some orchestration scripts may retarget outputs into `outputs/ablations/` or
`outputs/inlegalbert_*` depending on the table being generated.
