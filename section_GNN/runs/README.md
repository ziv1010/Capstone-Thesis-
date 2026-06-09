# runs

`runs` contains the baseline BGE-M3 timed-bucket experiment definitions. Each
bucket folder has the same shape:

```text
runs/<bucket>/
  config.yaml
  01_preprocess.sh
  02_build_graph.sh
  03_kfold_8gpu.sh
  run_all.sh
```

## Buckets

- `family_matrimonial_timed_mistral`
- `fin_fraud_timed_mistral`
- `land_property_timed_mistral`
- `motor_accidents_timed_mistral`
- `sexual_offences_timed_mistral`
- `cross_bucket_total_dataset`

## How to Run One Bucket

From `section_GNN`:

```bash
bash runs/fin_fraud_timed_mistral/run_all.sh
```

or step by step:

```bash
bash runs/fin_fraud_timed_mistral/01_preprocess.sh
bash runs/fin_fraud_timed_mistral/02_build_graph.sh
bash runs/fin_fraud_timed_mistral/03_kfold_8gpu.sh
```

## Outputs

Baseline generated data lands under:

```text
data/timed_bucket_runs/<bucket>/
outputs/timed_bucket_runs/<bucket>/
```

K-fold summaries are usually:

```text
outputs/timed_bucket_runs/<bucket>/models/<run_name>/kfold/kfold_summary.json
```

## Cross-Bucket Helpers

- `run_hierarchical_enc_all_buckets.sh`: runs the hierarchical encoding ablation.
- `run_section_sep_enc_all_buckets.sh`: runs the section-separated encoding ablation.
