# timed_mistral_configs

Dataset-specific fixed-open configs for the five timed Mistral buckets.

## Buckets

- `family_matrimonial_timed_mistral.yaml`
- `fin_fraud_timed_mistral.yaml`
- `land_property_timed_mistral.yaml`
- `motor_accidents_timed_mistral.yaml`
- `sexual_offences_timed_mistral.yaml`

## Usage

These configs can be passed to the fixed-open preprocessing script:

```bash
python experiments/fixed_open_pipeline/preprocess_fixed_open.py \
  --config experiments/fixed_open_pipeline/timed_mistral_configs/fin_fraud_timed_mistral.yaml
```

They follow the same relative-path convention as other `section_GNN` configs.
