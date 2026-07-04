# 🪣 timed_mistral_configs — Per-Bucket Preprocessing Configs

> Part of [`fixed_open_pipeline/`](../README.md).

Dataset-specific fixed-open configs for the five timed Mistral buckets:

- `family_matrimonial_timed_mistral.yaml`
- `fin_fraud_timed_mistral.yaml`
- `land_property_timed_mistral.yaml`
- `motor_accidents_timed_mistral.yaml`
- `sexual_offences_timed_mistral.yaml`

## ▶️ Usage

```bash
python experiments/fixed_open_pipeline/preprocess_fixed_open.py \
  --config experiments/fixed_open_pipeline/timed_mistral_configs/fin_fraud_timed_mistral.yaml
```

All configs follow the standard `section_GNN` relative-path convention.

---

⬆️ Back to [`fixed_open_pipeline/`](../README.md)
