# no_names

The no-names ablation removes or masks name-bearing information so the model
cannot rely directly on party, lawyer, judge, or other identity strings.

## Purpose

This tests whether model performance is driven by reusable legal structure and
case text rather than memorizing named entities.

## Run Example

```bash
bash ablations/no_names/fin_fraud_timed_mistral/run.sh
```

Generated outputs usually stay under the bucket's configured `outputs_dir`.
