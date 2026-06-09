# no_cross_case

The no-cross-case ablation prevents cross-case sharing of graph nodes.

## Purpose

The baseline graph can share canonical authority/context nodes across cases.
This ablation asks whether that global graph connectivity improves prediction
or whether local case-star structure is sufficient.

## Run Example

```bash
bash ablations/no_cross_case/cross_bucket_total_dataset/run.sh
```

Compare results against the matching baseline bucket in `runs/<bucket>/`.
