# fixed_open_pipeline

This folder converts sentence-level fixed-open JSON outputs into the cleaned
case schema used by the graph builders.

## Inputs

Default config:

```text
experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

Default raw input path in that config:

```text
../Fixed_GPU_OpenNyai/fin_fraud_labelled/labelled_jsons
```

Other configs in this folder cover cross-bucket and quick-test workflows.

## Main Script

```text
preprocess_fixed_open.py
```

It reads sentence-level case JSON files and writes:

```text
data/.../processed/cleaned_cases/
data/.../processed/normalized_entities/
data/.../audits/
data/.../processed/preprocess_summary.fixed_open.json
```

## Section Mapping

Typical role mapping:

- `PREAMBLE` -> `preamble`
- `FAC` -> `facts`
- `ARG_PETITIONER` -> `petitioner_arguments` and `arguments`
- `ARG_RESPONDENT` -> `respondent_arguments` and `arguments`
- `PRE_RELIED`, `PRE_NOT_RELIED`, `STA` -> `other_lawyer_arguments` and `arguments`

Roles such as `ANALYSIS`, `ISSUE`, `NONE`, `RATIO`, `RLC`, and `RPC` are
dropped conservatively by default.

## Run

From `section_GNN`:

```bash
micromamba run -n thesis_work python experiments/fixed_open_pipeline/preprocess_fixed_open.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
```

Then build/train with the normal graph scripts:

```bash
micromamba run -n thesis_work python final_graph/build_graph.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml

micromamba run -n thesis_work python src/scripts/train_gnn.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml \
  --run-name fin_fraud_labelled_reasoning
```
