# 🧼 fixed_open_pipeline — Sentence-Level → Cleaned-Case Preprocessing

> Part of [`experiments/`](../README.md) · the **standard preprocessing entry point** that
> converts sentence-level fixed-open JSON outputs into the cleaned-case schema used by all
> graph builders.

## 📄 Main Script

```text
preprocess_fixed_open.py
```

Reads sentence-level case JSONs and writes:

```text
data/.../processed/cleaned_cases/
data/.../processed/normalized_entities/
data/.../audits/
data/.../processed/preprocess_summary.fixed_open.json
```

## 🏷️ Rhetorical-Role → Section Mapping

| OpenNyAI role | Mapped section |
|---------------|----------------|
| `PREAMBLE` | `preamble` |
| `FAC` | `facts` |
| `ARG_PETITIONER` | `petitioner_arguments` + `arguments` |
| `ARG_RESPONDENT` | `respondent_arguments` + `arguments` |
| `PRE_RELIED`, `PRE_NOT_RELIED`, `STA` | `other_lawyer_arguments` + `arguments` |
| `ANALYSIS`, `ISSUE`, `NONE`, `RATIO`, `RLC`, `RPC` | 🛡️ **dropped** (leakage-conservative default) |

## ⚙️ Configs

- `fixed_open_reasoning_config.yaml` — default config (raw input:
  `../Fixed_GPU_OpenNyai/fin_fraud_labelled/labelled_jsons`).
- [`timed_mistral_configs/`](timed_mistral_configs/README.md) — per-bucket configs for the
  five timed Mistral buckets.
- Additional configs cover cross-bucket and quick-test workflows.

## ▶️ Run

From `section_GNN/`:

```bash
micromamba run -n thesis_work python experiments/fixed_open_pipeline/preprocess_fixed_open.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml

# then the usual build + train:
micromamba run -n thesis_work python final_graph/build_graph.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml
micromamba run -n thesis_work python src/scripts/train_gnn.py \
  --config experiments/fixed_open_pipeline/fixed_open_reasoning_config.yaml \
  --run-name fin_fraud_labelled_reasoning
```

---

⬆️ Back to [`experiments/`](../README.md)
