# 📤 final_outputs — Stage ② Generated Artifacts

> Part of [`Fixed_GPU_OpenNyai/`](../README.md) · **generated data** — large and mostly ignored by Git.

The main production artifacts of the OpenNyAI + Mistral pipeline. Each legal bucket
progresses through up to three output stages:

```text
<bucket>_extract/annotations/                    ← 1️⃣ OpenNyAI NER + rhetorical-role JSONs
<bucket>_summary_opennyai/enriched_jsons/        ← 2️⃣ + OpenNyAI summary fields
<bucket>_labelled_mistral/labelled_jsons/        ← 3️⃣ + Mistral outcome labels  (final)
```

## 🪣 Buckets

| Bucket | extract | summary | labelled |
|--------|:-------:|:-------:|:--------:|
| `family_matrimonial` | ✅ | ✅ | ✅ |
| `fin_fraud` | ✅ | ✅ | ✅ |
| `land_property` | ✅ | ✅ | ✅ |
| `motor_accidents` | ✅ | ✅ | ✅ |
| `sexual_offences` | ✅ | ✅ | ✅ |
| `food_safety` | ✅ | ✅ | — (held-out cross-domain bucket; labelled downstream when needed) |

A bucket may be missing a later stage if that stage has not been run or was not part of the
final selected dataset.

## 🧹 Runtime Cache Folders

Parallel extraction runs create per-worker runtime homes such as
`<bucket>_extract/.worker_home_0/`. These contain CUDA/CuPy/OpenNyAI model caches — **not**
thesis artifacts. Deleting them never deletes completed annotations, summaries, or labels;
reruns simply re-download caches.

## ➡️ Downstream

The `labelled_jsons/` folders feed
`../run_scripts/run_merge_timeline_from_final_outputs.sh`, which writes the merged
`*_timed_mistral/` datasets into `DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/`.

---

⬆️ Back to [`Fixed_GPU_OpenNyai/`](../README.md)
