# ⏱️ Timeline_Maker — Case Merging & Final Datasets

> Part of [`DATA_SET_BUILDER_AND_EXPLORER/`](../README.md) · **Stage ③** of the pipeline.

Merges per-hearing / per-stage case JSONs into consolidated **case timelines**, assembles the
cross-bucket corpora, and hosts the entity resolver that produces the final entity-resolved
dataset used by the GNN experiments and the Graph Visualiser.

---

## 🛠️ Key Scripts

| Script | Purpose |
|--------|---------|
| `merge_cases.py` / `merge_cases_v2.py` / **`merge_cases_v3.py`** | Merge related case records into consolidated timelines (v3 is the current version). |
| `build_cross_bucket_dataset.py` | Assemble cross-domain datasets from bucket-specific merged outputs (report: `cross_bucket_total_dataset_report.json`). |
| `compare_merges.py` | Compare merge versions and audit differences. |
| [`entity_resolver/resolve_entities.py`](entity_resolver/README.md) | Canonicalize statute, provision, and precedent references. |
| `visualiser.html` / `visualiser_dual.html` | Static browser viewers for inspecting merged case timelines. |

---

## 🗂️ Layout

| Folder | Status | Contents |
|--------|:------:|----------|
| `output_merged_v3/` | 📤 generated | Current **unresolved** merged timeline corpus used by downstream GNN runs. |
| `output_merged_v3_resolved/` | 📤 generated · ⭐ | **Entity-resolved** corpus — the preferred final dataset for the GNN entity-resolved runs and the Graph Visualiser. |
| `<bucket>_timed_mistral/` | 📤 generated | Per-bucket merged datasets written by `Fixed_GPU_OpenNyai/run_scripts/run_merge_timeline_from_final_outputs.sh`. |
| `old_outputs/` | 🗄️ archive | Older merge outputs kept for comparison audits. |
| `dump/` | 🗄️ archive | Superseded intermediate outputs. |
| `entity_resolver/` | ✅ active | Entity canonicalization tool (+ its `logs/`). |

> These output folders stay at this exact location because `section_GNN` configs reference
> them by relative path (`../DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/...`).

---

## ▶️ Usage

Merge one bucket's labelled output (example):

```bash
cd DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker
python merge_cases_v3.py \
  --input ../../Fixed_GPU_OpenNyai/final_outputs/fin_fraud_labelled_mistral/labelled_jsons \
  --output output_merged_v3/fin_fraud
```

Resolve entities over the merged corpus:

```bash
python entity_resolver/resolve_entities.py \
  --input-root output_merged_v3 \
  --output-root output_merged_v3_resolved
```

Browse timelines with the static viewers:

```bash
python3 -m http.server 8080      # then open visualiser.html / visualiser_dual.html
```

---

## 📦 Generated Data

The merged corpora are large and ignored by Git — regenerate them from the scripts above or
restore from external storage. Only scripts, the dataset report JSON, the HTML viewers, and
documentation are versioned.

---

⬆️ Back to [`DATA_SET_BUILDER_AND_EXPLORER/`](../README.md) · Next stage: [`section_GNN/`](../../section_GNN/README.md)
