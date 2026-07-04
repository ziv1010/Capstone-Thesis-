# 🧱 DATA_SET_BUILDER_AND_EXPLORER — Stage ③ · Dataset Construction

> **Pipeline position:** ① INPUT_DATA ▸ ② Fixed_GPU_OpenNyai ▸ **③ DATA_SET_BUILDER_AND_EXPLORER** ▸ ④ section_GNN ▸ ⑤ FINAL_EXPLANATION

This stage converts the labelled OpenNyAI outputs from Stage ② into the **thesis-ready case
datasets** consumed by the GNN experiments: merged case timelines, combined cross-bucket
corpora, and entity-resolved variants.

The active machinery lives entirely in one subfolder:

| Subfolder | Purpose |
|-----------|---------|
| [`Timeline_Maker/`](Timeline_Maker/README.md) | Case merging (`merge_cases*.py`), cross-bucket dataset assembly, merge audits, static timeline viewers, and the entity resolver. |

> 🗄️ Earlier exploration tools that used to live here (embedding-space bucket clustering,
> Nyaya bucket builders) have been archived under `DUMP_MISC/` and are not part of the
> active pipeline.

---

## 🔄 Typical Flow

1. **Merge** the per-bucket labelled JSONs produced by Stage ② into consolidated case
   timelines with `Timeline_Maker/merge_cases_v3.py` (the `*_timed_mistral/` inputs are
   created by `Fixed_GPU_OpenNyai/run_scripts/run_merge_timeline_from_final_outputs.sh`).
2. **Combine** buckets into cross-domain corpora with
   `Timeline_Maker/build_cross_bucket_dataset.py`.
3. **Resolve entities** — canonicalize statutes, provisions, and precedents with
   `Timeline_Maker/entity_resolver/resolve_entities.py` to produce
   `output_merged_v3_resolved/`, the preferred final dataset.
4. **Hand off** to [`section_GNN/`](../section_GNN/README.md), whose configs reference these
   folders by relative path.

---

## 📦 Data Policy

Generated corpora (`output_merged_v3/`, `output_merged_v3_resolved/`, `old_outputs/`, the
bucket-specific `*_timed_mistral/` datasets) are large and **ignored by Git**. Scripts,
configs, and documentation are versioned; regenerate or restore the datasets locally when
running experiments.

---

⬆️ Back to the [repository root](../README.md) · Previous: [`Fixed_GPU_OpenNyai/`](../Fixed_GPU_OpenNyai/README.md) · Next: [`section_GNN/`](../section_GNN/README.md)
