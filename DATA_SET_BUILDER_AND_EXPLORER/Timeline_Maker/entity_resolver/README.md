# 🔗 entity_resolver — Legal Entity Canonicalization

> Part of [`Timeline_Maker/`](../README.md) · **Stage ③** of the pipeline.

Canonicalizes legal entities inside merged case JSONs **before graph construction**, so that
the same statute/provision/precedent cited in different spellings becomes a single shared
node in the GNN graph.

## 🎯 Scope

The resolver targets three entity families and leaves all other labels untouched:

| Entity family | Example |
|---------------|---------|
| `STATUTE` | *Indian Penal Code* ≡ *IPC* |
| `PROVISION` | *Section 302 IPC* ≡ *S. 302, I.P.C.* |
| `PRECEDENT` | cited case names in varying citation formats |

The original document structure is preserved; resolved entities gain `canonical_id` and
`canonical_name` fields.

## ▶️ Usage

```bash
python DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/entity_resolver/resolve_entities.py \
  --input-root  DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3 \
  --output-root DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/output_merged_v3_resolved
```

## 📤 Outputs

- Resolved JSON files mirroring the input tree (→ `output_merged_v3_resolved/`).
- `_entity_maps/` — the canonicalization mappings used.
- `_audit/` — audit metadata for verifying resolution quality.
- `logs/` — resolution run logs.

Generated outputs are not committed. The resolved corpus feeds the `entity_resolved_data`
ablation in `section_GNN` and the final explanation pipeline.

---

⬆️ Back to [`Timeline_Maker/`](../README.md)
