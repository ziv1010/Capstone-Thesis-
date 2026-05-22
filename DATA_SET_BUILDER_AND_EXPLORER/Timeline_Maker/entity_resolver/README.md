# Entity Resolver

Canonicalizes legal entities inside merged case JSON files before graph construction.

## Scope

The resolver targets three entity families:

- `STATUTE`
- `PROVISION`
- `PRECEDENT`

It leaves other entity labels unchanged, then writes the original document structure with `canonical_id` and `canonical_name` fields added to resolved entities.

## Usage

```bash
python DATA_SET_BUILDER_AND_EXPLORER/Timeline_Maker/entity_resolver/resolve_entities.py \
  --input-root /path/to/merged_jsons \
  --output-root /path/to/resolved_jsons
```

The output includes resolved JSON files plus `_entity_maps/` and `_audit/` metadata. Those generated outputs are not committed.
