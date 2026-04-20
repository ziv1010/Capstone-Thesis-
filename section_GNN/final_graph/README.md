# Updated Graph

This folder contains a separate reasoning-focused graph variant so the original `src/graph` pipeline stays untouched.

## What Changed

Removed nodes:

- `org`
- `gpe`
- `date`
- `case_number`

Removed shortcut edges:

- `statute -> used_in_arguments -> arguments`
- `provision -> used_in_arguments -> arguments`
- `judge -> presided_arguments -> arguments`
- `petitioner -> is_party_in_arguments -> arguments`
- `respondent -> is_party_in_arguments -> arguments`
- `petitioner_lawyer -> citation -> arguments`
- `defence_lawyer -> citation -> arguments`
- `lawyer -> citation -> arguments`

Kept edges:

- `case -> has_* -> text / party / court / judge / lawyer nodes`
- `arguments -> cites_statute / cites_provision / cites_precedent`
- `provision -> belongs_to_statute -> statute`
- `petitioner_lawyer -> petitioner_arguments`
- `defence_lawyer -> respondent_arguments`
- `lawyer -> other_lawyer_arguments`
- `petitioner -> petitioner_arguments`
- `respondent -> respondent_arguments`

## Files

- `updated_graph/reasoning_graph_policy.py`: enforced node and edge policy
- `updated_graph/case_star_builder.py`: updated local case-star builder
- `updated_graph/pipeline.py`: graph bundle assembly using the updated builder
- `build_graph.py`: standalone entrypoint for this variant
- `graph_config_template.yaml`: graph section to merge into a dataset-specific config later

## Later Use

When you are ready to parse/build the dataset for this graph:

```bash
python "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/section_GNN/updated graph/build_graph.py" \
  --config /path/to/your/dataset_specific_config.yaml
```

The builder writes separate `*.reasoning_focused.*` metadata snapshots plus a `reasoning_focused_case_star_graph.pt` cache name by default.
