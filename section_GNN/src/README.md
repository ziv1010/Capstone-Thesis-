# src

`src` is the reusable Python package behind the `section_GNN` experiments. The
shell wrappers and experiment launchers call into this package rather than
duplicating preprocessing, graph construction, training, and evaluation logic.

## Subfolders

| Folder | Purpose |
| --- | --- |
| `preprocessing/` | Extracts leakage-safe text sections and normalized entity records from raw/cleaned case payloads. |
| `graph/` | Builds local case-star graphs, merges them into global heterogeneous graphs, and converts them to PyG `HeteroData`. |
| `models/` | Defines the hetero GNN model and MLP classifier head. |
| `training/` | Dataset checks, training loop, evaluation metrics, and plotting helpers. |
| `utils/` | YAML/JSON I/O, path resolution, logging, seeds, text encoders, and shared pipeline helpers. |
| `scripts/` | Command-line entry points for graph build, training, evaluation, and audits. |
| `visualization/` | Graph visualisation helpers used by post-hoc analysis scripts. |

## Import Assumption

Most entry points add `section_GNN` to `PYTHONPATH` before importing:

```python
from src.utils.io import load_yaml
```

If you run modules manually, either run from `section_GNN` or set:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## Config Path Resolution

Use `src.utils.io.load_yaml` for project configs. It resolves `paths.*` and
known `inference.*` path fields relative to `section_GNN`, so configs can stay
portable while scripts receive absolute runtime paths.
