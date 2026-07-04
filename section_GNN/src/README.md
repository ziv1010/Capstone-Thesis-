# 🧠 src — Reusable GNN Pipeline Package

> Part of [`section_GNN/`](../README.md) · the Python package every experiment launcher calls into.

`src` centralises preprocessing, graph construction, modelling, training, and utility logic so
the shell wrappers and experiment launchers never duplicate it.

## 📦 Subpackages

| Package | Purpose |
|---------|---------|
| [`preprocessing/`](preprocessing/README.md) | Leakage-safe text-section extraction and entity normalization from raw/cleaned case payloads. |
| [`graph/`](graph/README.md) | Case-star graph construction, global-graph merging, PyG `HeteroData` conversion. |
| [`models/`](models/README.md) | Heterogeneous GNN (HGT-style) classifier and MLP head. |
| [`training/`](training/README.md) | Dataset checks, training loop, evaluation, metrics, and plots. |
| [`utils/`](utils/README.md) | YAML/JSON I/O, portable path resolution, logging, seeding, text encoders. |
| [`scripts/`](scripts/README.md) | Command-line entry points: build, train, K-fold, evaluate, audits. |
| [`visualization/`](visualization/README.md) | Graph visualisation helpers used by post-hoc analysis. |

## 🐍 Import Assumption

Entry points add `section_GNN` to `PYTHONPATH` before importing:

```python
from src.utils.io import load_yaml
```

Running modules manually? Either run from `section_GNN/` or:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

## 🗺️ Config Path Resolution

Always load project configs with `src.utils.io.load_yaml` — it resolves `paths.*` (and known
`inference.*` path fields) relative to `section_GNN`, keeping configs portable across
machines while scripts receive absolute runtime paths.

---

⬆️ Back to [`section_GNN/`](../README.md)
