# 🧰 src/utils — Shared Infrastructure

> Part of [`section_GNN/src/`](../README.md) · used by nearly every entry point.

| File | Role |
|------|------|
| `io.py` | JSON/YAML helpers, directory creation, deep-merge, and **portable config path resolution** (`load_yaml`). |
| `logging_utils.py` | File + console logger setup. |
| `pipeline.py` | Shared graph-building pipeline utilities. |
| `seed.py` | Deterministic seed setup for reproducible runs. |
| `text_encoder.py` | Text-encoder wrappers: sentence-transformers (BGE-M3), Hugging Face (InLegalBERT, …), and a hashing fallback. |

## 🗺️ Why `load_yaml` Matters

Use `load_yaml` instead of raw `yaml.safe_load` for project configs: it expands relative
config paths against `section_GNN`, and `dump_yaml`/`dump_json` write repository-local paths
back out where possible. **This is what keeps every config reproducible across machines.**

---

⬆️ Back to [`src/`](../README.md)
