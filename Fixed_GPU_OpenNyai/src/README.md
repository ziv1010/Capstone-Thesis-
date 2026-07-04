# 🧩 src — Shared Pipeline Modules

> Part of [`Fixed_GPU_OpenNyai/`](../README.md) · support code for the Stage ② scripts.

Helper modules imported by the top-level extraction, summarization, and labelling scripts.
Users should normally run the top-level scripts or the wrappers in
[`../run_scripts/`](../run_scripts/README.md) rather than calling these modules directly.

| Module | Responsibility |
|--------|----------------|
| `config.py` | Configuration and repository-relative path handling. |
| `io_utils.py` | JSON and text I/O helpers (safe reads, atomic writes). |
| `output_formatter.py` | Canonical output JSON structure for annotations and summaries. |
| `pipeline_runner.py` | OpenNyAI pipeline orchestration — batching, GPU checks, per-document fault isolation, version-compatible kwargs filtering. |
| `validators.py` | Input/output validation checks (empty/short files, malformed JSON). |

---

⬆️ Back to [`Fixed_GPU_OpenNyai/`](../README.md)
