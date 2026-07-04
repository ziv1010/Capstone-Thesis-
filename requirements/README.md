# 🧪 requirements — Environment Index

> Supporting folder · a single place to see **which micromamba environment belongs to which
> workflow**, without searching every shell script.

The `.txt` files are human-readable package inventories, **not** strict cross-machine lock
files. Prefer the original YAML specs where they exist
(`Fixed_GPU_OpenNyai/environment.gpu.yml`, `FINAL_EXPLANATION/traceability_reports_env.yml`,
`model comparison/environment.yml`, `GRAPH_VISUALISER/setup_env.sh`).

## ⭐ Core Environments

| Inventory | Environment | Used by |
|-----------|-------------|---------|
| `fixed_gpu_opennyai_final.txt` | `fixed_gpu_opennyai_final` | Stage ② OpenNyAI NER/RR + summary GPU pipeline. |
| `llm.txt` | `llm` | Stage ② Mistral/vLLM outcome labelling. |
| `case_merge.txt` | `case_merge` | Stage ③ lightweight timeline/case merging. |
| `thesis_work.txt` | `thesis_work` | Stage ④ `section_GNN` (preprocess, graphs, training) + Stage ⑤ `FINAL_EXPLANATION` analyses & paper figures. |
| `graph_vis.txt` | `graph_vis` | Dash visualisers: Graph Visualiser, Stage Visualiser, Multi-Hearing Stage Test Visualiser. |
| `hgt_trace_reports.txt` | `hgt_trace_reports` | Stage ⑤ traceability report generation. |
| `model_comparison_inlegalllama.txt` | `model_comparison_inlegalllama` | Legal-LLM comparison and adapter evaluation. |

## 🗄️ Legacy / Local Helper Environments

| Inventory | Notes |
|-----------|-------|
| `graph_explainer_legacy.txt` | Legacy `Graph_Analyser` explainer pipeline (now archived in `DUMP_MISC/`); some old scripts still reference it. |
| `pdf_extract_local.txt` | Repo-local `.micromamba/pdf_extract` utility environment — matches the PDF-extraction dependency surface, not referenced by launch scripts. |

## ♻️ Re-Exporting Exact Installed State

```bash
micromamba list -n thesis_work --export
micromamba list -n graph_vis --export
micromamba list -n fixed_gpu_opennyai_final --export
micromamba list -n llm --export
micromamba list -n model_comparison_inlegalllama --export
micromamba list -n hgt_trace_reports --export
micromamba list -n case_merge --export
micromamba list -n graph_explainer --export
micromamba list -p .micromamba/pdf_extract --export
```

Environment files under `DUMP_MISC/` are archive material and are intentionally not promoted
here unless an active script outside the archive references them.

---

⬆️ Back to the [repository root](../README.md)
