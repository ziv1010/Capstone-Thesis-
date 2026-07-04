# 📋 posthoc_case_reports — Case-Level Report Generation

> Supporting analysis · converts explanation and multi-hearing outputs into **case-level
> CSVs and aggregate reports** that connect model explanations to case-stage timelines.
> Use after the Stage ④ (`section_GNN`) and Stage ⑤ (`FINAL_EXPLANATION`) outputs exist.

## 🗂️ Structure

| Path | Role |
|------|------|
| `convert_current_outputs_to_csv.py` | Converts current explanation outputs into tabular report inputs. |
| [`current/`](current/README.md) | 📤 Current generated case-level reports. |
| [`analysis/`](analysis/README.md) | 📤 Generated aggregate analysis outputs. |
| [`timeline_merger/`](timeline_merger/README.md) | Stage/timeline merge outputs — influence & connectivity summaries, stage factors. |
| [`old/`](old/README.md) | 🗄️ Older aggregate-analysis and early-detection scripts. |

## 📦 Git Policy

Generated report folders are ignored — keep only scripts and compact documentation in Git,
and regenerate reports when the upstream GNN, timeline, or explanation outputs change.

---

⬆️ Back to the [repository root](../README.md)
