# 📄 PAPER_DATA — Conference Paper Assets

> Part of [`Latex_Documentation/`](../README.md) · the source and figure assets of the
> accepted conference paper.

## 📄 Contents

| File | Role |
|------|------|
| `main.tex` | Paper LaTeX source. |
| `make_pipeline_diagram.py` | Regenerates the pipeline diagram (`pipeline_diagram.pdf/.png`). |
| `make_early_detection_figure.py` | Regenerates the early-detection signal figure (`early_detection_signals.pdf/.png`) from the multi-hearing results. |
| `gnn_embedding_comparison_full.csv` | Compact table backing the paper's encoder-comparison figures. |

## ♻️ Regenerating Figures

```bash
python Latex_Documentation/PAPER_DATA/make_pipeline_diagram.py
python Latex_Documentation/PAPER_DATA/make_early_detection_figure.py
```

The generated PDFs/PNGs here are small paper artifacts and are kept in Git to support
reproducible paper builds. The early-detection data originates from
[`section_GNN/multi_hearing_stage_test/`](../../section_GNN/multi_hearing_stage_test/README.md).

---

⬆️ Back to [`Latex_Documentation/`](../README.md)
