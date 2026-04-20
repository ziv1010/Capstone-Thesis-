from __future__ import annotations

from pathlib import Path
from typing import Any


def build_eda_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# EDA Report")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Total cases: **{summary['n_cases']}**")
    lines.append(f"- Cases with leakage_flag=true: **{summary['leakage_true_pct']:.2f}%**")
    lines.append(
        f"- Cases with non-empty decision_text: **{summary['decision_text_nonempty_pct']:.2f}%**"
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("- Tables: `outputs/eda/tables/*.csv`")
    lines.append("- Figures: `outputs/eda/figures/*.png`")
    lines.append("")
    lines.append("## Key Tables")
    for table_name in summary.get("table_files", []):
        lines.append(f"- `{table_name}`")

    lines.append("")
    lines.append("## Key Figures")
    for fig_name in summary.get("figure_files", []):
        lines.append(f"- `{fig_name}`")

    return "\n".join(lines) + "\n"


def write_eda_report(text: str, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
