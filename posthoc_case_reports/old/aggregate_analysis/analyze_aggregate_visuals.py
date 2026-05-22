#!/usr/bin/env python3
"""Dependency-free analysis and SVG visuals for aggregate test-set CSV."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "aggregate_test_set_analysis.csv"
DEFAULT_OUT = BASE_DIR / "analysis_outputs"
DEFAULT_FIGS = BASE_DIR / "figures"

NODE_COLUMNS = [
    "statutes_significant",
    "provisions_significant",
    "precedents_significant",
    "case_ids_significant",
    "judges_significant",
    "courts_significant",
    "lawyers_significant",
    "petitioners_significant",
    "respondents_significant",
]

PALETTE = ["#2A6F97", "#D1495B", "#EDA94A", "#2F7D57", "#6D597A", "#4D908E", "#B56576"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIGS))
    parser.add_argument("--top-n-nodes", type=int, default=20)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def fnum(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def pct(value: float) -> float:
    if math.isnan(value):
        return math.nan
    return round(100.0 * value, 2)


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def label_key(value: str) -> str:
    text = str(value or "").strip()
    if text in {"1", "WIN (1)", "WIN"}:
        return "WIN"
    if text in {"-1", "LOSE (-1)", "LOSS (-1)", "LOSE", "LOSS"}:
        return "LOSE"
    if text in {"0", "POSTPONED (0)", "POSTPONED"}:
        return "POSTPONED"
    return text or "UNKNOWN"


def split_node_entries(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value:
        return []
    if "[" not in value:
        return [part.strip() for part in value.split(";") if part.strip()]
    matches = re.findall(r"(.+?\[[^\]]+\])(?:;\s+|$)", value)
    if matches:
        return [m.strip() for m in matches]
    return [part.strip() for part in value.split(";") if part.strip()]


def node_name(entry: str) -> str:
    return re.sub(r"\s+\[[^\]]+\]\s*$", "", entry).strip()


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_base(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}'
        '.title{font-size:22px;font-weight:700}.axis{font-size:12px;fill:#52616b}'
        '.label{font-size:12px}.small{font-size:11px;fill:#52616b}</style>\n'
        f"{body}\n</svg>\n"
    )


def write_bar_chart(path: Path, title: str, rows: list[tuple[str, float]], *, value_suffix: str = "", max_value: float | None = None) -> None:
    rows = rows[:30]
    width = 1050
    left = 285
    right = 70
    top = 70
    row_h = 28
    height = max(170, top + 45 + row_h * len(rows))
    chart_w = width - left - right
    max_v = max_value if max_value is not None else max([v for _, v in rows] + [1.0])
    body = [f'<text x="24" y="36" class="title">{esc(title)}</text>']
    body.append(f'<line x1="{left}" y1="{top - 10}" x2="{left}" y2="{height - 35}" stroke="#d8dee4"/>')
    for i, (label, value) in enumerate(rows):
        y = top + i * row_h
        bar_w = 0 if max_v == 0 else chart_w * (value / max_v)
        color = PALETTE[i % len(PALETTE)]
        body.append(f'<text x="{left - 12}" y="{y + 16}" text-anchor="end" class="label">{esc(label[:44])}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" rx="3" fill="{color}"/>')
        body.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 14}" class="small">{value:.2f}{esc(value_suffix)}</text>')
    path.write_text(svg_base(width, height, "\n".join(body)), encoding="utf-8")


def write_heatmap(path: Path, title: str, matrix: dict[str, dict[str, int]], row_order: list[str], col_order: list[str]) -> None:
    cell = 92
    left = 140
    top = 90
    width = left + cell * len(col_order) + 60
    height = top + cell * len(row_order) + 60
    max_v = max([matrix.get(r, {}).get(c, 0) for r in row_order for c in col_order] + [1])
    body = [f'<text x="24" y="36" class="title">{esc(title)}</text>']
    body.append(f'<text x="{left - 8}" y="{top - 36}" text-anchor="end" class="axis">Actual</text>')
    body.append(f'<text x="{left + cell * len(col_order) / 2}" y="{top - 52}" text-anchor="middle" class="axis">Prediction</text>')
    for j, col in enumerate(col_order):
        body.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 18}" text-anchor="middle" class="label">{esc(col)}</text>')
    for i, row in enumerate(row_order):
        body.append(f'<text x="{left - 12}" y="{top + i * cell + cell / 2 + 4}" text-anchor="end" class="label">{esc(row)}</text>')
        for j, col in enumerate(col_order):
            value = matrix.get(row, {}).get(col, 0)
            intensity = value / max_v if max_v else 0
            blue = int(245 - 125 * intensity)
            fill = f"#{blue:02x}{int(248 - 96 * intensity):02x}{int(252 - 62 * intensity):02x}"
            x = left + j * cell
            y = top + i * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" fill="{fill}" stroke="#ffffff"/>')
            body.append(f'<text x="{x + cell / 2 - 2}" y="{y + cell / 2 + 5}" text-anchor="middle" class="label">{value}</text>')
    path.write_text(svg_base(width, height, "\n".join(body)), encoding="utf-8")


def mean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def bucket_metrics(rows: list[dict[str, str]]) -> list[dict]:
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket_raw"]].append(row)

    out = []
    for bucket, items in sorted(by_bucket.items()):
        n = len(items)
        correct = sum(truthy(r["correct"]) for r in items)
        confs = [fnum(r["confidence"]) for r in items]
        high_conf_wrong = sum((not truthy(r["correct"])) and fnum(r["confidence"]) >= 0.8 for r in items)
        out.append({
            "bucket": bucket,
            "n_cases": n,
            "accuracy_pct": pct(correct / n if n else math.nan),
            "mean_confidence": round(mean(confs), 4),
            "pred_win_rate_pct": pct(sum(label_key(r["prediction"]) == "WIN" for r in items) / n if n else math.nan),
            "actual_win_rate_pct": pct(sum(label_key(r["actual"]) == "WIN" for r in items) / n if n else math.nan),
            "high_conf_wrong_cases": high_conf_wrong,
            "high_conf_wrong_rate_pct": pct(high_conf_wrong / n if n else math.nan),
        })
    return out


def confusion_matrix(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[label_key(row["actual"])][label_key(row["prediction"])] += 1
    return {k: dict(v) for k, v in matrix.items()}


def node_coverage(rows: list[dict[str, str]]) -> list[dict]:
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket_raw"]].append(row)

    out = []
    for bucket, items in sorted(by_bucket.items()):
        n = len(items)
        for col in NODE_COLUMNS:
            count = sum(bool(str(r.get(col, "")).strip()) for r in items)
            out.append({
                "bucket": bucket,
                "node_type": col.replace("_significant", ""),
                "cases_with_node_type": count,
                "coverage_pct": pct(count / n if n else math.nan),
            })
    return out


def top_nodes(rows: list[dict[str, str]], top_n: int) -> list[dict]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        bucket = row["bucket_raw"]
        for col in NODE_COLUMNS:
            node_type = col.replace("_significant", "")
            for entry in split_node_entries(row.get(col, "")):
                name = node_name(entry)
                if name:
                    counts[(bucket, node_type, name)] += 1

    all_counts: Counter[tuple[str, str]] = Counter()
    for (_, node_type, name), count in counts.items():
        all_counts[(node_type, name)] += count

    rows_out = [
        {"scope": "overall", "bucket": "ALL", "node_type": nt, "node": node, "n_cases": count}
        for (nt, node), count in all_counts.most_common(top_n)
    ]
    for bucket in sorted({r["bucket_raw"] for r in rows}):
        bucket_counts = Counter({(nt, node): count for (b, nt, node), count in counts.items() if b == bucket})
        rows_out.extend(
            {"scope": "bucket", "bucket": bucket, "node_type": nt, "node": node, "n_cases": count}
            for (nt, node), count in bucket_counts.most_common(top_n)
        )
    return rows_out


def confidence_bins(rows: list[dict[str, str]]) -> list[dict]:
    bins = [("0.50-0.60", 0.50, 0.60), ("0.60-0.70", 0.60, 0.70), ("0.70-0.80", 0.70, 0.80), ("0.80-0.90", 0.80, 0.90), ("0.90-1.00", 0.90, 1.01)]
    out = []
    for label, lo, hi in bins:
        items = [r for r in rows if lo <= fnum(r["confidence"]) < hi]
        n = len(items)
        out.append({
            "confidence_bin": label,
            "n_cases": n,
            "accuracy_pct": pct(sum(truthy(r["correct"]) for r in items) / n if n else math.nan),
            "wrong_cases": sum(not truthy(r["correct"]) for r in items),
        })
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(Path(args.input))
    metrics = bucket_metrics(rows)
    matrix = confusion_matrix(rows)
    coverage = node_coverage(rows)
    nodes = top_nodes(rows, args.top_n_nodes)
    conf_bins = confidence_bins(rows)

    write_csv(out_dir / "bucket_metrics.csv", metrics, list(metrics[0].keys()))
    write_csv(out_dir / "confusion_matrix.csv", [
        {"actual": actual, "prediction": pred, "n_cases": matrix.get(actual, {}).get(pred, 0)}
        for actual in ["LOSE", "WIN"] for pred in ["LOSE", "WIN"]
    ], ["actual", "prediction", "n_cases"])
    write_csv(out_dir / "node_type_coverage_by_bucket.csv", coverage, list(coverage[0].keys()))
    write_csv(out_dir / "top_significant_nodes.csv", nodes, ["scope", "bucket", "node_type", "node", "n_cases"])
    write_csv(out_dir / "confidence_bins.csv", conf_bins, list(conf_bins[0].keys()))

    write_bar_chart(
        fig_dir / "accuracy_by_bucket.svg",
        "Aggregate Accuracy by Bucket",
        [(m["bucket"], float(m["accuracy_pct"])) for m in metrics],
        value_suffix="%",
        max_value=100,
    )
    write_heatmap(fig_dir / "confusion_heatmap.svg", "Aggregate Confusion Matrix", matrix, ["LOSE", "WIN"], ["LOSE", "WIN"])
    overall_coverage = Counter()
    for row in rows:
        for col in NODE_COLUMNS:
            if str(row.get(col, "")).strip():
                overall_coverage[col.replace("_significant", "")] += 1
    write_bar_chart(
        fig_dir / "node_type_coverage_overall.svg",
        "Cases with Significant Nodes by Type",
        [(k, v) for k, v in overall_coverage.most_common()],
    )
    write_bar_chart(
        fig_dir / "top_significant_nodes_overall.svg",
        "Top Significant Nodes Across Test Set",
        [(r["node"][:58], float(r["n_cases"])) for r in nodes if r["scope"] == "overall"][: args.top_n_nodes],
    )
    write_bar_chart(
        fig_dir / "confidence_bin_accuracy.svg",
        "Accuracy by Confidence Bin",
        [(r["confidence_bin"], float(r["accuracy_pct"]) if r["accuracy_pct"] == r["accuracy_pct"] else 0) for r in conf_bins],
        value_suffix="%",
        max_value=100,
    )

    summary = {
        "input": str(Path(args.input)),
        "n_cases": len(rows),
        "overall_accuracy_pct": pct(sum(truthy(r["correct"]) for r in rows) / len(rows)),
        "bucket_metrics_csv": str(out_dir / "bucket_metrics.csv"),
        "confusion_matrix_csv": str(out_dir / "confusion_matrix.csv"),
        "node_type_coverage_csv": str(out_dir / "node_type_coverage_by_bucket.csv"),
        "top_nodes_csv": str(out_dir / "top_significant_nodes.csv"),
        "figures_dir": str(fig_dir),
    }
    write_json(out_dir / "analysis_summary.json", summary)
    print(f"Wrote aggregate analysis tables -> {out_dir}")
    print(f"Wrote aggregate SVG figures -> {fig_dir}")
    print(f"Overall accuracy: {summary['overall_accuracy_pct']}% over {len(rows)} cases")


if __name__ == "__main__":
    main()
