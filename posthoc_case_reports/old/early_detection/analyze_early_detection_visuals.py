#!/usr/bin/env python3
"""Dependency-free analysis and SVG visuals for early-detection CSVs."""
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
DEFAULT_CASES = BASE_DIR / "multi_hearing_case_paths.csv"
DEFAULT_HEARINGS = BASE_DIR / "multi_hearing_hearing_level_analysis.csv"
DEFAULT_OUT = BASE_DIR / "analysis_outputs"
DEFAULT_FIGS = BASE_DIR / "figures"

PALETTE = ["#2A6F97", "#D1495B", "#EDA94A", "#2F7D57", "#6D597A", "#4D908E", "#B56576"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--hearings", default=str(DEFAULT_HEARINGS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIGS))
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


def mean(values: list[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def pred_key(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("WIN") or text == "1":
        return "WIN"
    if text.startswith("LOSE") or text == "-1":
        return "LOSE"
    return text or "UNKNOWN"


def actual_key(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("WIN") or text == "1":
        return "WIN"
    if text.startswith("LOSS") or text.startswith("LOSE") or text == "-1":
        return "LOSS"
    if text.startswith("POSTPONED") or text == "0":
        return "POSTPONED"
    return text or "UNKNOWN"


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
    width = 1080
    left = 300
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
        body.append(f'<text x="{left - 12}" y="{y + 16}" text-anchor="end" class="label">{esc(label[:48])}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="18" rx="3" fill="{PALETTE[i % len(PALETTE)]}"/>')
        body.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 14}" class="small">{value:.2f}{esc(value_suffix)}</text>')
    path.write_text(svg_base(width, height, "\n".join(body)), encoding="utf-8")


def write_grouped_bars(path: Path, title: str, rows: list[dict], label_key_name: str, series: list[tuple[str, str]]) -> None:
    width = 1120
    left = 260
    right = 80
    top = 80
    group_h = 46
    bar_h = 16
    height = max(220, top + group_h * len(rows) + 70)
    chart_w = width - left - right
    body = [f'<text x="24" y="36" class="title">{esc(title)}</text>']
    for idx, (field, label) in enumerate(series):
        body.append(f'<rect x="{left + idx * 170}" y="52" width="12" height="12" fill="{PALETTE[idx]}"/>')
        body.append(f'<text x="{left + idx * 170 + 18}" y="63" class="small">{esc(label)}</text>')
    for i, row in enumerate(rows):
        y0 = top + i * group_h
        body.append(f'<text x="{left - 12}" y="{y0 + 25}" text-anchor="end" class="label">{esc(str(row[label_key_name])[:38])}</text>')
        for j, (field, _) in enumerate(series):
            value = fnum(row.get(field))
            if math.isnan(value):
                value = 0
            y = y0 + j * (bar_h + 4)
            bar_w = chart_w * min(max(value, 0), 100) / 100
            body.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="3" fill="{PALETTE[j]}"/>')
            body.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 12}" class="small">{value:.2f}%</text>')
    path.write_text(svg_base(width, height, "\n".join(body)), encoding="utf-8")


def write_heatmap(path: Path, title: str, matrix: dict[str, dict[str, int]], row_order: list[str], col_order: list[str], row_label: str, col_label: str) -> None:
    cell = 94
    left = 145
    top = 92
    width = left + cell * len(col_order) + 80
    height = top + cell * len(row_order) + 70
    max_v = max([matrix.get(r, {}).get(c, 0) for r in row_order for c in col_order] + [1])
    body = [f'<text x="24" y="36" class="title">{esc(title)}</text>']
    body.append(f'<text x="{left - 8}" y="{top - 38}" text-anchor="end" class="axis">{esc(row_label)}</text>')
    body.append(f'<text x="{left + cell * len(col_order) / 2}" y="{top - 54}" text-anchor="middle" class="axis">{esc(col_label)}</text>')
    for j, col in enumerate(col_order):
        body.append(f'<text x="{left + j * cell + cell / 2}" y="{top - 18}" text-anchor="middle" class="label">{esc(col)}</text>')
    for i, row in enumerate(row_order):
        body.append(f'<text x="{left - 12}" y="{top + i * cell + cell / 2 + 4}" text-anchor="end" class="label">{esc(row)}</text>')
        for j, col in enumerate(col_order):
            value = matrix.get(row, {}).get(col, 0)
            intensity = value / max_v if max_v else 0
            fill = f"#{int(246 - 118 * intensity):02x}{int(248 - 91 * intensity):02x}{int(252 - 58 * intensity):02x}"
            x = left + j * cell
            y = top + i * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" fill="{fill}" stroke="#ffffff"/>')
            body.append(f'<text x="{x + cell / 2 - 2}" y="{y + cell / 2 + 5}" text-anchor="middle" class="label">{value}</text>')
    path.write_text(svg_base(width, height, "\n".join(body)), encoding="utf-8")


def compact_transition(row: dict[str, str]) -> str:
    n = int(row["n_hearings"])
    parts = []
    for i in range(1, n + 1):
        parts.append(pred_key(row.get(f"hearing{i}_prediction", "")))
    return " -> ".join(parts)


def bucket_summary(case_rows: list[dict[str, str]]) -> list[dict]:
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in case_rows:
        by_bucket[row["bucket_raw"]].append(row)
    out = []
    for bucket, rows in sorted(by_bucket.items()):
        n = len(rows)
        out.append({
            "bucket": bucket,
            "n_cases": n,
            "changed_prediction_cases": sum(truthy(r["changed_prediction"]) for r in rows),
            "changed_prediction_pct": pct(sum(truthy(r["changed_prediction"]) for r in rows) / n if n else math.nan),
            "final_prediction_accuracy_pct": pct(sum(truthy(r["final_prediction_correct"]) for r in rows) / n if n else math.nan),
            "winning_case_pct": pct(sum(actual_key(r["case_actual_result"]) == "WIN" for r in rows) / n if n else math.nan),
            "mean_hearing1_confidence": round(mean([fnum(r.get("hearing1_confidence")) for r in rows]), 4),
            "mean_final_confidence": round(mean([fnum(r.get(f"hearing{int(r['n_hearings'])}_confidence")) for r in rows]), 4),
        })
    return out


def hearing_metrics(hearing_rows: list[dict[str, str]]) -> list[dict]:
    by_index: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in hearing_rows:
        by_index[row["hearing_index"]].append(row)
    out = []
    for idx, rows in sorted(by_index.items(), key=lambda kv: int(kv[0])):
        n = len(rows)
        out.append({
            "hearing_index": idx,
            "n_hearings": n,
            "accuracy_vs_binary_target_pct": pct(sum(truthy(r["prediction_correct_against_binary_target"]) for r in rows) / n if n else math.nan),
            "mean_confidence": round(mean([fnum(r["confidence"]) for r in rows]), 4),
            "predicted_win_rate_pct": pct(sum(pred_key(r["prediction"]) == "WIN" for r in rows) / n if n else math.nan),
            "actual_postponed_rate_pct": pct(sum(actual_key(r["actual"]) == "POSTPONED" for r in rows) / n if n else math.nan),
            "actual_win_rate_pct": pct(sum(actual_key(r["actual"]) == "WIN" for r in rows) / n if n else math.nan),
        })
    return out


def transition_counts(case_rows: list[dict[str, str]]) -> list[dict]:
    counts = Counter(compact_transition(row) for row in case_rows)
    total = sum(counts.values())
    return [
        {"prediction_transition": transition, "n_cases": count, "pct_cases": pct(count / total if total else math.nan)}
        for transition, count in counts.most_common()
    ]


def stage1_final_matrix(case_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in case_rows:
        first = pred_key(row.get("hearing1_prediction"))
        final = pred_key(row.get(f"hearing{int(row['n_hearings'])}_prediction"))
        matrix[first][final] += 1
    return {k: dict(v) for k, v in matrix.items()}


def actual_path_counts(case_rows: list[dict[str, str]]) -> list[dict]:
    counts = Counter()
    for row in case_rows:
        n = int(row["n_hearings"])
        parts = [actual_key(row.get(f"hearing{i}_actual_raw", "")) for i in range(1, n + 1)]
        counts[" -> ".join(parts)] += 1
    total = sum(counts.values())
    return [
        {"raw_actual_transition": transition, "n_cases": count, "pct_cases": pct(count / total if total else math.nan)}
        for transition, count in counts.most_common()
    ]


def source_coverage(hearing_rows: list[dict[str, str]]) -> list[dict]:
    counts = Counter(row.get("significance_source", "missing") for row in hearing_rows)
    total = sum(counts.values())
    return [
        {"significance_source": source, "n_hearings": count, "pct_hearings": pct(count / total if total else math.nan)}
        for source, count in counts.most_common()
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    case_rows = read_rows(Path(args.cases))
    hearing_rows = read_rows(Path(args.hearings))

    buckets = bucket_summary(case_rows)
    hmetrics = hearing_metrics(hearing_rows)
    transitions = transition_counts(case_rows)
    actual_paths = actual_path_counts(case_rows)
    sources = source_coverage(hearing_rows)
    matrix = stage1_final_matrix(case_rows)

    write_csv(out_dir / "bucket_summary.csv", buckets, list(buckets[0].keys()))
    write_csv(out_dir / "hearing_index_metrics.csv", hmetrics, list(hmetrics[0].keys()))
    write_csv(out_dir / "prediction_transition_counts.csv", transitions, list(transitions[0].keys()))
    write_csv(out_dir / "raw_actual_transition_counts.csv", actual_paths, list(actual_paths[0].keys()))
    write_csv(out_dir / "significance_source_coverage.csv", sources, list(sources[0].keys()))
    write_csv(out_dir / "stage1_to_final_prediction_matrix.csv", [
        {"stage1_prediction": r, "final_prediction": c, "n_cases": matrix.get(r, {}).get(c, 0)}
        for r in ["LOSE", "WIN"] for c in ["LOSE", "WIN"]
    ], ["stage1_prediction", "final_prediction", "n_cases"])

    write_grouped_bars(
        fig_dir / "bucket_changed_and_final_accuracy.svg",
        "Changed Prediction and Final Accuracy by Bucket",
        buckets,
        "bucket",
        [("changed_prediction_pct", "Changed Prediction"), ("final_prediction_accuracy_pct", "Final Accuracy")],
    )
    write_bar_chart(
        fig_dir / "prediction_transition_counts.svg",
        "Prediction Paths Across Hearings",
        [(r["prediction_transition"], float(r["n_cases"])) for r in transitions],
    )
    # Add a derived percentage field for plotting mean confidence.
    for row in hmetrics:
        row["mean_confidence_pct"] = pct(fnum(row["mean_confidence"]))
    write_grouped_bars(
        fig_dir / "hearing_index_accuracy_confidence.svg",
        "Hearing-Level Accuracy and Confidence",
        hmetrics,
        "hearing_index",
        [("accuracy_vs_binary_target_pct", "Accuracy"), ("mean_confidence_pct", "Mean Confidence")],
    )
    write_heatmap(
        fig_dir / "stage1_to_final_prediction_matrix.svg",
        "Stage 1 to Final Prediction Matrix",
        matrix,
        ["LOSE", "WIN"],
        ["LOSE", "WIN"],
        "Stage 1",
        "Final",
    )
    write_bar_chart(
        fig_dir / "raw_actual_transition_counts.svg",
        "Actual Outcome Paths Across Hearings",
        [(r["raw_actual_transition"], float(r["n_cases"])) for r in actual_paths],
    )
    write_bar_chart(
        fig_dir / "significance_source_coverage.svg",
        "Hearing Significance Source Coverage",
        [(r["significance_source"], float(r["n_hearings"])) for r in sources],
    )

    summary = {
        "case_rows": len(case_rows),
        "hearing_rows": len(hearing_rows),
        "cases_with_changed_prediction": sum(truthy(r["changed_prediction"]) for r in case_rows),
        "n_hearings_distribution": dict(Counter(r["n_hearings"] for r in case_rows)),
        "top_prediction_transition": transitions[0] if transitions else None,
        "tables_dir": str(out_dir),
        "figures_dir": str(fig_dir),
    }
    write_json(out_dir / "analysis_summary.json", summary)
    print(f"Wrote early-detection analysis tables -> {out_dir}")
    print(f"Wrote early-detection SVG figures -> {fig_dir}")
    print(f"Cases with changed predictions: {summary['cases_with_changed_prediction']} / {len(case_rows)}")


if __name__ == "__main__":
    main()
