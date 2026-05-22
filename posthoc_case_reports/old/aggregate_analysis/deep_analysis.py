#!/usr/bin/env python3
"""
Deep aggregate analysis for the GNN test set.  (v2 — improved)

Produces 7 focused, publication-ready SVG figures:
  1.  calibration_curve.svg            – confidence vs actual accuracy
  2.  accuracy_by_bucket.svg           – clean per-bucket accuracy bars
  3.  win_rate_bias_by_bucket.svg      – predicted vs actual WIN rate per bucket
  4.  jurisdiction_accuracy.svg        – best/worst courts by accuracy
  5.  node_evidence_coverage.svg       – citation rate for LEGAL-EVIDENCE nodes: WIN vs LOSE
  6.  node_structural_coverage.svg     – citation rate for STRUCTURAL nodes: WIN vs LOSE
  7.  citation_bias_courts_judges.svg  – court & judge citation bias: WIN rate when cited as significant

Run from this directory:
    python3 deep_analysis.py
"""
from __future__ import annotations

import csv
import html
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "aggregate_test_set_analysis.csv"
FIGS_DIR = BASE_DIR / "figures_v2"
DATA_DIR = BASE_DIR / "analysis_outputs_v2"

LEGAL_NODE_COLS = [
    "statutes_significant", "provisions_significant",
    "precedents_significant", "case_ids_significant",
]
STRUCTURAL_NODE_COLS = [
    "judges_significant", "courts_significant",
    "lawyers_significant", "petitioners_significant", "respondents_significant",
]
ALL_NODE_COLS = LEGAL_NODE_COLS + STRUCTURAL_NODE_COLS

LEGAL_LABELS = {
    "statutes_significant": "Statutes",
    "provisions_significant": "Provisions",
    "precedents_significant": "Precedents",
    "case_ids_significant": "Similar Cases",
}
STRUCTURAL_LABELS = {
    "judges_significant": "Judges",
    "courts_significant": "Courts",
    "lawyers_significant": "Lawyers",
    "petitioners_significant": "Petitioners",
    "respondents_significant": "Respondents",
}

BUCKET_DISPLAY = {
    "family_matrimonial": "Family / Matrimonial",
    "financial":          "Financial Fraud",
    "land":               "Land & Property",
    "motor_accidents":    "Motor Accidents",
    "sexual_offences":    "Sexual Offences",
}

C_BLUE   = "#2A6F97"
C_RED    = "#C53B3B"
C_ORANGE = "#D9841A"
C_GREEN  = "#2F7D57"
C_PURPLE = "#6D597A"
C_TEAL   = "#4D908E"
C_GREY   = "#8A9BB0"
PALETTE  = [C_BLUE, C_RED, C_ORANGE, C_GREEN, C_PURPLE, C_TEAL, C_GREY]


# ── helpers ──────────────────────────────────────────────────────────────────

def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def fnum(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def mean(vals: list[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def is_correct(row: dict) -> bool:
    return str(row.get("correct", "")).strip().lower() == "true"


def label_key(v) -> str:
    v = str(v or "").strip()
    if v.startswith("WIN") or v == "1":   return "WIN"
    if "LOSE" in v or "LOSS" in v or v == "-1": return "LOSE"
    return v


def has_node(row: dict, col: str) -> bool:
    return bool(str(row.get(col, "")).strip())


def split_entries(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value: return []
    matches = re.findall(r"(.+?\[[^\]]+\])(?:;\s+|$)", value)
    return [m.strip() for m in matches] if matches else [p.strip() for p in value.split(";") if p.strip()]


def node_name(entry: str) -> str:
    return re.sub(r"\s+\[[^\]]+\]\s*$", "", entry).strip()


def esc(t) -> str:
    return html.escape(str(t), quote=True)


# ── SVG primitives ───────────────────────────────────────────────────────────

def svg_wrap(w: int, h: int, body: str, title: str = "") -> str:
    desc = f"<title>{esc(title)}</title>\n" if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f"{desc}"
        '<rect width="100%" height="100%" fill="#fafafa"/>\n'
        "<style>"
        "text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1f2933}"
        ".fig-title{font-size:17px;font-weight:700}"
        ".subtitle{font-size:11px;fill:#52616b}"
        ".axis-label{font-size:12px;fill:#52616b}"
        ".tick{font-size:11px;fill:#6b7c93}"
        ".bar-label{font-size:12px;fill:#1f2933}"
        ".note{font-size:10px;fill:#8a9bb0}"
        ".legend{font-size:11px;fill:#1f2933}"
        ".annot{font-size:13px;font-weight:600}"
        ".section{font-size:12px;font-weight:600;fill:#2A6F97}"
        "</style>\n"
        f"{body}\n</svg>\n"
    )


def hline(x1, y1, x2, y2, color="#d8dee4", sw=1, dash="") -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{da}/>'


def rect_el(x, y, w, h, fill, rx=3, opacity=1.0) -> str:
    op = f' opacity="{opacity}"' if opacity < 1 else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0):.2f}" height="{max(h,0):.2f}" rx="{rx}" fill="{fill}"{op}/>'


def txt(x, y, text, cls="", anchor="start") -> str:
    cls_a = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{cls_a}>{esc(text)}</text>'


def rotated_text(x, y, text, cls="axis-label") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" class="{cls}" '
            f'transform="rotate(-90,{x:.1f},{y:.1f})">{esc(text)}</text>')


# ── Figure 1: Calibration Curve ───────────────────────────────────────────────

def fig_calibration(rows: list[dict], path: Path) -> list[dict]:
    bins = [(0.5 + i * 0.025, 0.5 + (i + 1) * 0.025) for i in range(20)]
    binned = []
    for lo, hi in bins:
        mid = round((lo + hi) / 2, 4)
        items = [r for r in rows if lo <= fnum(r["confidence"]) < hi]
        n = len(items)
        acc = sum(is_correct(r) for r in items) / n if n else math.nan
        binned.append({"confidence_mid": mid, "accuracy": round(acc, 4) if n else None, "n_cases": n})

    W, H = 600, 460
    L, R, T, B = 70, 40, 60, 72
    cw, ch = W - L - R, H - T - B

    def cx(c): return L + (c - 0.5) / 0.5 * cw
    def cy(a): return T + ch - a * ch

    body = [txt(W / 2, 32, "Model Calibration: Confidence vs Actual Accuracy", "fig-title", "middle"),
            txt(W / 2, 52, "Bubble size ∝ number of cases in bin", "subtitle", "middle")]

    for tick in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = cy(tick)
        body.append(hline(L, y, W - R, y, "#e5e9f0"))
        body.append(txt(L - 7, y + 4, f"{int(tick*100)}%", "tick", "end"))
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = cx(tick)
        body.append(hline(x, T, x, T + ch, "#e5e9f0"))
        body.append(txt(x, T + ch + 16, f"{tick:.1f}", "tick", "middle"))

    body.append(hline(L, T + ch, W - R, T + ch, "#9aabbd", 1.5))
    body.append(hline(L, T, L, T + ch, "#9aabbd", 1.5))

    # Diagonal
    body.append(f'<line x1="{cx(0.5):.1f}" y1="{cy(0.5):.1f}" x2="{cx(1.0):.1f}" y2="{cy(1.0):.1f}" '
                f'stroke="#aab4c4" stroke-width="1.5" stroke-dasharray="6 3"/>')
    body.append(txt(cx(0.73), cy(0.71), "perfect calibration", "note"))

    valid = [(b["confidence_mid"], b["accuracy"], b["n_cases"])
             for b in binned if b["accuracy"] is not None]
    max_n = max(n for _, _, n in valid)

    pts = " ".join(f"{cx(c):.1f},{cy(a):.1f}" for c, a, _ in valid)
    body.append(f'<polyline points="{pts}" fill="none" stroke="{C_BLUE}" stroke-width="2" opacity="0.55"/>')

    for conf, acc, n in valid:
        r_b = 5 + 16 * math.sqrt(n / max_n)
        x, y = cx(conf), cy(acc)
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_b:.1f}" '
                    f'fill="{C_BLUE}" opacity="0.68" stroke="#fff" stroke-width="1"/>')
        if n >= 400:
            body.append(txt(x, y - r_b - 5, f"{acc*100:.0f}%", "tick", "middle"))

    body.append(txt(W / 2, H - 18, "Predicted Confidence", "axis-label", "middle"))
    body.append(rotated_text(15, T + ch / 2, "Actual Accuracy (%)"))

    path.write_text(svg_wrap(W, H, "\n".join(body), "Calibration Curve"), encoding="utf-8")
    return binned


# ── Figure 2: Accuracy by Bucket (clean) ─────────────────────────────────────

def fig_accuracy_by_bucket(rows: list[dict], path: Path) -> list[dict]:
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    data = []
    for bucket, items in sorted(by_bucket.items()):
        n = len(items)
        acc = sum(is_correct(r) for r in items) / n * 100
        hcw = sum((not is_correct(r)) and fnum(r["confidence"]) >= 0.8 for r in items) / n * 100
        data.append({"bucket": BUCKET_DISPLAY.get(bucket, bucket), "n": n,
                     "accuracy_pct": round(acc, 1), "high_conf_wrong_pct": round(hcw, 1)})

    overall = sum(is_correct(r) for r in rows) / len(rows) * 100
    W, H = 620, 360
    L, R, T, B = 180, 120, 62, 56
    cw, ch = W - L - R, H - T - B
    n_bars = len(data)
    row_h = ch / n_bars
    bar_h = row_h * 0.42

    body = [txt(W / 2, 30, "Test-Set Accuracy by Legal Domain", "fig-title", "middle"),
            txt(W / 2, 50, f"Overall accuracy = {overall:.1f}%  |  n = {len(rows):,} cases", "subtitle", "middle")]

    for tick in [60, 70, 80, 90, 100]:
        x = L + (tick - 60) / 40 * cw
        body.append(hline(x, T - 4, x, T + ch, "#e5e9f0"))
        body.append(txt(x, T + ch + 14, f"{tick}%", "tick", "middle"))

    overall_x = L + (overall - 60) / 40 * cw
    body.append(hline(overall_x, T - 4, overall_x, T + ch, "#aab4c4", 1.5, "5 3"))
    body.append(txt(overall_x + 4, T - 10, "overall avg", "note"))

    body.append(hline(L, T, L, T + ch, "#9aabbd", 1.5))
    body.append(hline(L, T + ch, W - R, T + ch, "#9aabbd", 1.5))

    for i, d in enumerate(data):
        yc = T + i * row_h + row_h / 2
        body.append(txt(L - 10, yc + 5, d["bucket"], "bar-label", "end"))
        # accuracy bar
        bw = (d["accuracy_pct"] - 60) / 40 * cw
        body.append(rect_el(L, yc - bar_h / 2, bw, bar_h, C_BLUE))
        body.append(txt(L + bw + 8, yc + 5, f"{d['accuracy_pct']:.1f}%", "annot"))
        # high-conf-wrong annotation on right
        body.append(txt(W - R + 8, yc - 5, f"n={d['n']:,}", "note"))
        body.append(txt(W - R + 8, yc + 9, f"HCW={d['high_conf_wrong_pct']:.1f}%", "note"))

    body.append(txt(W - R + 8, T - 4, "HCW = high-", "note"))
    body.append(txt(W - R + 8, T + 9, "confidence", "note"))
    body.append(txt(W - R + 8, T + 22, "wrong rate", "note"))
    body.append(txt(W / 2, H - 14, "Accuracy (x-axis starts at 60% for legibility)", "note", "middle"))

    path.write_text(svg_wrap(W, H, "\n".join(body), "Accuracy by Bucket"), encoding="utf-8")
    return data


# ── Figure 3: WIN-Rate Bias by Bucket (separate, clean) ──────────────────────

def fig_win_rate_bias(rows: list[dict], path: Path) -> list[dict]:
    """
    For each bucket: predicted WIN% vs actual WIN% as a side-by-side comparison,
    with gap/bias clearly annotated. Completely separate from the accuracy chart.
    """
    by_bucket = defaultdict(list)
    for r in rows:
        by_bucket[r["bucket"]].append(r)

    data = []
    for bucket, items in sorted(by_bucket.items()):
        n = len(items)
        pred_win = sum(label_key(r["prediction"]) == "WIN" for r in items) / n * 100
        actual_win = sum(label_key(r["actual"]) == "WIN" for r in items) / n * 100
        data.append({"bucket": BUCKET_DISPLAY.get(bucket, bucket), "n": n,
                     "pred_win_pct": round(pred_win, 1),
                     "actual_win_pct": round(actual_win, 1),
                     "bias_pp": round(pred_win - actual_win, 1)})

    # Fixed geometry — bars never overlap regardless of item count
    BAR_H   = 22   # height of each individual bar
    BAR_GAP = 8    # gap between the two bars in one group
    GRP_GAP = 32   # gap between consecutive groups
    ROW_H   = BAR_H * 2 + BAR_GAP + GRP_GAP   # = 84 px per group

    L, R, T, B = 185, 140, 82, 52
    chart_h = ROW_H * len(data)
    W, H = 700, T + chart_h + B
    cw = W - L - R

    body = [txt(W / 2, 30, "Predicted vs Actual WIN Rate by Legal Domain", "fig-title", "middle"),
            txt(W / 2, 50, "Negative bias = model under-predicts wins", "subtitle", "middle")]

    body.append(rect_el(L, 58, 14, 11, C_BLUE))
    body.append(txt(L + 20, 68, "Predicted WIN%", "legend"))
    body.append(rect_el(L + 175, 58, 14, 11, C_TEAL))
    body.append(txt(L + 195, 68, "Actual WIN%", "legend"))

    # Grid + x-axis ticks
    for tick in [0, 20, 40, 60, 80, 100]:
        x = L + tick / 100 * cw
        body.append(hline(x, T - 6, x, T + chart_h, "#e5e9f0"))
        body.append(txt(x, T + chart_h + 16, f"{tick}%", "tick", "middle"))

    body.append(hline(L, T - 6, L, T + chart_h, "#9aabbd", 1.5))
    body.append(hline(L, T + chart_h, W - R, T + chart_h, "#9aabbd", 1.5))

    for i, d in enumerate(data):
        grp_top = T + i * ROW_H
        label_y = grp_top + BAR_H + BAR_GAP / 2 + 5   # vertically centred between bars

        body.append(txt(L - 10, label_y, d["bucket"], "bar-label", "end"))

        # Predicted bar
        pw = d["pred_win_pct"] / 100 * cw
        body.append(rect_el(L, grp_top, pw, BAR_H, C_BLUE, rx=3))
        body.append(txt(L + pw + 6, grp_top + BAR_H - 5, f"{d['pred_win_pct']:.0f}%", "tick"))

        # Actual bar
        aw = d["actual_win_pct"] / 100 * cw
        body.append(rect_el(L, grp_top + BAR_H + BAR_GAP, aw, BAR_H, C_TEAL, rx=3))
        body.append(txt(L + aw + 6, grp_top + BAR_H * 2 + BAR_GAP - 5, f"{d['actual_win_pct']:.0f}%", "tick"))

        # Bias + n annotation
        bias = d["bias_pp"]
        b_cls = "tick" if abs(bias) >= 3 else "note"
        body.append(txt(W - R + 10, grp_top + BAR_H - 3,   f"bias: {bias:+.1f} pp", b_cls))
        body.append(txt(W - R + 10, grp_top + BAR_H + BAR_GAP + BAR_H - 3, f"n={d['n']:,}", "note"))

    body.append(txt(W / 2, H - 14,
                    "All buckets show negative bias; Land & Property is the most under-predicted (−12.2 pp)",
                    "note", "middle"))
    path.write_text(svg_wrap(W, H, "\n".join(body), "WIN Rate Bias"), encoding="utf-8")
    return data


# ── Figure 4: Jurisdiction Accuracy ──────────────────────────────────────────

def fig_jurisdiction(rows: list[dict], path: Path, min_n: int = 100) -> list[dict]:
    court_stats: dict[str, list] = defaultdict(lambda: [0, 0])
    for r in rows:
        field = str(r.get("courts_significant", "")).strip()
        if not field: continue
        entries = re.findall(r"(.+?\[[^\]]+\])(?:;\s+|$)", field)
        if not entries: continue
        court = re.sub(r"\s+\[[^\]]+\]\s*$", "", entries[0]).strip()
        court_stats[court][1] += 1
        if is_correct(r): court_stats[court][0] += 1

    overall_acc = sum(is_correct(r) for r in rows) / len(rows) * 100
    qualified = [{"court": c, "n": t, "accuracy_pct": round(cc / t * 100, 1),
                  "delta": round(cc / t * 100 - overall_acc, 1)}
                 for c, (cc, t) in court_stats.items() if t >= min_n]
    qualified.sort(key=lambda x: -x["accuracy_pct"])
    top8 = qualified[:8]
    bot8 = list(reversed(qualified[-8:]))
    all_display = top8 + [None] + bot8
    n_real = len(top8) + len(bot8)

    W, H = 840, 90 + 28 * (n_real + 2)
    L, R, T = 300, 50, 72
    chart_w = W - L - R
    overall_x = L + (overall_acc - 60) / 40 * chart_w

    body = [txt(W / 2, 30, "Model Accuracy by Jurisdiction", "fig-title", "middle"),
            txt(W / 2, 52, f"Courts with ≥{min_n} test cases  |  Overall = {overall_acc:.1f}%  |  Gap = {top8[0]['accuracy_pct']-bot8[-1]['accuracy_pct']:.1f} pp",
                "subtitle", "middle")]

    for tick in [60, 65, 70, 75, 80, 85, 90, 95, 100]:
        x = L + (tick - 60) / 40 * chart_w
        body.append(hline(x, T - 6, x, T + 28 * (n_real + 1), "#e5e9f0"))
        body.append(txt(x, T - 10, f"{tick}%", "tick", "middle"))

    body.append(hline(overall_x, T - 6, overall_x, T + 28 * (n_real + 1), "#aab4c4", 1.5, "5 3"))

    real_i = 0
    for entry in all_display:
        if entry is None:
            y_sep = T + real_i * 28 + 14
            body.append(hline(L - 8, y_sep, W - R, y_sep, "#dde3ec", 1, "4 4"))
            body.append(txt(L - 12, y_sep + 4, "▲ best  ··  worst ▼", "note", "end"))
            real_i += 1; continue
        y = T + real_i * 28
        bw = max(0, (entry["accuracy_pct"] - 60) / 40 * chart_w)
        color = C_GREEN if entry["delta"] >= 0 else C_RED
        label = (entry["court"][:50] + "…") if len(entry["court"]) > 50 else entry["court"]
        body.append(txt(L - 10, y + 18, label, "bar-label", "end"))
        body.append(rect_el(L, y + 4, bw, 18, color, rx=3))
        x_end = L + bw
        body.append(txt(x_end + 6, y + 17, f"{entry['accuracy_pct']}%  (n={entry['n']})", "tick"))
        real_i += 1

    body.append(hline(L, T - 6, L, T + 28 * (n_real + 1), "#9aabbd", 1.5))
    body.append(hline(L, T + 28 * (n_real + 1), W - R, T + 28 * (n_real + 1), "#9aabbd", 1.5))
    path.write_text(svg_wrap(W, H, "\n".join(body), "Jurisdiction Accuracy"), encoding="utf-8")
    return qualified


# ── Figures 5 & 6: Node Coverage Split (Legal Evidence vs Structural) ─────────

def _node_coverage_panel(
    path: Path, title: str, subtitle: str, col_map: dict, rows: list[dict],
    pred_win: list[dict], pred_lose: list[dict]
) -> list[dict]:
    nw, nl = len(pred_win), len(pred_lose)
    data = []
    for col, label in col_map.items():
        w_pct = sum(has_node(r, col) for r in pred_win) / nw * 100
        l_pct = sum(has_node(r, col) for r in pred_lose) / nl * 100
        data.append({"node_type": label, "win_pct": round(w_pct, 1),
                     "lose_pct": round(l_pct, 1), "diff_pp": round(w_pct - l_pct, 1)})
    data.sort(key=lambda x: -x["diff_pp"])

    # Fixed geometry — guarantees no overlap
    BAR_H   = 22
    BAR_GAP = 8
    GRP_GAP = 30
    ROW_H   = BAR_H * 2 + BAR_GAP + GRP_GAP   # = 82 px

    L, R, T, B = 140, 110, 82, 48
    chart_h = ROW_H * len(data)
    W, H = 640, T + chart_h + B
    cw = W - L - R

    body = [txt(W / 2, 30, title, "fig-title", "middle"),
            txt(W / 2, 50, subtitle, "subtitle", "middle")]

    body.append(rect_el(L, 58, 14, 11, C_BLUE))
    body.append(txt(L + 18, 68, f"WIN prediction  (n={nw:,})", "legend"))
    body.append(rect_el(L + 230, 58, 14, 11, C_RED))
    body.append(txt(L + 248, 68, f"LOSE prediction  (n={nl:,})", "legend"))

    for tick in [0, 20, 40, 60, 80, 100]:
        x = L + tick / 100 * cw
        body.append(hline(x, T - 6, x, T + chart_h, "#e5e9f0"))
        body.append(txt(x, T + chart_h + 16, f"{tick}%", "tick", "middle"))
    body.append(hline(L, T - 6, L, T + chart_h, "#9aabbd", 1.5))
    body.append(hline(L, T + chart_h, W - R, T + chart_h, "#9aabbd", 1.5))

    for i, d in enumerate(data):
        grp_top = T + i * ROW_H
        label_y = grp_top + BAR_H + BAR_GAP / 2 + 5

        body.append(txt(L - 10, label_y, d["node_type"], "bar-label", "end"))

        # WIN bar
        pw = d["win_pct"] / 100 * cw
        body.append(rect_el(L, grp_top, pw, BAR_H, C_BLUE, rx=3))
        body.append(txt(L + pw + 5, grp_top + BAR_H - 5, f"{d['win_pct']:.0f}%", "tick"))

        # LOSE bar
        lw = d["lose_pct"] / 100 * cw
        body.append(rect_el(L, grp_top + BAR_H + BAR_GAP, lw, BAR_H, C_RED, rx=3))
        body.append(txt(L + lw + 5, grp_top + BAR_H * 2 + BAR_GAP - 5, f"{d['lose_pct']:.0f}%", "tick"))

        dp = d["diff_pp"]
        body.append(txt(W - R + 8, label_y, f"Δ {dp:+.1f} pp", "note" if abs(dp) < 3 else "tick"))

    body.append(txt(W - R + 8, T - 10, "Δ WIN−LOSE", "note"))
    path.write_text(svg_wrap(W, H, "\n".join(body), title), encoding="utf-8")
    return data


def fig_node_coverage_split(rows: list[dict], path_legal: Path, path_structural: Path) -> tuple:
    pred_win = [r for r in rows if label_key(r["prediction"]) == "WIN"]
    pred_lose = [r for r in rows if label_key(r["prediction"]) == "LOSE"]

    legal_data = _node_coverage_panel(
        path_legal,
        "Legal-Evidence Node Coverage: WIN vs LOSE Predictions",
        "% of cases with ≥1 significant LEGAL node cited (statutes, provisions, precedents, similar cases)",
        LEGAL_LABELS, rows, pred_win, pred_lose)

    struct_data = _node_coverage_panel(
        path_structural,
        "Structural Node Coverage: WIN vs LOSE Predictions",
        "% of cases with ≥1 significant STRUCTURAL node cited (judges, courts, lawyers, parties)",
        STRUCTURAL_LABELS, rows, pred_win, pred_lose)

    return legal_data, struct_data


# ── Figure 7: Court & Judge Citation Bias ────────────────────────────────────

def fig_citation_bias(rows: list[dict], path: Path,
                      min_n: int = 40, max_n: int = 5_000,
                      top_k: int = 12) -> dict:
    """
    For each court / judge cited as a significant node: WIN rate among those citations
    vs the overall baseline. Answers which nodes the model associates with winning
    when it finds them influential in a prediction.
    Case-number artefacts (names containing 4+ consecutive digits) are excluded.
    """
    total_win = sum(1 for r in rows if label_key(r.get("prediction", "")) == "WIN")
    baseline  = total_win / len(rows) * 100   # ~56.6 %

    def build(col: str) -> list[dict]:
        wc: Counter = Counter()
        lc: Counter = Counter()
        for r in rows:
            pred = label_key(r.get("prediction", ""))
            for e in split_entries(r.get(col, "")):
                n = node_name(e)
                if not n: continue
                if re.search(r"\d{4,}", n): continue   # skip case-number artefacts
                if pred == "WIN":  wc[n] += 1
                elif pred == "LOSE": lc[n] += 1
        all_k = {k for k in set(wc) | set(lc)
                 if min_n <= wc.get(k, 0) + lc.get(k, 0) <= max_n}
        top = sorted(all_k, key=lambda k: -(wc.get(k, 0) + lc.get(k, 0)))[:top_k]
        result = []
        for nd in top:
            nc = wc.get(nd, 0) + lc.get(nd, 0)
            wr = wc.get(nd, 0) / nc * 100
            result.append({"node": nd, "n_cited": nc,
                           "win_rate_pct": round(wr, 1),
                           "delta_pp":     round(wr - baseline, 1)})
        result.sort(key=lambda x: -x["win_rate_pct"])
        return result

    courts  = build("courts_significant")
    judges  = build("judges_significant")

    # ── geometry ──────────────────────────────────────────────────────────────
    BAR_H   = 16
    GRP_GAP = 10
    ROW_H   = BAR_H + GRP_GAP   # 26 px

    SEC_HDR = 28
    SEC_GAP = 22

    L, R, T, B = 245, 85, 68, 46
    chart_w = 360
    W = L + chart_w + R

    nk_c = len(courts)
    nk_j = len(judges)
    sect_h_c = SEC_HDR + nk_c * ROW_H
    sect_h_j = SEC_HDR + nk_j * ROW_H
    H = T + sect_h_c + SEC_GAP + sect_h_j + B

    bx = L + baseline / 100 * chart_w   # baseline x-position

    body = [
        txt(W / 2, 27, "Court & Judge Citation Bias: WIN Rate When Cited as Significant",
            "fig-title", "middle"),
        txt(W / 2, 46, f"Baseline overall WIN rate = {baseline:.1f}%  |  "
            f"n = cases where this node was flagged as significant by the GNN explainer",
            "subtitle", "middle"),
    ]

    def draw_section(data: list[dict], label: str, y_off: int, sect_h: int) -> None:
        body.append(txt(L, y_off + 20, label, "section"))
        body.append(hline(L, y_off + 24, L + chart_w, y_off + 24, "#d0d8e4", 1))

        for tick in [0, 20, 40, 60, 80, 100]:
            x = L + tick / 100 * chart_w
            body.append(hline(x, y_off + SEC_HDR, x, y_off + sect_h, "#e5e9f0"))
            body.append(txt(x, y_off + sect_h + 14, f"{tick}%", "tick", "middle"))

        # baseline reference line
        body.append(hline(bx, y_off + SEC_HDR, bx, y_off + sect_h, "#9aabbd", 1.5, "4 3"))
        body.append(txt(bx, y_off + SEC_HDR - 5, f"base {baseline:.0f}%", "note", "middle"))

        for i, d in enumerate(data):
            gy  = y_off + SEC_HDR + i * ROW_H
            lbl = d["node"].title()
            if len(lbl) > 38: lbl = lbl[:36] + "…"
            body.append(txt(L - 8, gy + BAR_H - 3, lbl, "bar-label", "end"))

            bw = d["win_rate_pct"] / 100 * chart_w
            color = C_GREEN if d["delta_pp"] >= 0 else C_RED
            body.append(rect_el(L, gy, bw, BAR_H, color, rx=3, opacity=0.85))
            body.append(txt(L + bw + 5, gy + BAR_H - 3,
                            f"{d['win_rate_pct']:.0f}%  (n={d['n_cited']})", "tick"))

        body.append(hline(L, y_off + SEC_HDR, L, y_off + sect_h, "#9aabbd", 1.5))
        body.append(hline(L, y_off + sect_h, L + chart_w, y_off + sect_h, "#9aabbd", 1.5))

    draw_section(courts, "Courts",  T,               sect_h_c)
    draw_section(judges, "Judges",  T + sect_h_c + SEC_GAP, sect_h_j)

    body.append(txt(W / 2, H - 14,
                    "Green = cited more in WIN predictions than baseline  |  "
                    "Red = cited more in LOSE predictions than baseline",
                    "note", "middle"))

    path.write_text(svg_wrap(W, H, "\n".join(body), "Court Judge Citation Bias"), encoding="utf-8")
    return {"courts": courts, "judges": judges, "baseline_pct": round(baseline, 1)}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows(DEFAULT_INPUT)
    print(f"Loaded {len(rows):,} test-set rows")

    print("Fig 1: calibration curve …")
    cal = fig_calibration(rows, FIGS_DIR / "calibration_curve.svg")
    write_csv(DATA_DIR / "calibration_bins.csv",
              [b for b in cal if b["accuracy"] is not None],
              ["confidence_mid", "accuracy", "n_cases"])

    print("Fig 2: accuracy by bucket …")
    bkt = fig_accuracy_by_bucket(rows, FIGS_DIR / "accuracy_by_bucket.svg")
    write_csv(DATA_DIR / "accuracy_by_bucket.csv", bkt, list(bkt[0].keys()))

    print("Fig 3: WIN-rate bias …")
    bias = fig_win_rate_bias(rows, FIGS_DIR / "win_rate_bias_by_bucket.svg")
    write_csv(DATA_DIR / "win_rate_bias_by_bucket.csv", bias, list(bias[0].keys()))

    print("Fig 4: jurisdiction accuracy …")
    jur = fig_jurisdiction(rows, FIGS_DIR / "jurisdiction_accuracy.svg")
    write_csv(DATA_DIR / "jurisdiction_accuracy.csv", jur, list(jur[0].keys()))

    print("Figs 5 & 6: node coverage split …")
    leg, struct = fig_node_coverage_split(
        rows,
        FIGS_DIR / "node_evidence_coverage.svg",
        FIGS_DIR / "node_structural_coverage.svg")
    write_csv(DATA_DIR / "node_legal_coverage.csv", leg, list(leg[0].keys()))
    write_csv(DATA_DIR / "node_structural_coverage.csv", struct, list(struct[0].keys()))

    print("Fig 7: court & judge citation bias …")
    bias7 = fig_citation_bias(rows, FIGS_DIR / "citation_bias_courts_judges.svg")
    write_csv(DATA_DIR / "citation_bias_courts.csv",  bias7["courts"],
              ["node", "n_cited", "win_rate_pct", "delta_pp"])
    write_csv(DATA_DIR / "citation_bias_judges.csv",  bias7["judges"],
              ["node", "n_cited", "win_rate_pct", "delta_pp"])

    print(f"\nFigures → {FIGS_DIR}")
    print(f"Tables  → {DATA_DIR}")
    print("\nKey findings:")
    print(f"  Overall accuracy: {sum(is_correct(r) for r in rows)/len(rows)*100:.2f}%")
    for d in bias:
        print(f"  {d['bucket']:25s}  WIN bias={d['bias_pp']:+.1f}pp")


if __name__ == "__main__":
    main()
