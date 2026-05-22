#!/usr/bin/env python3
"""
Deep early-detection analysis for multi-hearing cases.

Produces 1 focused, publication-ready SVG figure:
  1. flip_rate_accuracy_by_bucket.svg – per-bucket: flip rate + H1 vs final accuracy

Run from this directory:
    python3 deep_analysis.py
"""
from __future__ import annotations

import csv
import html
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = BASE_DIR / "multi_hearing_case_paths.csv"
DEFAULT_HEARINGS = BASE_DIR / "multi_hearing_hearing_level_analysis.csv"
FIGS_DIR = BASE_DIR / "figures_v2"
DATA_DIR = BASE_DIR / "analysis_outputs_v2"

C_BLUE = "#2A6F97"
C_RED = "#C53B3B"
C_ORANGE = "#D9841A"
C_GREEN = "#2F7D57"
C_PURPLE = "#6D597A"
C_GREY = "#8A9BB0"
C_TEAL = "#4D908E"


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


def pct(v: float) -> float:
    return round(v * 100, 2) if not math.isnan(v) else math.nan


def mean(vals: list[float]) -> float:
    valid = [v for v in vals if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else math.nan


def is_true(v) -> bool:
    return str(v or "").strip().lower() == "true"


def pred_key(v) -> str:
    v = str(v or "").strip()
    if v.startswith("WIN") or v == "1":
        return "WIN"
    if "LOSE" in v or "LOSS" in v or v == "-1":
        return "LOSE"
    return v


def split_entries(value: str) -> list[str]:
    value = str(value or "").strip()
    if not value: return []
    matches = re.findall(r"(.+?\[[^\]]+\])(?:;\s+|$)", value)
    return [m.strip() for m in matches] if matches else [p.strip() for p in value.split(";") if p.strip()]


def node_name(entry: str) -> str:
    return re.sub(r"\s+\[[^\]]+\]\s*$", "", entry).strip()


def esc(t) -> str:
    return html.escape(str(t), quote=True)


def svg_wrap(w: int, h: int, body: str, title: str = "") -> str:
    desc = f'<title>{esc(title)}</title>\n' if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        f'{desc}'
        '<rect width="100%" height="100%" fill="#fafafa"/>\n'
        '<style>'
        'text{font-family:Inter,Arial,Helvetica,sans-serif;fill:#1f2933}'
        '.fig-title{font-size:17px;font-weight:700;fill:#1f2933}'
        '.axis-label{font-size:12px;fill:#52616b}'
        '.tick{font-size:11px;fill:#6b7c93}'
        '.bar-label{font-size:12px;fill:#1f2933}'
        '.note{font-size:10px;fill:#8a9bb0}'
        '.legend{font-size:11px;fill:#1f2933}'
        '.annot{font-size:13px;font-weight:600;fill:#1f2933}'
        '</style>\n'
        f'{body}\n</svg>\n'
    )


def hline(x1, y1, x2, y2, color="#d8dee4", sw=1, dash="") -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{da}/>'


def rect_elem(x, y, w, h, fill, rx=3, opacity=1.0) -> str:
    op = f' opacity="{opacity}"' if opacity < 1 else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}"{op}/>'


def text_elem(x, y, text, cls="", anchor="start", dy="0") -> str:
    cls_attr = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dy="{dy}"{cls_attr}>{esc(text)}</text>'


def classify_case(row: dict) -> str:
    """
    Returns one of 4 outcome categories per case.
    """
    n = int(row["n_hearings"])
    h1 = pred_key(row.get("hearing1_prediction", ""))
    hn = pred_key(row.get(f"hearing{n}_prediction", ""))
    final_correct = is_true(row.get("final_prediction_correct"))
    changed = is_true(row.get("changed_prediction"))

    if not changed:
        return "stable_correct" if final_correct else "stable_wrong"
    return f"{h1}→{hn}"


# ── Figure 1: Per-Bucket Flip Rate + Accuracy ────────────────────────────────

def fig_bucket_flip_accuracy(case_rows: list[dict], path: Path) -> list[dict]:
    """
    For each bucket: flip rate, H1 accuracy, final accuracy as grouped bars.
    Shows whether multi-hearing improves predictions and which buckets are most volatile.
    """
    by_bucket: dict[str, list] = defaultdict(list)
    for r in case_rows:
        by_bucket[r["bucket_raw"]].append(r)

    bucket_data = []
    for bucket, rows in sorted(by_bucket.items()):
        n = len(rows)
        flip_pct = sum(is_true(r["changed_prediction"]) for r in rows) / n * 100

        # H1 accuracy: prediction at hearing 1 vs final binary label
        h1_acc = []
        final_acc = []
        for r in rows:
            final_lbl = pred_key(r.get("case_actual_result", ""))
            h1_pred = pred_key(r.get("hearing1_prediction", ""))
            h1_acc.append(1 if h1_pred == final_lbl else 0)
            final_acc.append(1 if is_true(r.get("final_prediction_correct")) else 0)

        bucket_data.append({
            "bucket": bucket.replace("_", " ").title(),
            "n_cases": n,
            "flip_rate_pct": round(flip_pct, 1),
            "h1_accuracy_pct": round(mean([float(v) for v in h1_acc]) * 100, 1),
            "final_accuracy_pct": round(mean([float(v) for v in final_acc]) * 100, 1),
            "accuracy_gain_pp": round(
                mean([float(v) for v in final_acc]) * 100 - mean([float(v) for v in h1_acc]) * 100, 1),
        })

    # Fixed geometry — 3 bars per group, no overlap
    BAR_H   = 18
    BAR_GAP = 5    # gap between adjacent bars within a group
    GRP_GAP = 28   # gap between groups
    ROW_H   = BAR_H * 3 + BAR_GAP * 2 + GRP_GAP   # = 112 px

    L, R, T, B = 185, 90, 82, 52
    chart_h = ROW_H * len(bucket_data)
    W, H = 720, T + chart_h + B
    cw = W - L - R

    body = [text_elem(W / 2, 30, "Per-Bucket: Flip Rate and Prediction Accuracy", "fig-title", "middle")]

    legend_items = [("Flip rate", C_ORANGE), ("H1 Accuracy", C_BLUE), ("Final Accuracy", C_GREEN)]
    for j, (lbl, col) in enumerate(legend_items):
        body.append(rect_elem(L + j * 160, 54, 14, 11, col))
        body.append(text_elem(L + j * 160 + 18, 64, lbl, "legend"))

    for tick in [0, 20, 40, 60, 80, 100]:
        x = L + tick / 100 * cw
        body.append(hline(x, T - 6, x, T + chart_h, "#e5e9f0"))
        body.append(text_elem(x, T + chart_h + 16, f"{tick}%", "tick", "middle"))
    body.append(hline(L, T - 6, L, T + chart_h, "#9aabbd", 1.5))
    body.append(hline(L, T + chart_h, W - R, T + chart_h, "#9aabbd", 1.5))

    for i, bd in enumerate(bucket_data):
        grp_top = T + i * ROW_H
        label_y = grp_top + (BAR_H * 3 + BAR_GAP * 2) / 2 + 5

        body.append(text_elem(L - 10, label_y - 6, bd["bucket"], "bar-label", "end"))
        body.append(text_elem(L - 10, label_y + 10, f"n={bd['n_cases']}", "note", "end"))

        bar_defs = [
            (bd["flip_rate_pct"],    C_ORANGE),
            (bd["h1_accuracy_pct"],  C_BLUE),
            (bd["final_accuracy_pct"], C_GREEN),
        ]
        for j, (val, color) in enumerate(bar_defs):
            bar_top = grp_top + j * (BAR_H + BAR_GAP)
            bw = val / 100 * cw
            body.append(rect_elem(L, bar_top, bw, BAR_H, color, rx=2))
            body.append(text_elem(L + bw + 5, bar_top + BAR_H - 4, f"{val:.0f}%", "tick"))

        # Accuracy gain annotation on the right
        gain = bd["accuracy_gain_pp"]
        g_cls = "tick" if gain != 0 else "note"
        body.append(text_elem(W - R + 6, grp_top + BAR_H - 4, "H1 acc", "note"))
        body.append(text_elem(W - R + 6, grp_top + 2 * (BAR_H + BAR_GAP) + BAR_H - 4, "Final acc", "note"))
        body.append(text_elem(W - R + 6, grp_top + BAR_H * 3 + BAR_GAP * 2 + 6,
                              f"gain: {gain:+.1f} pp", g_cls))

    body.append(text_elem(W / 2, H - 18,
                          "Accuracy gain = Final Accuracy − H1 Accuracy   (positive = multi-hearing improves prediction)",
                          "note", "middle"))
    path.write_text(svg_wrap(W, H, "\n".join(body), "Bucket Flip and Accuracy"), encoding="utf-8")
    return bucket_data


# ── Figure 2: H1 Nodes — Flip vs Stable Cases ────────────────────────────────

def fig_h1_nodes_flip_stable(case_rows: list[dict], hearing_rows: list[dict],
                              path: Path, top_k: int = 10) -> list[dict]:
    """
    Top statute + precedent nodes cited at Hearing 1, broken down by whether
    the case eventually flipped its prediction (ORANGE) vs stayed stable (TEAL).
    Reveals which legal citations were present when the model got its first
    prediction wrong and later had to reverse.
    """
    flip_ids   = {r["base_case_id"] for r in case_rows if is_true(r.get("changed_prediction", ""))}
    stable_ids = {r["base_case_id"] for r in case_rows if not is_true(r.get("changed_prediction", ""))}

    h1_rows    = [r for r in hearing_rows if str(r.get("hearing_index", "")).strip() == "1"]
    h1_flip    = [r for r in h1_rows if r["base_case_id"] in flip_ids]
    h1_stable  = [r for r in h1_rows if r["base_case_id"] in stable_ids]
    nf, ns     = len(h1_flip), len(h1_stable)

    flip_cnt:   Counter = Counter()
    stable_cnt: Counter = Counter()

    for r in h1_flip:
        for col in ("statutes_significant", "precedents_significant"):
            for e in split_entries(r.get(col, "")):
                n = node_name(e)
                if n: flip_cnt[n] += 1

    for r in h1_stable:
        for col in ("statutes_significant", "precedents_significant"):
            for e in split_entries(r.get(col, "")):
                n = node_name(e)
                if n: stable_cnt[n] += 1

    all_keys = set(flip_cnt) | set(stable_cnt)
    overall  = {k: flip_cnt.get(k, 0) + stable_cnt.get(k, 0) for k in all_keys}
    top      = sorted(all_keys, key=lambda k: -overall[k])[:top_k]

    data = []
    for nd in top:
        fp = round(flip_cnt.get(nd, 0)   / nf * 100, 1) if nf else 0.0
        sp = round(stable_cnt.get(nd, 0) / ns * 100, 1) if ns else 0.0
        data.append({"node": nd, "flip_pct": fp, "stable_pct": sp,
                     "overall_n": overall[nd], "delta_pp": round(fp - sp, 1)})

    # ── geometry ──────────────────────────────────────────────────────────────
    BAR_H   = 14
    BAR_GAP = 3
    GRP_GAP = 12
    ROW_H   = BAR_H * 2 + BAR_GAP + GRP_GAP   # 43 px

    L, R, T, B = 230, 110, 82, 52
    chart_w = 340
    W       = L + chart_w + R
    chart_h = top_k * ROW_H
    H       = T + chart_h + B

    max_pct = 80
    def bw(v): return max(v / max_pct * chart_w, 0)

    body = [
        svg_wrap.__doc__,   # placeholder — replaced below
    ]
    body = []

    body.append(text_elem(W / 2, 28,
                          "Nodes Cited at Hearing 1: Cases That Flip vs Stay Stable",
                          "fig-title", "middle"))
    body.append(text_elem(W / 2, 47,
                          f"% of H1 hearings citing each statute/precedent  |  "
                          f"Flip n={nf}, Stable n={ns}",
                          "note", "middle"))

    # legend
    body += [
        rect_elem(L,       T - 20, 12, 10, C_ORANGE),
        text_elem(L + 16,  T - 11, f"Flipped prediction (n={nf})", "legend"),
        rect_elem(L + 195, T - 20, 12, 10, C_TEAL),
        text_elem(L + 211, T - 11, f"Stable prediction (n={ns})", "legend"),
    ]

    # grid + x-axis ticks
    for tick in range(0, max_pct + 1, 20):
        x = L + tick / max_pct * chart_w
        body.append(hline(x, T - 6, x, T + chart_h, "#e5e9f0"))
        body.append(text_elem(x, T + chart_h + 16, f"{tick}%", "tick", "middle"))

    for i, d in enumerate(data):
        gy  = T + i * ROW_H
        lbl = d["node"]
        if len(lbl) > 35: lbl = lbl[:33] + "…"
        mid_y = gy + BAR_H + BAR_GAP / 2 + 1
        body.append(text_elem(L - 8, mid_y + 6, lbl, "bar-label", "end"))

        # flip bar (orange)
        w1 = bw(d["flip_pct"])
        body.append(rect_elem(L, gy, w1, BAR_H, C_ORANGE, rx=2))
        if d["flip_pct"] >= 3:
            body.append(text_elem(L + w1 + 4, gy + BAR_H - 2,
                                  f"{d['flip_pct']:.0f}%", "tick"))

        # stable bar (teal)
        w2 = bw(d["stable_pct"])
        body.append(rect_elem(L, gy + BAR_H + BAR_GAP, w2, BAR_H, C_TEAL, rx=2))
        if d["stable_pct"] >= 3:
            body.append(text_elem(L + w2 + 4, gy + BAR_H * 2 + BAR_GAP - 2,
                                  f"{d['stable_pct']:.0f}%", "tick"))

        # delta annotation
        g = d["delta_pp"]
        g_col = C_ORANGE if g > 2 else (C_TEAL if g < -2 else C_GREY)
        body.append(f'<text x="{L + chart_w + 8:.1f}" y="{gy + BAR_H + 6:.1f}" '
                    f'class="tick" fill="{g_col}">{g:+.1f}pp</text>')

    body.append(hline(L, T - 6, L, T + chart_h, "#9aabbd", 1.5))
    body.append(hline(L, T + chart_h, L + chart_w, T + chart_h, "#9aabbd", 1.5))
    body.append(text_elem(W / 2, H - 16,
                          "Δ Flip − Stable  (positive = cited more at H1 in cases that later flipped)",
                          "note", "middle"))

    path.write_text(svg_wrap(W, H, "\n".join(body),
                             "H1 Nodes Flip vs Stable"), encoding="utf-8")
    return data


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    case_rows    = read_rows(DEFAULT_CASES)
    hearing_rows = read_rows(DEFAULT_HEARINGS)
    print(f"Loaded {len(case_rows)} multi-hearing cases, {len(hearing_rows)} hearing-level rows")

    print("Figure 1: bucket flip rate and accuracy...")
    ba = fig_bucket_flip_accuracy(case_rows, FIGS_DIR / "flip_rate_accuracy_by_bucket.svg")
    write_csv(DATA_DIR / "bucket_flip_accuracy.csv", ba, list(ba[0].keys()))

    print("Figure 2: H1 nodes — flip vs stable cases...")
    nd = fig_h1_nodes_flip_stable(case_rows, hearing_rows,
                                   FIGS_DIR / "h1_nodes_flip_vs_stable.svg")
    write_csv(DATA_DIR / "h1_nodes_flip_vs_stable.csv", nd,
              ["node", "flip_pct", "stable_pct", "delta_pp", "overall_n"])

    print(f"\nFigures → {FIGS_DIR}")
    print(f"Tables  → {DATA_DIR}")

    print("\n── Key Findings ──────────────────────────────────────────────")
    for r in ba:
        print(f"  {r['bucket']:25s} flip={r['flip_rate_pct']:.0f}%  H1_acc={r['h1_accuracy_pct']:.0f}%  "
              f"final_acc={r['final_accuracy_pct']:.0f}%  gain={r['accuracy_gain_pp']:+.1f}pp")


if __name__ == "__main__":
    main()
