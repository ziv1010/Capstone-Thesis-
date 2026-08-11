#!/usr/bin/env python3
"""Render all_results.json into a single self-contained HTML dashboard.

Run collect_all_results.py first. Output: outputs/dashboard.html
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DATA = _HERE / "all_results.json"
OUT = _HERE / "dashboard.html"

FAMILY = {
    "trivial": ("Trivial", "s1"),
    "classical": ("Classical non-LLM", "s1"),
    "gnn": ("GNN variant (new)", "s2"),
    "hgt": ("HGT — published", "s3"),
}


def fmt_p(value: float | None) -> str:
    if value is None:
        return "&mdash;"
    if value == 0.0:
        return "&lt;1e-300"
    if value < 1e-3:
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"{mantissa}&times;10<sup>{int(exponent)}</sup>"
    return f"{value:.3f}"


def verdict(record: dict, hgt_accuracy: float) -> tuple[str, str]:
    if record["family"] == "hgt":
        return "reference", "ref"
    sig = record.get("significance") or {}
    p = sig.get("p_value")
    if p is not None and p > 0.05:
        return "tied with HGT", "tie"
    return ("ahead of HGT", "ahead") if record["aggregate"]["accuracy_mean"] > hgt_accuracy else ("behind HGT", "behind")


SECTION_ORDER = ["preamble only", "facts only", "arguments only", "facts + arguments", "all three (fixed C)"]


def _section_block(data: dict, majority: dict, hgt: dict) -> str:
    """Panel decomposing the flat-text signal by rhetorical section."""
    sect = data.get("section_ablation")
    if not sect:
        return ""
    base = majority["aggregate"]["accuracy_mean"]
    rows = []
    for key in SECTION_ORDER:
        if key not in sect:
            continue
        entry = sect[key]
        tag = ' <span class="tag">cf. B3</span>' if key.startswith("all three") else ""
        rows.append(
            f"<tr><td>{key}{tag}</td>"
            f'<td class="num">{entry["accuracy_mean"] * 100:.2f}'
            f'<span class="pm">&plusmn;{entry["accuracy_std"] * 100:.2f}</span></td>'
            f'<td class="num">{entry["macro_f1_mean"]:.4f}</td>'
            f'<td class="num">{entry["n_features"]:,}</td>'
            f'<td class="num">{(entry["accuracy_mean"] - base) * 100:+.2f}</td>'
            f'<td class="num">{(entry["accuracy_mean"] - hgt["accuracy"]) * 100:+.2f}</td></tr>'
        )

    reading = ""
    pre = sect.get("preamble only", {}).get("accuracy_mean")
    nopre = sect.get("facts + arguments", {}).get("accuracy_mean")
    full = sect.get("all three (fixed C)", {}).get("accuracy_mean")
    if pre and nopre and full:
        prior_gain = (pre - base) * 100
        marginal = (full - nopre) * 100
        reading = (
            '<p class="note"><strong>Reading.</strong> The preamble alone &mdash; party names, court and '
            "petition type, i.e. registry-level base rates rather than legal reasoning &mdash; reaches "
            f"<strong>{pre * 100:.2f}%</strong>, {prior_gain:+.2f} points over the majority class. That looks "
            "like a shortcut worth worrying about, but it is almost entirely redundant with the narrative: "
            f"adding the preamble on top of facts and arguments is worth only <strong>{marginal:+.2f} points</strong>. "
            "<br><br>Facts and arguments on their own &mdash; every party name, court and petition type removed "
            f"&mdash; reach <strong>{nopre * 100:.2f}%</strong>, already "
            f'{(nopre - hgt["accuracy"]) * 100:+.2f} against the HGT. So the flat-text advantage is not an '
            "identity shortcut. It comes from the case narrative, which is exactly the evidence the graph "
            "model is meant to reason over &mdash; which points at the frozen text encoder, not the graph, "
            "as the bottleneck.</p>"
        )

    return f"""<section>
    <h2>Where does the flat-text signal come from?</h2>
    <p class="sub">Same folds, same sanitizer, same Linear SVM as B3 &mdash; only the sections fed to
    TF-IDF change. This separates case-type base rates (preamble) from the case narrative
    (facts, arguments).</p>
    <div class="card">
      <div class="scroll"><table>
        <thead><tr><th>Sections used</th><th class="num">Accuracy %</th><th class="num">Macro-F1</th>
          <th class="num">Features</th><th class="num">&Delta; maj.</th><th class="num">&Delta; HGT</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table></div>
      {reading}
    </div>
  </section>"""


def main() -> None:
    data = json.loads(DATA.read_text())
    models = data["models"]
    hgt = data["hgt_reference"]
    audit = data["leakage_audit"]
    san = audit["sanitization"]

    lo, hi = 0.50, 0.86  # accuracy axis domain, chosen to include the random floor
    scale = lambda v: (v - lo) / (hi - lo) * 100  # noqa: E731

    # ---- ranked bar rows
    bars = []
    for record in models:
        aggregate = record["aggregate"]
        accuracy, std = aggregate["accuracy_mean"], aggregate["accuracy_std"]
        _, slot = FAMILY[record["family"]]
        text, chip = verdict(record, hgt["accuracy"])
        sig = record.get("significance") or {}
        tip = (
            f"{record['label']} &middot; {aggregate['accuracy_mean'] * 100:.2f}% accuracy "
            f"&plusmn;{std * 100:.2f}<br>macro-F1 {aggregate['macro_f1_mean']:.4f}"
            f"<br>{record['note']}"
        )
        if sig.get("p_value") is not None:
            tip += f"<br>McNemar vs HGT: p = {fmt_p(sig['p_value'])} over {sig['n_paired']:,} paired cases"
        whisker = ""
        if std > 0:
            whisker = (
                f'<span class="whisk" style="left:{scale(accuracy - std):.3f}%;'
                f'width:{scale(accuracy + std) - scale(accuracy - std):.3f}%"></span>'
            )
        bars.append(
            f'''<div class="row" data-family="{record['family']}" tabindex="0">
      <div class="rlabel"><span class="dot {slot}"></span><span class="rname">{record['label']}</span></div>
      <div class="track">
        <span class="bar {slot}" style="width:{scale(accuracy):.3f}%"></span>{whisker}
        <span class="vlabel" style="left:{scale(accuracy):.3f}%">{accuracy * 100:.2f}</span>
        <div class="tip">{tip}</div>
      </div>
      <div class="rverdict"><span class="pill {chip}">{text}</span></div>
    </div>'''
        )

    # ---- full table
    rows = []
    for record in models:
        aggregate = record["aggregate"]
        sig = record.get("significance") or {}
        text, chip = verdict(record, hgt["accuracy"])
        delta = (aggregate["accuracy_mean"] - hgt["accuracy"]) * 100
        params = record.get("n_parameters")
        size = f"{params:,}" if params else (f"{record['n_features']:,} feat." if record.get("n_features") else "&mdash;")
        auc = aggregate.get("roc_auc_mean")
        rows.append(
            f"""<tr data-family="{record['family']}">
      <td><span class="dot {FAMILY[record['family']][1]}"></span>{record['label']}
          {'<span class="tag">published, not re-run</span>' if record['family'] == 'hgt' else ''}</td>
      <td class="mono muted">{record['id']}</td>
      <td class="num">{aggregate['accuracy_mean'] * 100:.2f}<span class="pm">&plusmn;{aggregate['accuracy_std'] * 100:.2f}</span></td>
      <td class="num">{aggregate['macro_f1_mean']:.4f}<span class="pm">&plusmn;{aggregate['macro_f1_std']:.4f}</span></td>
      <td class="num">{f'{auc:.4f}' if auc else '&mdash;'}</td>
      <td class="num">{size}</td>
      <td class="num {'pos' if delta > 0 else ('neg' if delta < 0 else '')}">{'&mdash;' if record['family'] == 'hgt' else f'{delta:+.2f}'}</td>
      <td class="num">{fmt_p(sig.get('p_value'))}</td>
      <td><span class="pill {chip}">{text}</span></td>
    </tr>"""
        )

    llm_rows = "".join(
        f"""<tr><td>{r['label']}</td><td class="num">{r['accuracy'] * 100:.2f}</td>
        <td class="num">{r['macro_f1']:.4f}</td>
        <td class="num">{f"{r['parse_rate'] * 100:.1f}%" if r['parse_rate'] else '&mdash;'}</td>
        <td class="num">{r['selective_accuracy'] * 100:.2f}</td>
        <td class="num">{r['selective_macro_f1']:.4f}</td>
        <td class="num neg">{(r['accuracy'] - hgt['accuracy']) * 100:+.2f}</td></tr>"""
        for r in data["llm_baselines"]
    )

    tfidf = [m for m in models if m.get("accuracy_before_sanitizer")]
    sanitizer_rows = "".join(
        f"""<tr><td>{m['label']}<span class="mono muted"> {m['id']}</span></td>
        <td class="num">{m['accuracy_before_sanitizer'] * 100:.2f}</td>
        <td class="num">{m['aggregate']['accuracy_mean'] * 100:.2f}</td>
        <td class="num neg">{(m['aggregate']['accuracy_mean'] - m['accuracy_before_sanitizer']) * 100:+.2f}</td></tr>"""
        for m in sorted(tfidf, key=lambda m: -m["aggregate"]["accuracy_mean"])
    )
    worst_drop = max(
        (m["accuracy_before_sanitizer"] - m["aggregate"]["accuracy_mean"]) * 100 for m in tfidf
    ) if tfidf else 0.0

    best = models[0]
    best_gap = (best["aggregate"]["accuracy_mean"] - hgt["accuracy"]) * 100
    majority = next(m for m in models if m["id"] == "B0_majority")
    over_trivial = (hgt["accuracy"] - majority["aggregate"]["accuracy_mean"]) * 100
    mlp = next(m for m in models if m["id"] == "arch_mlp_kfold")
    graph_gain = (hgt["accuracy"] - mlp["aggregate"]["accuracy_mean"]) * 100

    checks = [
        ("Source text is byte-identical to the HGT case node", audit["source_equals_hgt_case_node_content"]),
        ("Case order and labels match the HGT's predictions.csv", audit["case_order_and_labels_equal_hgt_predictions"]),
        ("No decision role (RPC / RATIO / RLC / ANALYSIS / ISSUE) retained", audit["unexpected_retained_role_count"] == 0),
        ("Test folds disjoint and covering all 71,813 cases", audit["test_fold_overlap"] == 0 and audit["test_fold_union"] == 71813),
        (f"All {san['source_mask_tokens']:,} [LEAKAGE_MASK] artifacts removed before TF-IDF", san["sanitized_mask_tokens"] == 0),
        (f"All {san['source_outcome_terms']:,} operative-outcome terms removed before TF-IDF", san["sanitized_outcome_terms"] == 0),
    ]
    check_html = "".join(
        f'<li class="{"ok" if passed else "bad"}"><span class="mk">{"✓" if passed else "✕"}</span>{label}</li>'
        for label, passed in checks
    )
    all_pass = all(p for _, p in checks)

    section_block = _section_block(data, majority, hgt)

    ticks = "".join(
        f'<span class="tick" style="left:{scale(v / 100):.3f}%"><i></i>{v:.0f}</span>'
        for v in (50, 55, 60, 65, 70, 75, 80, 85)
    )

    html = f"""<title>Reviewer 3 results — LegalGraph-LJP</title>
<style>
  :root {{
    --paper:#faf9f6; --card:#fffefb; --ink:#171614; --ink2:#55534c; --ink3:#8a877e;
    --rule:#e4e1d7; --hair:rgba(23,22,20,.09);
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
    --good:#0ca30c; --bad:#d03b3b; --warn:#fab219;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,"Book Antiqua",Georgia,serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  }}
  @media (prefers-color-scheme:dark) {{
    :root:where(:not([data-theme=light])) {{
      --paper:#141412; --card:#1c1c1a; --ink:#f4f2ec; --ink2:#b6b3a8; --ink3:#8a877e;
      --rule:#2e2e2b; --hair:rgba(244,242,236,.10);
      --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    }}
  }}
  :root[data-theme=dark] {{
    --paper:#141412; --card:#1c1c1a; --ink:#f4f2ec; --ink2:#b6b3a8; --ink3:#8a877e;
    --rule:#2e2e2b; --hair:rgba(244,242,236,.10);
    --s1:#3987e5; --s2:#d95926; --s3:#199e70;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans);
         font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:48px 24px 96px; display:flex; flex-direction:column; gap:44px; }}
  .eyebrow {{ font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--ink3); margin:0 0 10px; }}
  h1 {{ font-family:var(--serif); font-weight:600; font-size:clamp(28px,4.4vw,40px); line-height:1.14;
        margin:0 0 12px; text-wrap:balance; letter-spacing:-.01em; }}
  h2 {{ font-family:var(--serif); font-weight:600; font-size:22px; margin:0 0 4px; letter-spacing:-.005em; }}
  .sub {{ color:var(--ink2); max-width:66ch; margin:0; }}
  section {{ display:flex; flex-direction:column; gap:16px; }}
  .card {{ background:var(--card); border:1px solid var(--hair); border-radius:10px; padding:22px 24px; }}

  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; }}
  .tile {{ background:var(--card); border:1px solid var(--hair); border-radius:10px; padding:18px 20px;
           display:flex; flex-direction:column; gap:4px; }}
  .tile .k {{ font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:var(--ink3); }}
  .tile .v {{ font-size:30px; font-weight:600; letter-spacing:-.02em; line-height:1.1; }}
  .tile .n {{ font-size:12.5px; color:var(--ink2); }}
  .tile.flag {{ border-left:3px solid var(--warn); }}

  .chart {{ display:flex; flex-direction:column; gap:3px; position:relative; }}
  .row {{ display:grid; grid-template-columns:188px 1fr 118px; align-items:center; gap:12px;
          padding:3px 0; border-radius:6px; outline:none; }}
  .row:hover, .row:focus-visible {{ background:color-mix(in srgb, var(--ink) 4%, transparent); }}
  .row:focus-visible {{ box-shadow:0 0 0 2px var(--s1); }}
  .rlabel {{ display:flex; align-items:center; gap:7px; font-size:13px; min-width:0; }}
  .rname {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .dot {{ width:9px; height:9px; border-radius:2px; flex:0 0 auto; }}
  .dot.s1 {{ background:var(--s1); }} .dot.s2 {{ background:var(--s2); }} .dot.s3 {{ background:var(--s3); }}
  .track {{ position:relative; height:22px; }}
  .bar {{ position:absolute; left:0; top:5px; height:12px; border-radius:0 4px 4px 0; display:block; }}
  .bar.s1 {{ background:var(--s1); }} .bar.s2 {{ background:var(--s2); }} .bar.s3 {{ background:var(--s3); }}
  .whisk {{ position:absolute; top:10px; height:2px; background:var(--ink); opacity:.34; }}
  .vlabel {{ position:absolute; top:2px; margin-left:8px; font-size:12px; font-variant-numeric:tabular-nums;
             color:var(--ink2); white-space:nowrap; }}
  .rverdict {{ text-align:right; }}
  .refline {{ position:absolute; top:0; bottom:18px; width:0; border-left:1.5px dashed var(--s3); opacity:.85; }}
  .refcap {{ position:absolute; top:-19px; transform:translateX(-50%); font-size:11px; color:var(--s3);
             white-space:nowrap; font-weight:600; }}
  .axis {{ position:relative; height:20px; margin:8px 130px 0 200px; border-top:1px solid var(--rule); }}
  .tick {{ position:absolute; transform:translateX(-50%); font-size:11px; color:var(--ink3);
           font-variant-numeric:tabular-nums; padding-top:5px; }}
  .tick i {{ position:absolute; top:0; left:50%; width:1px; height:4px; background:var(--rule); }}
  .axname {{ margin:2px 130px 0 200px; font-size:11px; color:var(--ink3); }}

  .tip {{ position:absolute; left:0; bottom:26px; z-index:5; background:var(--ink); color:var(--paper);
          padding:9px 11px; border-radius:7px; font-size:12px; line-height:1.45; max-width:340px;
          opacity:0; visibility:hidden; transition:opacity .1s; pointer-events:none; }}
  .row:hover .tip, .row:focus-visible .tip {{ opacity:1; visibility:visible; }}

  .legend {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .lg {{ display:inline-flex; align-items:center; gap:6px; font-size:12.5px; padding:5px 11px;
         border:1px solid var(--rule); border-radius:999px; background:none; color:var(--ink);
         font-family:inherit; cursor:pointer; }}
  .lg[aria-pressed=false] {{ opacity:.42; }}
  .lg:focus-visible {{ outline:2px solid var(--s1); outline-offset:2px; }}

  .pill {{ font-size:11px; padding:2.5px 8px; border-radius:999px; white-space:nowrap; font-weight:500;
           border:1px solid transparent; }}
  .pill.ahead {{ color:var(--bad); border-color:color-mix(in srgb,var(--bad) 40%,transparent);
                 background:color-mix(in srgb,var(--bad) 9%,transparent); }}
  .pill.behind {{ color:var(--ink2); border-color:var(--rule); }}
  .pill.tie {{ color:var(--ink2); border-color:var(--rule); background:color-mix(in srgb,var(--ink) 5%,transparent); }}
  .pill.ref {{ color:var(--s3); border-color:color-mix(in srgb,var(--s3) 45%,transparent); font-weight:600; }}

  .scroll {{ overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; font-size:11px; letter-spacing:.07em; text-transform:uppercase; color:var(--ink3);
        font-weight:600; padding:0 10px 9px; border-bottom:1px solid var(--rule); white-space:nowrap; }}
  td {{ padding:9px 10px; border-bottom:1px solid var(--hair); vertical-align:middle; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  th.num {{ text-align:right; }}
  .pm {{ color:var(--ink3); font-size:11px; margin-left:4px; }}
  .mono {{ font-family:var(--mono); font-size:11.5px; }}
  .muted {{ color:var(--ink3); }}
  .pos {{ color:var(--bad); }} .neg {{ color:var(--ink2); }}
  .tag {{ font-size:10px; letter-spacing:.05em; text-transform:uppercase; color:var(--s3);
          border:1px solid color-mix(in srgb,var(--s3) 40%,transparent); border-radius:4px;
          padding:1px 5px; margin-left:7px; white-space:nowrap; }}

  .auditbar {{ display:flex; align-items:center; gap:10px; padding:12px 16px; border-radius:8px;
               background:color-mix(in srgb,var(--good) 10%,transparent);
               border:1px solid color-mix(in srgb,var(--good) 35%,transparent); font-weight:500; }}
  .auditbar .big {{ font-size:15px; color:var(--good); }}
  ul.checks {{ list-style:none; margin:0; padding:0; display:grid; gap:7px;
               grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }}
  ul.checks li {{ display:flex; gap:9px; align-items:flex-start; font-size:13px; color:var(--ink2); }}
  .mk {{ flex:0 0 auto; font-weight:700; }}
  li.ok .mk {{ color:var(--good); }} li.bad .mk {{ color:var(--bad); }}
  .note {{ font-size:13px; color:var(--ink2); max-width:72ch; }}
  .note strong {{ color:var(--ink); }}
  code {{ font-family:var(--mono); font-size:12px; background:color-mix(in srgb,var(--ink) 6%,transparent);
          padding:1px 5px; border-radius:4px; }}
  footer {{ font-size:12px; color:var(--ink3); border-top:1px solid var(--rule); padding-top:18px; }}
  @media (max-width:640px) {{
    .row {{ grid-template-columns:130px 1fr; }} .rverdict {{ display:none; }}
    .axis, .axname {{ margin:8px 0 0 142px; }}
  }}
  @media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">ASONAM 2026 &middot; Reviewers 3 and 6 &middot; R3-03, R3-04 &amp; R6-02</p>
    <h1>What happens to LegalGraph-LJP when you put it next to simpler models</h1>
    <p class="sub">Every model below was trained and scored on the <strong>same 71,813 cases and the same five
    folds</strong> as the published HGT run &mdash; splits read out of that run's own <code>predictions.csv</code>,
    never re-derived. The HGT row is the published result, read from disk; it was not re-run.</p>
  </header>

  <section>
    <div class="tiles">
      <div class="tile">
        <span class="k">Published HGT</span>
        <span class="v">{hgt['accuracy'] * 100:.2f}%</span>
        <span class="n">macro-F1 {hgt['macro_f1']:.4f} &middot; 5-fold mean</span>
      </div>
      <div class="tile">
        <span class="k">Gain over a trivial baseline</span>
        <span class="v">+{over_trivial:.1f}</span>
        <span class="n">accuracy points over majority class ({majority['aggregate']['accuracy_mean'] * 100:.2f}%)</span>
      </div>
      <div class="tile">
        <span class="k">Gain from the graph itself</span>
        <span class="v">+{graph_gain:.1f}</span>
        <span class="n">vs. the same model with every edge removed</span>
      </div>
      <div class="tile flag">
        <span class="k">Best simpler model</span>
        <span class="v">{best['aggregate']['accuracy_mean'] * 100:.2f}%</span>
        <span class="n">{best['label']} on sanitized TF-IDF &mdash; {best_gap:+.2f} pts vs. HGT</span>
      </div>
    </div>
  </section>

  <section>
    <h2>Accuracy, ranked</h2>
    <p class="sub">Bars are five-fold mean accuracy; the thin horizontal line through each bar spans
    &plusmn;1 population s.d. across folds. The dashed line marks the published HGT. Hover or focus a row for detail.</p>
    <div class="legend" role="group" aria-label="Filter by model family">
      <button class="lg" data-f="classical" aria-pressed="true"><span class="dot s1"></span>Classical non-LLM</button>
      <button class="lg" data-f="trivial" aria-pressed="true"><span class="dot s1"></span>Trivial</button>
      <button class="lg" data-f="gnn" aria-pressed="true"><span class="dot s2"></span>GNN variant (new)</button>
      <button class="lg" data-f="hgt" aria-pressed="true"><span class="dot s3"></span>HGT (published)</button>
    </div>
    <div class="card">
      <div class="chart" id="chart" style="padding-top:22px">
        <div class="refline" style="left:calc(200px + (100% - 330px) * {scale(hgt['accuracy']) / 100:.5f})">
          <span class="refcap">HGT {hgt['accuracy'] * 100:.2f}</span>
        </div>
        {''.join(bars)}
      </div>
      <div class="axis">{ticks}</div>
      <p class="axname">Accuracy (%) &mdash; axis starts at 50, not 0, to separate models that all beat chance</p>
    </div>
  </section>

  <section>
    <h2>Did anything leak?</h2>
    <p class="sub">This is the question that decides whether the rows above HGT mean anything. Two separate
    guarantees: the text is the same text the HGT reads, and no outcome vocabulary reaches the sparse models.</p>
    <div class="auditbar"><span class="big">{'✓' if all_pass else '✕'}</span>
      <span>{'All six input-provenance and leakage assertions pass over all 71,813 cases.' if all_pass else 'One or more assertions FAILED — do not use these numbers.'}</span></div>
    <div class="card"><ul class="checks">{check_html}</ul></div>
    <div class="card">
      <h2 style="font-size:16px">How much did the extra sanitizer actually change?</h2>
      <p class="note">The HGT reads the cleaned-case text as-is. The flat-text models get one
      <em>additional</em> guard on top: <strong>{san['source_mask_tokens']:,}</strong> <code>[LEAKAGE_MASK]</code>
      artifacts and <strong>{san['source_outcome_terms']:,}</strong> operative-outcome term occurrences
      ({san['characters_removed'] / 1e6:.2f}M characters) are stripped before vectorisation. If the TF-IDF
      advantage were leakage, removing all of it would collapse the number. It moves by at most
      <strong>{worst_drop:.2f} accuracy points</strong> &mdash; an order of magnitude smaller than the
      {best_gap:+.2f}-point gap to HGT.</p>
      <div class="scroll"><table>
        <thead><tr><th>Flat-text model</th><th class="num">Before sanitizer</th>
          <th class="num">After sanitizer (reported)</th><th class="num">Change</th></tr></thead>
        <tbody>{sanitizer_rows}</tbody>
      </table></div>
    </div>
  </section>

  {section_block}

  <section>
    <h2>Every model, every number</h2>
    <p class="sub">McNemar is the exact two-sided test on the {models[0].get('significance', {}).get('n_paired', 71813):,}
    paired held-out predictions you get by pooling each run's five disjoint test folds &mdash; every case is
    held out exactly once, so this is a per-case paired test, not a comparison of five fold means.</p>
    <div class="scroll"><table id="tbl">
      <thead><tr>
        <th>Model</th><th>Run ID</th><th class="num">Accuracy %</th><th class="num">Macro-F1</th>
        <th class="num">ROC-AUC</th><th class="num">Size</th><th class="num">&Delta; vs HGT</th>
        <th class="num">McNemar p</th><th>Verdict</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table></div>
  </section>

  <section>
    <h2>Generative LLM baselines, for context</h2>
    <p class="sub">Single fold (fold 0). Primary accuracy and macro-F1 use all 14,363 cases:
    unparseable outputs are incorrect abstentions. Parsed-only selective metrics are diagnostics.</p>
    <div class="card scroll"><table>
      <thead><tr><th>Model</th><th class="num">Full-denom. accuracy %</th><th class="num">Full-denom. macro-F1</th>
        <th class="num">Coverage</th><th class="num">Selective accuracy %</th>
        <th class="num">Selective macro-F1</th><th class="num">&Delta; vs HGT</th></tr></thead>
      <tbody>{llm_rows}</tbody>
    </table></div>
  </section>

  <section>
    <h2>How to read this</h2>
    <div class="card" style="display:flex;flex-direction:column;gap:12px">
      <p class="note"><strong>The graph is doing real work.</strong> Strip every edge and keep the same case
      features and the same trainer, and accuracy falls {graph_gain:.2f} points. That effect is the largest
      in the table and it is what the paper's representation claim rests on.</p>
      <p class="note"><strong>The heterogeneous attention is not what earns the number.</strong> A plain GAT
      on the type-collapsed graph &mdash; no node types, no per-relation weights, 36% fewer parameters &mdash;
      is statistically indistinguishable from HGT (p = 0.73). Relational GAT is slightly ahead of it. The
      accuracy-based argument for choosing HGT specifically does not survive; the explainability argument
      (typed attention is what makes the relation-level counterfactuals of &sect;4.2 expressible) does.</p>
      <p class="note"><strong>A sparse lexical model beats the graph on this corpus.</strong> TF-IDF with a
      linear SVM reaches {best['aggregate']['accuracy_mean'] * 100:.2f}%. The likely reason is capacity where
      it matters: 300,000 explicit n-gram features retain lexical detail that a 1024-dimensional mean-pooled
      BGE-M3 section embedding averages away. This is reported as found, not tuned into or out of existence.</p>
      <p class="note"><strong>Entity counts alone are weak.</strong> Logistic Regression on the GNN's own 12
      case scalars plus entity counts reaches only 60.20% &mdash; below the majority class. The structured
      metadata carries little signal without the text.</p>
    </div>
  </section>

  <footer>
    Sources: <code>R3_03_non_llm_baselines/outputs/baselines_summary.json</code>,
    <code>R3_04_gnn_architecture_ablation/outputs/models/*/kfold/kfold_summary.json</code>,
    <code>section_GNN/outputs/&hellip;/kfold_summary.json</code> (HGT, unmodified),
    <code>model comparison/outputs/*/metrics.json</code>.
    Metrics computed by <code>section_GNN/src/training/metrics.py::compute_metrics</code> &mdash; the same
    function that produced the published numbers. Regenerate with
    <code>collect_all_results.py &amp;&amp; build_dashboard.py</code>.
    Sanitized-corpus SHA-256 <span class="mono">{audit['sanitized_documents_sha256'][:16]}&hellip;</span>
  </footer>
</div>

<script>
  const buttons = [...document.querySelectorAll('.lg')];
  const off = new Set();
  function apply() {{
    for (const el of document.querySelectorAll('#chart .row, #tbl tbody tr')) {{
      el.style.display = off.has(el.dataset.family) ? 'none' : '';
    }}
  }}
  for (const b of buttons) {{
    b.addEventListener('click', () => {{
      const f = b.dataset.f;
      const on = b.getAttribute('aria-pressed') === 'true';
      if (on) {{ off.add(f); }} else {{ off.delete(f); }}
      b.setAttribute('aria-pressed', String(!on));
      apply();
    }});
  }}
</script>
"""
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(html) / 1024:.0f} KB, {len(models)} models)")


if __name__ == "__main__":
    main()
