#!/usr/bin/env python3
"""
Launch a single examiner-facing hub for the repo's populated visualisers.

Default behaviour starts the NER/RR inspector, final explanation browser,
multi-hearing/early-detection app, and the supplementary graph/entity apps.
Each app is included only when its expected artefacts are present.

Usage:
  python3 run_visualisers.py
  python3 run_visualisers.py --no-extras
  python3 run_visualisers.py --no-browser
"""
from __future__ import annotations

import argparse
import atexit
import html as html_lib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class Site:
    slug: str
    name: str
    url: str
    description: str
    kind: str
    focus: str
    metrics: tuple[str, ...] = ()
    featured: bool = False
    status: str = "started"


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen


PROCESSES: list[ManagedProcess] = []
TEMP_DIRS: list[tempfile.TemporaryDirectory] = []
USED_PORTS: set[int] = set()


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def choose_port(name: str, preferred: int) -> int:
    for port in range(preferred, preferred + 100):
        if port not in USED_PORTS and port_is_free(port):
            USED_PORTS.add(port)
            if port != preferred:
                print(f"[port] {name}: {preferred} is in use, using {port}")
            return port
    raise RuntimeError(f"{name}: no free port found in {preferred}-{preferred + 99}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the visualiser hub and related local servers.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface used by the hub and apps (default: 127.0.0.1; use 0.0.0.0 for direct remote access).",
    )
    parser.add_argument("--hub-port", type=int, default=8090)
    parser.add_argument("--timeline-port", type=int, default=8081)
    parser.add_argument("--gnn-static-port", type=int, default=8082)
    parser.add_argument("--explainer-port", type=int, default=8899)
    parser.add_argument("--embedding-port", type=int, default=8084)
    parser.add_argument("--graph-dash-port", type=int, default=8050)
    parser.add_argument("--stage-dash-port", type=int, default=8051)
    parser.add_argument("--entity-dash-port", type=int, default=8052)
    parser.add_argument("--pipeline-stage-port", type=int, default=8053)
    parser.add_argument("--static-only", action="store_true", help="Do not start Dash apps.")
    parser.add_argument("--no-extras", action="store_true", help="Start only the three examiner-focused apps.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the hub in a browser.")
    return parser.parse_args()


def start_process(name: str, cmd: list[str], cwd: Path) -> subprocess.Popen | None:
    if not cwd.exists():
        print(f"[skip] {name}: missing directory {cwd}")
        return None
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        print(f"[skip] {name}: {exc}")
        return None
    PROCESSES.append(ManagedProcess(name=name, process=process))
    return process


def start_static_server(
    name: str,
    root: Path,
    preferred_port: int,
    host: str = "0.0.0.0",
) -> int | None:
    try:
        port = choose_port(name, preferred_port)
    except RuntimeError as exc:
        print(f"[skip] {exc}")
        return None
    process = start_process(
        name,
        [sys.executable, "-m", "http.server", "--bind", host, str(port)],
        cwd=root,
    )
    if not process:
        return None
    time.sleep(0.2)
    if process.poll() is not None:
        print(f"[skip] {name}: server exited immediately")
        return None
    print(f"[ok] {name}: http://localhost:{port}/")
    return port


def prepare_timeline_root() -> Path | None:
    timeline = REPO_ROOT / "DATA_SET_BUILDER_AND_EXPLORER" / "Timeline_Maker"
    if not (timeline / "visualiser.html").exists():
        return None

    temp = tempfile.TemporaryDirectory(prefix="timeline_visualiser_")
    TEMP_DIRS.append(temp)
    root = Path(temp.name)

    for html_name in ("visualiser.html", "visualiser_dual.html"):
        src = timeline / html_name
        if src.exists():
            (root / html_name).symlink_to(src)

    # These two legacy HTML files use relative paths from an older dump layout.
    # Symlinking keeps the source files unchanged while making their fetch()
    # calls resolve correctly.
    links = {
        "output_merged": timeline / "dump" / "output_merged",
        "family_matrimonial_timed": timeline / "dump" / "family_matrimonial_timed",
        "output_merged_fin_fraud": timeline / "dump" / "output_merged_fin_fraud",
    }
    for name, target in links.items():
        if target.exists():
            (root / name).symlink_to(target, target_is_directory=True)

    return root


def micromamba_python(env_name: str) -> list[str] | None:
    if shutil.which("micromamba"):
        return ["micromamba", "run", "-n", env_name, "python"]
    return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def compact_number(value: int | float) -> str:
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def annotation_inventory(final_outputs: Path) -> dict[str, int]:
    """Count non-empty NER/RR annotations actually available to the inspector."""
    inventory: dict[str, int] = {}
    for extract_dir in sorted(final_outputs.glob("*_extract")):
        annotations = extract_dir / "annotations"
        if not annotations.is_dir():
            continue
        bucket = extract_dir.name.removesuffix("_extract")
        inventory[bucket] = sum(
            1 for path in annotations.glob("*.json") if path.stat().st_size > 0
        )
    return inventory


def skipped_site(
    slug: str,
    name: str,
    description: str,
    focus: str,
    reason: str,
    *,
    featured: bool = False,
) -> Site:
    return Site(
        slug=slug,
        name=name,
        url="",
        description=description,
        kind="Unavailable",
        focus=focus,
        metrics=(reason,),
        featured=featured,
        status="skipped",
    )


def start_apps(args: argparse.Namespace, sites: list[Site]) -> None:
    graph_python = micromamba_python("graph_vis")

    # The final explainer is the primary paper-facing explanation browser and is
    # intentionally started even with --static-only (it is a stdlib HTTP app,
    # not Dash).
    explainer_dir = REPO_ROOT / "FINAL_EXPLANATION"
    explainer_artifact = (
        explainer_dir
        / "outputs"
        / "entity_resolved_section_sep_lr_decay_cross_bucket_fold00"
        / "case_summary.csv"
    )
    explainer_python = micromamba_python("thesis_work")
    explain_summary = read_json(explainer_artifact.parent / "run_summary.json")
    if explainer_python and (explainer_dir / "visualizer.py").exists() and explainer_artifact.exists():
        explainer_port = choose_port("Final explanation visualizer", args.explainer_port)
        cmd = explainer_python + [
            "visualizer.py",
            "--host",
            args.host,
            "--port",
            str(explainer_port),
        ]
        if start_process("Final explanation visualizer", cmd, explainer_dir):
            sites.append(
                Site(
                    slug="explainability",
                    name="Prediction Explainability",
                    url=f"http://localhost:{explainer_port}/",
                    description=(
                        "Inspect case-level counterfactual evidence, typed legal paths, faithfulness curves, "
                        "prediction buckets, and identity/leakage audits for the frozen HGT."
                    ),
                    kind="Examiner view 02",
                    focus="Why did the model predict this outcome?",
                    metrics=(
                        f"{compact_number(explain_summary.get('processed_cases', 0))} explained cases",
                        f"{compact_number(explain_summary.get('processed_groups', 0))} evidence masks",
                        "Counterfactual + attention audit",
                    ),
                    featured=True,
                )
            )
            print(f"[ok] Final explanation visualizer: http://localhost:{explainer_port}/")
    else:
        reason = "Explanation output or thesis_work environment is missing"
        sites.append(
            skipped_site(
                "explainability",
                "Prediction Explainability",
                "Case-level counterfactual explanations and validation outputs.",
                "Why did the model predict this outcome?",
                reason,
                featured=True,
            )
        )

    if args.static_only:
        return

    pipeline_dir = REPO_ROOT / "STAGE_VISUALISER"
    pipeline_outputs = REPO_ROOT / "Fixed_GPU_OpenNyai" / "final_outputs"
    ner_inventory = annotation_inventory(pipeline_outputs)
    browsable_annotations = sum(ner_inventory.values())
    populated_buckets = sum(count > 0 for count in ner_inventory.values())
    if graph_python and (pipeline_dir / "app.py").exists() and pipeline_outputs.exists():
        pipeline_port = choose_port("NER and rhetorical-role inspector", args.pipeline_stage_port)
        cmd = graph_python + [
            "app.py",
            "--host",
            args.host,
            "--port",
            str(pipeline_port),
        ]
        if start_process("NER and rhetorical-role inspector", cmd, pipeline_dir):
            sites.append(
                Site(
                    slug="ner-rr",
                    name="NER + Rhetorical Roles",
                    url=f"http://localhost:{pipeline_port}/",
                    description=(
                        "Choose a legal domain and case, then inspect highlighted named entities and rhetorical "
                        "roles alongside summaries and outcome-label stages."
                    ),
                    kind="Examiner view 01",
                    focus="What structured information was extracted from each judgment?",
                    metrics=(
                        f"{compact_number(browsable_annotations)} browsable annotations",
                        f"{populated_buckets}/{len(ner_inventory)} buckets populated",
                        "Sentence-level entities + rhetorical roles",
                    ),
                    featured=True,
                )
            )
            print(f"[ok] NER + RR inspector: http://localhost:{pipeline_port}/")
    else:
        sites.append(
            skipped_site(
                "ner-rr",
                "NER + Rhetorical Roles",
                "Per-case OpenNyAI extraction and pipeline-stage inspection.",
                "What structured information was extracted from each judgment?",
                "OpenNyAI outputs or graph_vis environment is missing",
                featured=True,
            )
        )

    hearing_dir = REPO_ROOT / "section_GNN" / "multi_hearing_stage_test" / "visualiser"
    hearing_outputs = REPO_ROOT / "section_GNN" / "multi_hearing_stage_test" / "outputs"
    hearing_artifact = hearing_outputs / "analysis" / "stage_transitions.csv"
    hearing_summary = read_json(hearing_outputs / "analysis" / "summary.json")
    early_summary = read_json(
        hearing_outputs / "analysis" / "early_signal_test" / "early_signal_summary.json"
    )
    if graph_python and (hearing_dir / "app.py").exists() and hearing_artifact.exists():
        hearing_port = choose_port("Multi-hearing and early-detection visualizer", args.stage_dash_port)
        cmd = graph_python + [
            "app.py",
            "--host",
            args.host,
            "--port",
            str(hearing_port),
        ]
        if start_process("Multi-hearing and early-detection visualizer", cmd, hearing_dir):
            accuracy = 100 * float(hearing_summary.get("final_pred_accuracy", 0))
            early_accuracy = 100 * float(early_summary.get("early_correct_rate", 0))
            correction_rate = 100 * float(
                early_summary.get("correction_rate_among_initially_wrong", 0)
            )
            sites.append(
                Site(
                    slug="multi-hearing",
                    name="Multi-Hearing + Early Detection",
                    url=f"http://localhost:{hearing_port}/",
                    description=(
                        "Replay predictions as hearings accumulate. Review early-stage correctness, transition "
                        "patterns, confidence movement, decisive factors, and the evidence timeline for each case."
                    ),
                    kind="Examiner views 03–04",
                    focus="How early is the outcome detectable, and what changes across hearings?",
                    metrics=(
                        f"{compact_number(hearing_summary.get('n_cases', 0))} case timelines",
                        f"{early_accuracy:.1f}% correct at first hearing "
                        f"({compact_number(early_summary.get('n_cases', 0))}-case audit)",
                        f"{correction_rate:.1f}% of early errors later corrected",
                        f"{accuracy:.1f}% final-stage accuracy",
                    ),
                    featured=True,
                )
            )
            print(f"[ok] Multi-hearing + early detection: http://localhost:{hearing_port}/")
    else:
        sites.append(
            skipped_site(
                "multi-hearing",
                "Multi-Hearing + Early Detection",
                "Hearing-by-hearing prediction replay and early-detection results.",
                "How early is the outcome detectable, and what changes across hearings?",
                "Stage-transition outputs or graph_vis environment is missing",
                featured=True,
            )
        )

    if args.no_extras:
        return

    graph_dir = REPO_ROOT / "GRAPH_VISUALISER"
    graph_artifact = graph_dir / "outputs" / "graph_sample.pkl"
    graph_stats = read_json(graph_dir / "outputs" / "stats.json")
    if graph_python and graph_artifact.exists():
        graph_port = choose_port("Legal case graph visualizer", args.graph_dash_port)
        cmd = graph_python + ["app.py", "--config", "config.yaml", "--port", str(graph_port)]
        if start_process("Legal case graph visualizer", cmd, graph_dir):
            sites.append(
                Site(
                    slug="case-graph",
                    name="Legal Case Graph Explorer",
                    url=f"http://localhost:{graph_port}/",
                    description="Explore sampled cases, shared legal entities, hubs, bridges, and connected-case paths.",
                    kind="Supplementary",
                    focus="How are cases connected through legal entities?",
                    metrics=(
                        f"{compact_number(graph_stats.get('total_nodes', 0))} full-graph nodes",
                        f"{compact_number(graph_stats.get('total_edges', 0))} full-graph edges",
                    ),
                )
            )
            print(f"[ok] Legal case graph: http://localhost:{graph_port}/")
    else:
        sites.append(
            skipped_site(
                "case-graph",
                "Legal Case Graph Explorer",
                "Interactive case/entity graph exploration.",
                "How are cases connected through legal entities?",
                "Graph artefacts or graph_vis environment is missing",
            )
        )

    entity_dir = graph_dir / "entity_analysis"
    entity_artifact = entity_dir / "outputs" / "cross_bucket" / "cross_bucket_analysis.json"
    entity_summary = read_json(entity_dir / "outputs" / "within_bucket" / "_summary.json")
    if graph_python and (entity_dir / "app.py").exists() and entity_artifact.exists():
        entity_port = choose_port("Entity network analysis", args.entity_dash_port)
        cmd = graph_python + [
            "app.py",
            "--host",
            args.host,
            "--port",
            str(entity_port),
        ]
        if start_process("Entity network analysis", cmd, entity_dir):
            bucket_count = len(entity_summary) if isinstance(entity_summary, list) else 0
            sites.append(
                Site(
                    slug="entity-network",
                    name="Entity Network Analysis",
                    url=f"http://localhost:{entity_port}/",
                    description="Compare within-domain and cross-domain entity co-occurrence networks and centrality rankings.",
                    kind="Supplementary",
                    focus="Which statutes, provisions, courts, and people bridge domains?",
                    metrics=(f"{bucket_count} legal domains", "Within + cross-domain views"),
                )
            )
            print(f"[ok] Entity network analysis: http://localhost:{entity_port}/")
    else:
        sites.append(
            skipped_site(
                "entity-network",
                "Entity Network Analysis",
                "Within-domain and cross-domain entity networks.",
                "Which entities bridge legal domains?",
                "Entity-analysis outputs or graph_vis environment is missing",
            )
        )


def write_hub(root: Path, sites: list[Site]) -> None:
    def card(site: Site) -> str:
        name = html_lib.escape(site.name)
        description = html_lib.escape(site.description)
        kind = html_lib.escape(site.kind)
        focus = html_lib.escape(site.focus)
        metric_html = "".join(
            f'<span class="metric">{html_lib.escape(metric)}</span>' for metric in site.metrics
        )
        if site.url and site.status == "started":
            link = (
                f'<a class="open app-link" href="{html_lib.escape(site.url)}" '
                f'data-url="{html_lib.escape(site.url)}" target="_blank" rel="noreferrer">'
                'Open dashboard <span aria-hidden="true">↗</span></a>'
            )
            state = '<span class="state live"><i></i> Available</span>'
        else:
            link = '<span class="open disabled">Not available</span>'
            state = '<span class="state unavailable"><i></i> Missing output</span>'
        return f"""
        <article class="card {'featured' if site.featured else ''}" id="{html_lib.escape(site.slug)}">
          <div class="card-top"><span class="meta">{kind}</span>{state}</div>
          <h3>{name}</h3>
          <p class="focus">{focus}</p>
          <p class="description">{description}</p>
          <div class="metrics">{metric_html}</div>
          <div class="card-action">{link}</div>
        </article>
        """

    examiner_order = {"ner-rr": 1, "explainability": 2, "multi-hearing": 3}
    featured_sites = sorted(
        (site for site in sites if site.featured),
        key=lambda site: examiner_order.get(site.slug, 99),
    )
    featured = "".join(card(site) for site in featured_sites)
    supplementary = "".join(card(site) for site in sites if not site.featured)
    available = sum(site.status == "started" for site in sites)
    total = len(sites)

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>Legal Judgment Prediction · Examiner Dashboard</title>
  <style>
    :root {{ --ink:#edf3f8; --muted:#9eabb9; --line:#293442; --panel:#151c24; --panel2:#10171f; --teal:#5eead4; --amber:#f5bd62; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:#0b1117; font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif; }}
    body::before {{ content:""; position:fixed; inset:0; pointer-events:none; background:radial-gradient(circle at 80% 0%,rgba(38,132,129,.18),transparent 34%),radial-gradient(circle at 5% 22%,rgba(54,84,128,.16),transparent 28%); }}
    main {{ position:relative; max-width:1240px; margin:auto; padding:34px 24px 64px; }}
    .eyebrow {{ color:var(--teal); font:700 11px/1.2 ui-monospace,SFMono-Regular,monospace; letter-spacing:.16em; text-transform:uppercase; }}
    header {{ padding:28px 0 34px; border-bottom:1px solid var(--line); }}
    .header-row {{ display:flex; justify-content:space-between; gap:28px; align-items:end; }}
    h1 {{ max-width:780px; margin:12px 0 12px; font-family:Georgia,serif; font-size:clamp(36px,5vw,62px); font-weight:500; line-height:1.02; letter-spacing:-.035em; }}
    .lede {{ max-width:760px; margin:0; color:#bcc7d2; font-size:17px; line-height:1.65; }}
    .availability {{ min-width:150px; border-left:1px solid var(--line); padding-left:24px; color:var(--muted); }}
    .availability strong {{ display:block; color:var(--ink); font:500 34px/1 Georgia,serif; margin-bottom:8px; }}
    .path {{ display:grid; grid-template-columns:repeat(4,1fr); margin:26px 0 0; border:1px solid var(--line); border-radius:10px; overflow:hidden; background:rgba(15,22,29,.7); }}
    .path a {{ color:var(--muted); padding:14px 16px; text-decoration:none; border-right:1px solid var(--line); font-size:13px; }}
    .path a:last-child {{ border-right:0; }} .path b {{ color:var(--amber); margin-right:7px; }}
    section {{ padding-top:34px; }}
    .section-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin-bottom:16px; }}
    h2 {{ margin:0; font:500 25px/1.2 Georgia,serif; }}
    .section-note {{ color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; grid-template-columns:repeat(12,1fr); gap:16px; }}
    .card {{ grid-column:span 6; display:flex; flex-direction:column; min-height:302px; padding:22px; border:1px solid var(--line); border-radius:12px; background:linear-gradient(145deg,rgba(24,33,43,.96),rgba(15,22,29,.96)); box-shadow:0 16px 44px rgba(0,0,0,.15); }}
    .card.featured:first-child {{ grid-column:span 6; }}
    .card-top {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }}
    .meta {{ color:#7dd3fc; font:700 10px/1 ui-monospace,SFMono-Regular,monospace; letter-spacing:.12em; text-transform:uppercase; }}
    .state {{ display:inline-flex; align-items:center; gap:7px; color:var(--muted); font-size:11px; }}
    .state i {{ width:7px; height:7px; border-radius:50%; background:#64748b; }} .state.live i {{ background:#34d399; box-shadow:0 0 0 4px rgba(52,211,153,.1); }}
    h3 {{ margin:26px 0 7px; font:500 27px/1.15 Georgia,serif; }}
    .focus {{ color:var(--amber); font-size:14px; line-height:1.45; margin:0 0 12px; }}
    .description {{ color:#aebbc8; line-height:1.55; font-size:14px; margin:0; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:7px; margin:18px 0; }}
    .metric {{ border:1px solid #334154; background:#101720; color:#c7d2df; padding:6px 8px; border-radius:5px; font:600 11px/1.2 ui-monospace,SFMono-Regular,monospace; }}
    .card-action {{ margin-top:auto; }}
    .open {{ display:inline-flex; align-items:center; gap:10px; color:#09201d; background:var(--teal); padding:10px 14px; border-radius:6px; text-decoration:none; font-size:13px; font-weight:800; }}
    .open:hover {{ background:#99f6e4; transform:translateY(-1px); }} .open.disabled {{ color:#7f8a96; background:#26313d; }}
    .supplementary .card {{ grid-column:span 6; min-height:260px; }}
    footer {{ margin-top:42px; padding-top:20px; border-top:1px solid var(--line); color:#8492a1; font-size:12px; line-height:1.6; }}
    code {{ color:#cbd5e1; background:#111923; padding:3px 6px; border:1px solid #263240; border-radius:4px; }}
    @media (max-width:760px) {{ main {{ padding:20px 14px 44px; }} .header-row {{ display:block; }} .availability {{ margin-top:22px; border-left:0; padding-left:0; }} .path {{ grid-template-columns:1fr 1fr; }} .path a:nth-child(2) {{ border-right:0; }} .path a:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }} .card,.card.featured:first-child,.supplementary .card {{ grid-column:1/-1; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Thesis examination · output browser</div>
      <div class="header-row">
        <div><h1>Legal Judgment Prediction, from extraction to explanation.</h1><p class="lede">A single entry point to the thesis outputs. The primary path follows what was extracted, why the model predicted an outcome, and how that prediction evolved over multiple hearings.</p></div>
        <div class="availability"><strong>{available}/{total}</strong>dashboards available</div>
      </div>
      <nav class="path" aria-label="Suggested examination order">
        <a href="#ner-rr"><b>01</b> NER + RR</a><a href="#explainability"><b>02</b> Explainability</a><a href="#multi-hearing"><b>03</b> Multi-hearing</a><a href="#multi-hearing"><b>04</b> Early detection</a>
      </nav>
    </header>
    <section>
      <div class="section-head"><h2>Examiner views</h2><span class="section-note">Suggested order: 01 → 04</span></div>
      <div class="grid">{featured}</div>
    </section>
    {f'<section class="supplementary"><div class="section-head"><h2>Supplementary exploration</h2><span class="section-note">Graph structure and corpus-wide entity patterns</span></div><div class="grid">{supplementary}</div></section>' if supplementary else ''}
    <footer>Links automatically use the hostname from which this hub was opened, so they work on localhost or a directly reachable research server. Keep <code>run_visualisers.py</code> running; press Ctrl+C in its terminal to stop every child server.</footer>
  </main>
  <script>
    document.querySelectorAll('.app-link').forEach((link) => {{
      const target = new URL(link.dataset.url);
      target.hostname = window.location.hostname;
      link.href = target.toString();
    }});
  </script>
</body>
</html>
"""
    (root / "index.html").write_text(textwrap.dedent(page), encoding="utf-8")


def cleanup() -> None:
    for managed in PROCESSES:
        process = managed.process
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for managed in PROCESSES:
        try:
            managed.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(managed.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for temp in TEMP_DIRS:
        temp.cleanup()


def main() -> None:
    args = parse_args()
    atexit.register(cleanup)

    sites: list[Site] = []
    start_apps(args, sites)

    hub_temp = tempfile.TemporaryDirectory(prefix="visualiser_hub_")
    TEMP_DIRS.append(hub_temp)
    hub_root = Path(hub_temp.name)
    write_hub(hub_root, sites)
    hub_port = start_static_server("Visualiser hub", hub_root, args.hub_port, host=args.host)
    if hub_port:
        hub_url = f"http://localhost:{hub_port}/"
        print("")
        print(f"Examiner hub: {hub_url}")
        app_ports = [site.url.split(":")[2].split("/")[0] for site in sites if site.url]
        forwards = " ".join(f"-L {port}:localhost:{port}" for port in [str(hub_port), *app_ports])
        print("Remote access (run from the examiner's computer):")
        print(f"  ssh {forwards} <user>@<server>")
        print("Keep this terminal open. Press Ctrl+C to stop all servers.")
        if not args.no_browser:
            webbrowser.open(hub_url)

    try:
        while True:
            time.sleep(1)
            for managed in PROCESSES:
                if managed.process.poll() is not None:
                    print(f"[exit] {managed.name} stopped with code {managed.process.returncode}")
                    PROCESSES.remove(managed)
                    break
    except KeyboardInterrupt:
        print("\nStopping visualiser servers...")


if __name__ == "__main__":
    main()
