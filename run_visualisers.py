#!/usr/bin/env python3
"""
Launch a single local hub for the repo's visualisers.

Default behaviour starts the static HTML viewers and a hub page. Dash apps are
enabled by default when their expected artefacts are present; use --static-only
to skip them.

Usage:
  python3 run_visualisers.py
  python3 run_visualisers.py --static-only
  python3 run_visualisers.py --no-browser
"""
from __future__ import annotations

import argparse
import atexit
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
    name: str
    url: str
    description: str
    kind: str
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
    parser.add_argument("--hub-port", type=int, default=8090)
    parser.add_argument("--timeline-port", type=int, default=8081)
    parser.add_argument("--gnn-static-port", type=int, default=8082)
    parser.add_argument("--explainer-port", type=int, default=8083)
    parser.add_argument("--embedding-port", type=int, default=8084)
    parser.add_argument("--graph-dash-port", type=int, default=8050)
    parser.add_argument("--stage-dash-port", type=int, default=8051)
    parser.add_argument("--entity-dash-port", type=int, default=8052)
    parser.add_argument("--pipeline-stage-port", type=int, default=8053)
    parser.add_argument("--static-only", action="store_true", help="Do not start Dash apps.")
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


def start_static_server(name: str, root: Path, preferred_port: int) -> int | None:
    try:
        port = choose_port(name, preferred_port)
    except RuntimeError as exc:
        print(f"[skip] {exc}")
        return None
    process = start_process(name, [sys.executable, "-m", "http.server", str(port)], cwd=root)
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


def start_dash_apps(args: argparse.Namespace, sites: list[Site]) -> None:
    if args.static_only:
        return

    graph_dir = REPO_ROOT / "GRAPH_VISUALISER"
    graph_artifact = graph_dir / "outputs" / "graph_sample.pkl"
    graph_python = micromamba_python("graph_vis")
    if graph_artifact.exists() and graph_python:
        graph_dash_port = choose_port("Dash graph visualiser", args.graph_dash_port)
        cmd = graph_python + ["app.py", "--config", "config.yaml", "--port", str(graph_dash_port)]
        if start_process("Dash graph visualiser", cmd, graph_dir):
            sites.append(
                Site(
                    name="Dash Legal Case Graph Visualiser",
                    url=f"http://localhost:{graph_dash_port}/",
                    description="Interactive Plotly/Dash graph over the sampled legal case graph, with case/entity details and bridge views.",
                    kind="Dash",
                )
            )
            print(f"[ok] Dash graph visualiser: http://localhost:{graph_dash_port}/")
    else:
        reason = "missing graph_sample.pkl" if not graph_artifact.exists() else "micromamba not found"
        sites.append(
            Site(
                name="Dash Legal Case Graph Visualiser",
                url="",
                description="Skipped: " + reason + ".",
                kind="Dash",
                status="skipped",
            )
        )

    # entity_dir = graph_dir / "entity_analysis"
    # entity_python = graph_python
    # if entity_python and (entity_dir / "app.py").exists():
    #     cmd = entity_python + ["app.py", "--port", str(args.entity_dash_port)]
    #     if start_process("Dash entity network visualiser", cmd, entity_dir):
    #         sites.append(
    #             Site(
    #                 name="Entity Network Analysis",
    #                 url=f"http://localhost:{args.entity_dash_port}/",
    #                 description="Interactive within-bucket and cross-bucket entity co-occurrence network analysis.",
    #                 kind="Dash",
    #             )
    #         )
    #         print(f"[ok] Entity network visualiser: http://localhost:{args.entity_dash_port}/")

    pipeline_stage_dir = REPO_ROOT / "STAGE_VISUALISER"
    pipeline_stage_artifact = REPO_ROOT / "Fixed_GPU_OpenNyai" / "final_outputs"
    if graph_python and (pipeline_stage_dir / "app.py").exists() and pipeline_stage_artifact.exists():
        pipeline_stage_port = choose_port("Pipeline stage visualiser", args.pipeline_stage_port)
        cmd = graph_python + ["app.py", "--port", str(pipeline_stage_port)]
        if start_process("Pipeline stage visualiser", cmd, pipeline_stage_dir):
            sites.append(
                Site(
                    name="Pipeline Stage Visualiser",
                    url=f"http://localhost:{pipeline_stage_port}/",
                    description="Per-case inspector for the Fixed_GPU_OpenNyai pipeline: NER+RR extracts, OpenNyai summary, Mistral outcome label, and cross-validated outcome side-by-side.",
                    kind="Dash",
                )
            )
            print(f"[ok] Pipeline stage visualiser: http://localhost:{pipeline_stage_port}/")
    else:
        reason = (
            "missing Fixed_GPU_OpenNyai/final_outputs"
            if not pipeline_stage_artifact.exists()
            else "micromamba not found"
            if not graph_python
            else "missing app.py"
        )
        sites.append(
            Site(
                name="Pipeline Stage Visualiser",
                url="",
                description="Skipped: " + reason + ".",
                kind="Dash",
                status="skipped",
            )
        )

    stage_dir = REPO_ROOT / "section_GNN" / "multi_hearing_stage_test" / "visualiser"
    stage_outputs = REPO_ROOT / "section_GNN" / "multi_hearing_stage_test" / "outputs"
    stage_artifact = stage_outputs / "analysis" / "stage_transitions.csv"
    if graph_python and (stage_dir / "app.py").exists() and stage_artifact.exists():
        stage_dash_port = choose_port("Multi-hearing stage visualiser", args.stage_dash_port)
        cmd = graph_python + ["app.py", "--port", str(stage_dash_port)]
        if start_process("Multi-hearing stage visualiser", cmd, stage_dir):
            sites.append(
                Site(
                    name="Multi-Hearing Stage Test",
                    url=f"http://localhost:{stage_dash_port}/",
                    description="Dash view of stage-to-stage prediction changes, raw actual outcomes 0/-1/1, transition factors, and per-case evidence.",
                    kind="Dash",
                )
            )
            print(f"[ok] Multi-hearing stage visualiser: http://localhost:{stage_dash_port}/")
    else:
        reason = (
            "missing stage_transitions.csv"
            if not stage_artifact.exists()
            else "micromamba not found"
            if not graph_python
            else "missing app.py"
        )
        sites.append(
            Site(
                name="Multi-Hearing Stage Test",
                url="",
                description="Skipped: " + reason + ".",
                kind="Dash",
                status="skipped",
            )
        )


def write_hub(root: Path, sites: list[Site]) -> None:
    cards = []
    for site in sites:
        link = (
            f'<a class="open" href="{site.url}" target="_blank" rel="noreferrer">Open</a>'
            if site.url and site.status == "started"
            else '<span class="open disabled">Skipped</span>'
        )
        cards.append(
            f"""
            <article class="card">
              <div class="meta">{site.kind}</div>
              <h2>{site.name}</h2>
              <p>{site.description}</p>
              {link}
            </article>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Thesis Visualiser Hub</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, sans-serif; background: #10131a; color: #e5e7eb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ margin: 0 0 24px; color: #9ca3af; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #2f3548; background: #181d29; border-radius: 8px; padding: 18px; min-height: 160px; }}
    .meta {{ color: #7dd3fc; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }}
    h2 {{ font-size: 17px; margin: 8px 0 8px; }}
    p {{ color: #cbd5e1; line-height: 1.5; margin: 0 0 18px; }}
    .open {{ display: inline-block; color: #07111f; background: #93c5fd; padding: 8px 12px; border-radius: 6px; text-decoration: none; font-weight: 700; }}
    .open.disabled {{ background: #374151; color: #9ca3af; }}
    code {{ background: #111827; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <main>
    <h1>Thesis Visualiser Hub</h1>
    <p class="sub">Started by <code>run_visualisers.py</code>. Keep this terminal open; press Ctrl+C to stop all child servers.</p>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    (root / "index.html").write_text(textwrap.dedent(html), encoding="utf-8")


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

    # timeline_root = prepare_timeline_root()
    # if timeline_root and start_static_server("Timeline merger visualisers", timeline_root, args.timeline_port):
    #     sites.extend(
    #         [
    #             Site(
    #                 name="Case Merger Visualiser",
    #                 url=f"http://localhost:{args.timeline_port}/visualiser.html",
    #                 description="Single-report timeline/merge viewer for duplicate and multi-date case JSONs, including merged text and raw JSON tabs.",
    #                 kind="Static HTML",
    #             ),
    #             Site(
    #                 name="Dual-Bucket Case Merger Visualiser",
    #                 url=f"http://localhost:{args.timeline_port}/visualiser_dual.html",
    #                 description="Two-bucket merger viewer comparing family/matrimonial and financial-fraud reports, with cross-bucket matches highlighted.",
    #                 kind="Static HTML",
    #             ),
    #         ]
    #     )

    # gnn_static = REPO_ROOT / "section_GNN" / "dump2" / "visualiser"
    # if start_static_server("Legal GNN static visualiser", gnn_static, args.gnn_static_port):
    #     sites.extend(
    #         [
    #             Site(
    #                 name="Legal GNN Case Graph Visualiser",
    #                 url=f"http://localhost:{args.gnn_static_port}/",
    #                 description="D3 case-star graph viewer for selected cases, shared legal entities, connected cases, and node/edge metadata.",
    #                 kind="Static HTML",
    #             ),
    #             Site(
    #                 name="Entity Stats Dashboard",
    #                 url=f"http://localhost:{args.gnn_static_port}/stats/",
    #                 description="Static dashboard of extracted entity statistics, outcomes, courts, statutes, judges, lawyers, and parties.",
    #                 kind="Static HTML",
    #             ),
    #             Site(
    #                 name="Layers Explorer",
    #                 url=f"http://localhost:{args.gnn_static_port}/layers/",
    #                 description="Static explainer for graph layers, HGT model layers, and how related-case links should be read.",
    #                 kind="Static HTML",
    #             ),
    #         ]
    #     )

    explainer_root = REPO_ROOT / "Graph_Analyser"
    explainer_port = start_static_server("GNN explainer UI", explainer_root, args.explainer_port)
    if explainer_port:
        sites.append(
            Site(
                name="GNN Explainer UI",
                url=f"http://localhost:{explainer_port}/UI/",
                description="Browser UI for Phase 5 explanations, pipeline-stage evidence, final summaries, reasoning graphs, and dataset analytics.",
                kind="Static HTML",
            )
        )

    # embedding_root = REPO_ROOT / "DATA_SET_BUILDER_AND_EXPLORER" / "ENCODING_CLASSIFICATION"
    # embedding_html = embedding_root / "outputs" / "figures_interactive" / "interactive_3d_pca.html"
    # if embedding_html.exists() and start_static_server("Embedding projection viewer", embedding_root, args.embedding_port):
    #     sites.append(
    #         Site(
    #             name="Interactive 3D Case Embeddings",
    #             url=f"http://localhost:{args.embedding_port}/outputs/figures_interactive/interactive_3d_pca.html",
    #             description="Plotly 3D PCA projection of case embeddings, switchable between bucket and discovered-cluster colouring.",
    #             kind="Static HTML",
    #         )
    #     )

    start_dash_apps(args, sites)

    hub_temp = tempfile.TemporaryDirectory(prefix="visualiser_hub_")
    TEMP_DIRS.append(hub_temp)
    hub_root = Path(hub_temp.name)
    write_hub(hub_root, sites)
    hub_port = start_static_server("Visualiser hub", hub_root, args.hub_port)
    if hub_port:
        hub_url = f"http://localhost:{hub_port}/"
        print("")
        print(f"Hub: {hub_url}")
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
