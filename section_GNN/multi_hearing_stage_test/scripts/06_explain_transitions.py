#!/usr/bin/env python3
"""
Step 06 (optional) — Drive the Graph_Analyser explainability pipeline against
our test graph + the trained model, focused on cases whose prediction changed
between stages. For each such case it materialises a side-by-side comparison
of the top entities/precedents/provisions the explainer attributed to the
prediction at each stage.

This script writes a Graph_Analyser config snapshot and runs phases 1–4
(no LLM). Outputs land in <EXP_ROOT>/outputs/explainer/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
SECTION_GNN_ROOT = EXP_ROOT.parent
REPO_ROOT = SECTION_GNN_ROOT.parent
GRAPH_ANALYSER = REPO_ROOT / "Graph_Analyser"


def resolve_from_section_gnn(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    return (SECTION_GNN_ROOT / expanded).resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(EXP_ROOT / "config.yaml"))
    p.add_argument("--fold", type=int, default=0,
                   help="Which fold's checkpoint to explain against (Graph_Analyser uses one fold).")
    p.add_argument("--skip-explainer-train", action="store_true",
                   help="Skip phase 3 (explainer training) — phase 4 will fail without it.")
    p.add_argument("--mamba-env", default="graph_explainer")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    inf = cfg["inference"]
    test_graph = resolve_from_section_gnn(cfg["paths"]["graph_cache_dir"]) / cfg["graph"]["cache_name"]
    fold_dir = resolve_from_section_gnn(inf["trained_model_run_dir"]) / f"fold_{args.fold:02d}"
    output_root = EXP_ROOT / "outputs" / "explainer"
    output_root.mkdir(parents=True, exist_ok=True)

    explainer_cfg = {
        "section_gnn_root": str(SECTION_GNN_ROOT),
        "graph_cache": str(test_graph),
        "checkpoint_dir": str(fold_dir),
        "model": inf["model"],
        "output_root": str(output_root),
        "faiss": {"top_k_similar_cases": 3, "metric": "cosine"},
        "explainer": {
            "epochs": 30, "lr": 3.0e-3, "coeff_size": 0.01, "coeff_ent": 1.0e-4,
            "temperature_start": 5.0, "temperature_end": 1.0,
            "max_train_cases_per_epoch": 256, "hidden_dim": 64,
        },
        "extraction": {
            "top_k_statutes": 5, "top_k_provisions": 5, "top_k_precedents": 5,
            "top_k_arguments": 4, "top_k_similar_arg_contexts": 4,
            "top_n_test_cases": 99999, "target_case_text_chars": 1200,
            "similar_case_arg_chars": 500,
        },
    }
    cfg_path = output_root / "explainer_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(explainer_cfg, f)

    env_name = args.mamba_env
    def run(cmd: list[str]) -> None:
        full = ["micromamba", "run", "-n", env_name, "python"] + cmd
        print(f"[06_explain] {' '.join(full)}", flush=True)
        subprocess.run(full, check=True, cwd=str(GRAPH_ANALYSER))

    run([str(GRAPH_ANALYSER / "scripts/phase1_2_inference_and_index.py"), "--config", str(cfg_path)])
    if not args.skip_explainer_train:
        run([str(GRAPH_ANALYSER / "scripts/phase3_train_explainer.py"), "--config", str(cfg_path)])
        run([str(GRAPH_ANALYSER / "scripts/phase4_extract_importance.py"), "--config", str(cfg_path)])

    # Build the per-case stage comparison from phase4 outputs.
    transitions_csv = EXP_ROOT / "outputs/analysis/stage_transitions.csv"
    if not transitions_csv.exists():
        print("[06_explain] stage_transitions.csv missing; run step 05 first.", file=sys.stderr)
        return
    transitions = pd.read_csv(transitions_csv)
    changed = transitions[transitions["changed_prediction"] == True]

    cases_dir = output_root / "phase4_explanations" / "cases"
    if not cases_dir.exists():
        print(f"[06_explain] phase4 outputs not found at {cases_dir} — skipping comparison build.")
        return

    p1_inference = output_root / "phase1_2_inference"
    pred_df = pd.read_csv(p1_inference / "predictions.csv")
    case_id_to_node = dict(zip(pred_df["case_id"].astype(str), pred_df["node_index"].astype(int)))

    comparisons_dir = output_root / "stage_comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for _, row in changed.iterrows():
        base = row["base_case_id"]
        n_stages = int(row["n_stages"])
        stage_payloads = []
        for i in range(1, n_stages + 1):
            cid = f"STAGE{i}__{base}"
            node_idx = case_id_to_node.get(cid)
            if node_idx is None:
                continue
            case_json = cases_dir / f"case_{node_idx}.json"
            if not case_json.exists():
                continue
            with open(case_json) as f:
                stage_payloads.append({"stage_index": i, "node_index": node_idx, "explanation": json.load(f)})
        if len(stage_payloads) >= 2:
            with open(comparisons_dir / f"{base}.json", "w") as f:
                json.dump({
                    "base_case_id": base,
                    "transition": row["transition"],
                    "true_label": row["true_label"],
                    "stages": stage_payloads,
                }, f, indent=2)
            written += 1
    print(f"[06_explain] wrote {written} stage-comparison files -> {comparisons_dir}")


if __name__ == "__main__":
    main()
