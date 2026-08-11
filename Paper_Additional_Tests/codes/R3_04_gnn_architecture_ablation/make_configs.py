#!/usr/bin/env python3
"""Generate one config per architecture from the paper's own config.

Every generated file is a verbatim copy of

    section_GNN/ablations/entity_resolved_data/configs/section/
    cross_bucket_total_dataset/config.yaml

with exactly two keys changed: ``model.architecture`` and ``paths.outputs_dir``.
``check_harness_equivalence.py`` asserts that this is true, so the ablation
cannot silently drift from the reference run.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
PAPER_CONFIG = (
    _REPO
    / "section_GNN/ablations/entity_resolved_data/configs/section/cross_bucket_total_dataset/config.yaml"
)
OUTPUTS_DIR = _HERE / "outputs"

# Architectures to generate. "hgt" is intentionally absent: the paper's HGT run
# is NOT re-run; its recorded numbers are used as the reference row.
ARCHITECTURES = ["mlp", "gcn", "sage", "gat", "rgcn", "hgat"]

# Optional parameter-matched controls, so a reviewer cannot argue the simple
# GNNs were handicapped on capacity. Enabled with --wide.
WIDE_ARCHITECTURES = {"gcn_wide": ("gcn", 192), "sage_wide": ("sage", 192)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", action="store_true", help="also emit width-matched gcn/sage controls")
    args = parser.parse_args()

    base = yaml.safe_load(PAPER_CONFIG.read_text())
    config_dir = _HERE / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, tuple[str, int | None]] = {name: (name, None) for name in ARCHITECTURES}
    if args.wide:
        variants.update(WIDE_ARCHITECTURES)

    for name, (architecture, hidden_dim) in variants.items():
        cfg = copy.deepcopy(base)
        cfg["model"]["architecture"] = architecture
        cfg["paths"]["outputs_dir"] = str(OUTPUTS_DIR)
        if hidden_dim is not None:
            cfg["model"]["hidden_dim"] = hidden_dim
        path = config_dir / f"arch_{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        print(f"wrote {path.relative_to(_HERE)}  (architecture={architecture}, hidden_dim={cfg['model']['hidden_dim']})")


if __name__ == "__main__":
    main()
