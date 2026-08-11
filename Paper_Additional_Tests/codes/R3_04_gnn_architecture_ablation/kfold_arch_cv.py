#!/usr/bin/env python3
"""Run the paper's own K-fold harness with a different message-passing operator.

This script deliberately adds **no** training, splitting or metric logic. It
injects :class:`ArchLegalOutcomeGNN` into ``train_v2`` and then hands control to
``kfold_cv_v2.main()``, so the folds, the optimiser, the early stopping, the
per-fold ``predictions.csv``/``metrics.json`` and the ``kfold_summary.json``
aggregation are byte-for-byte the same code that produced the paper numbers.

``train_v2.train_model`` resolves ``HeteroLegalOutcomeGNN`` as a module global at
call time, so rebinding that name is sufficient and nothing under
``section_GNN/src`` or ``section_GNN/runs_v2`` is modified.

Usage mirrors ``kfold_cv_v2.py`` exactly::

    python kfold_arch_cv.py --config configs/arch_gcn.yaml \
        --graph-cache /abs/path/to/graph.pt --run-name arch_gcn_kfold --fold 0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SECTION_GNN = _HERE.parents[2] / "section_GNN"
_V2_SCRIPTS = _SECTION_GNN / "runs_v2" / "party_args_lr_decay" / "scripts"
for _path in (str(_HERE), str(_V2_SCRIPTS), str(_SECTION_GNN)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import kfold_cv_v2  # noqa: E402
import train_v2  # noqa: E402
from arch_gnn import ARCHITECTURE_PROPERTIES, ArchLegalOutcomeGNN  # noqa: E402
from src.utils.io import dump_json, ensure_dir, load_yaml  # noqa: E402

# Filled in by the recording wrapper below, then written next to the fold outputs.
_MODEL_INFO: dict[str, int | str] = {}


class _RecordingArchGNN(ArchLegalOutcomeGNN):
    """Identical to ArchLegalOutcomeGNN; also records its parameter count."""

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        _MODEL_INFO["architecture"] = self.architecture
        _MODEL_INFO["n_parameters"] = sum(p.numel() for p in self.parameters() if p.requires_grad)


def _write_model_info(config_path: str, run_name: str) -> None:
    if not _MODEL_INFO:
        return
    cfg = load_yaml(config_path)
    run_dir = ensure_dir(Path(cfg.get("paths", {})["outputs_dir"]) / "models" / run_name / "kfold")
    architecture = str(_MODEL_INFO.get("architecture", ""))
    payload = dict(_MODEL_INFO)
    payload.update(ARCHITECTURE_PROPERTIES.get(architecture, {}))
    payload["run_name"] = run_name
    dump_json(payload, run_dir / "arch_info.json")


def main() -> None:
    # The injection point: train_v2.train_model looks this name up at call time.
    train_v2.HeteroLegalOutcomeGNN = _RecordingArchGNN

    argv = sys.argv[1:]
    config_path = argv[argv.index("--config") + 1] if "--config" in argv else None
    run_name = argv[argv.index("--run-name") + 1] if "--run-name" in argv else "kfold_run_v2"

    kfold_cv_v2.main()

    if config_path is not None:
        _write_model_info(config_path, run_name)


if __name__ == "__main__":
    main()
