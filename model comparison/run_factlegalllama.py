#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from prepare_factlegalllama_adapter import prepare_adapter


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a TathyaNyaya/FactLegalLlama adapter checkpoint, then run the shared "
            "filtered-case inference script."
        )
    )
    parser.add_argument("--checkpoint", default="nyaya_facts_single")
    parser.add_argument("--adapter-root", type=Path, default=Path("models/factlegalllama"))
    parser.add_argument("--cache-dir", default="/scratch/ziv_baretto/Thesis_Ziv/hf_cache")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--print-adapter-dir", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    adapter_dir = prepare_adapter(
        checkpoint=args.checkpoint,
        output_root=args.adapter_root,
        cache_dir=args.cache_dir,
        force=args.force_extract,
    )
    if args.prepare_only or args.print_adapter_dir:
        print(adapter_dir)
        return

    script_dir = Path(__file__).resolve().parent
    run_script = script_dir / "run_inlegalllama.py"
    os.environ.setdefault("HF_HOME", args.cache_dir)

    sys.argv = [
        str(run_script),
        "--model-name",
        str(adapter_dir),
        "--model-subfolder",
        "",
        "--adapter-mode",
        "peft",
        *passthrough,
    ]
    from run_inlegalllama import main as run_inference

    run_inference()


if __name__ == "__main__":
    main()
