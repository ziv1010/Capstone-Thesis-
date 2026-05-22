#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPDATED_GRAPH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$UPDATED_GRAPH_ROOT/.." && pwd)"

DEFAULT_BASE_CONFIG="$UPDATED_GRAPH_ROOT/fixed_open_pipeline/cross_bucket_cases_8k_each_mistral_reasoning_config.yaml"
DEFAULT_VARIANTS_CONFIG="$SCRIPT_DIR/encoder_variants.yaml"
DEFAULT_ARTIFACT_ROOT="$PROJECT_ROOT/outputs/cross_bucket_cases_8k_each_mistral_reasoning/multi_embed_test"
DEFAULT_MAMBA_PREFIX="/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/old_scripts_pt2/GNN/.micromamba/gnn_case_star"
DEFAULT_CUDA_DEVICES="0,1,2,3,4,5,6,7"
DEFAULT_RUN_PREFIX="reasoning_multi_embed_test"

BASE_CONFIG="$DEFAULT_BASE_CONFIG"
VARIANTS_CONFIG="$DEFAULT_VARIANTS_CONFIG"
ARTIFACT_ROOT="$DEFAULT_ARTIFACT_ROOT"
MAMBA_PREFIX="${MAMBA_PREFIX:-$DEFAULT_MAMBA_PREFIX}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_DEVICES}"
RUN_PREFIX="${RUN_PREFIX:-$DEFAULT_RUN_PREFIX}"
SKIP_BGE_BUILD=0
SKIP_HASHING_BUILD=0
SKIP_TRAIN=0
SKIP_SUMMARY=0
SEQUENTIAL_TRAIN=0

usage() {
    cat <<'EOF'
Usage:
  bash run_cross_bucket_bge_hashing_8gpu.sh
  bash run_cross_bucket_bge_hashing_8gpu.sh --cuda-visible-devices 0,1,2,3,4,5,6,7
  bash run_cross_bucket_bge_hashing_8gpu.sh --skip-bge-build
  bash run_cross_bucket_bge_hashing_8gpu.sh --skip-hashing-build
  bash run_cross_bucket_bge_hashing_8gpu.sh --skip-train
  bash run_cross_bucket_bge_hashing_8gpu.sh --skip-summary
  bash run_cross_bucket_bge_hashing_8gpu.sh --sequential-train

What it does:
  1. Generates two variant configs under the multi_embed_test artifact root.
  2. Builds the BGE-M3 graph first with all visible GPUs available.
  3. Builds the hashing graph second.
  4. Trains hashing and BGE-M3 models and writes the comparison summary.

Notes:
  - This script assumes the cross-bucket cleaned cases already exist.
  - BGE-M3 embedding can use all visible GPUs.
  - Hashing is CPU-only, so visible GPUs do not accelerate that build.
  - The current GNN trainer is single-GPU per process. This wrapper runs the two
    training jobs in parallel on separate GPUs for maximum throughput without code refactors.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base-config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --variants-config)
            VARIANTS_CONFIG="$2"
            shift 2
            ;;
        --artifact-root)
            ARTIFACT_ROOT="$2"
            shift 2
            ;;
        --mamba-prefix)
            MAMBA_PREFIX="$2"
            shift 2
            ;;
        --cuda-visible-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --run-prefix)
            RUN_PREFIX="$2"
            shift 2
            ;;
        --skip-bge-build)
            SKIP_BGE_BUILD=1
            shift
            ;;
        --skip-hashing-build)
            SKIP_HASHING_BUILD=1
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=1
            shift
            ;;
        --skip-summary)
            SKIP_SUMMARY=1
            shift
            ;;
        --sequential-train)
            SEQUENTIAL_TRAIN=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

PY_BIN="${PY_BIN:-$MAMBA_PREFIX/bin/python}"
BUILD_SCRIPT="$UPDATED_GRAPH_ROOT/build_graph.py"
TRAIN_SCRIPT="$PROJECT_ROOT/src/scripts/train_gnn.py"
SUMMARY_HELPER="$SCRIPT_DIR/run_fixed_open_multi_embed_test.py"

for required_path in "$BASE_CONFIG" "$VARIANTS_CONFIG" "$BUILD_SCRIPT" "$TRAIN_SCRIPT" "$SUMMARY_HELPER"; do
    if [ ! -e "$required_path" ]; then
        echo "Required path not found: $required_path" >&2
        exit 1
    fi
done

if [ ! -x "$PY_BIN" ]; then
    echo "Python interpreter not found in env prefix: $PY_BIN" >&2
    exit 1
fi

CLEANED_CASE_DIR="$("$PY_BIN" - "$BASE_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print(str(cfg.get("paths", {}).get("cleaned_case_dir", "")))
PY
)"

if [ -z "$CLEANED_CASE_DIR" ] || [ ! -d "$CLEANED_CASE_DIR" ]; then
    echo "Cleaned case dir is missing. Run preprocess first for the base config." >&2
    echo "Expected cleaned_case_dir: $CLEANED_CASE_DIR" >&2
    exit 1
fi

echo "============================================================"
echo "  Cross-Bucket BGE vs Hashing Pipeline"
echo "  Python        : $PY_BIN"
echo "  Base Config   : $BASE_CONFIG"
echo "  Variants YAML : $VARIANTS_CONFIG"
echo "  Artifact Root : $ARTIFACT_ROOT"
echo "  Visible GPUs  : $CUDA_DEVICES"
echo "  Run Prefix    : $RUN_PREFIX"
echo "  Cleaned Cases : $CLEANED_CASE_DIR"
echo "============================================================"
echo ""

echo ">>> Step 0/4: generate variant configs"
"$PY_BIN" - "$BASE_CONFIG" "$VARIANTS_CONFIG" "$ARTIFACT_ROOT" "$RUN_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML at {path} must parse to a mapping.")
    return data


def dump_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def dump_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


base_config_path = Path(sys.argv[1]).resolve()
variants_config_path = Path(sys.argv[2]).resolve()
artifact_root = Path(sys.argv[3]).resolve()
run_prefix = str(sys.argv[4])

base_cfg = load_yaml(base_config_path)
variants_payload = load_yaml(variants_config_path)
variants_cfg = dict(variants_payload.get("variants", {}))
selected_variants = ["hashing", "bge_m3"]

configs_dir = artifact_root / "configs"
configs_dir.mkdir(parents=True, exist_ok=True)

base_cache_name = str(base_cfg.get("graph", {}).get("cache_name", "reasoning_focused_graph.pt"))
cache_stem = Path(base_cache_name).stem
cache_suffix = Path(base_cache_name).suffix or ".pt"
variant_specs = []

for variant_name in selected_variants:
    variant_payload = dict(variants_cfg[variant_name])
    variant_root = artifact_root / "variants" / variant_name
    variant_outputs = variant_root / "outputs"
    variant_graph_cache = variant_root / "graph_cache"
    variant_embeddings = variant_root / "embeddings_cache"
    run_name = f"{run_prefix}__{variant_name}"

    override = {
        "project": {
            "name": f"{base_cfg.get('project', {}).get('name', 'cross_bucket')}_{variant_name}",
        },
        "paths": {
            "embeddings_cache_dir": str(variant_embeddings),
            "graph_cache_dir": str(variant_graph_cache),
            "outputs_dir": str(variant_outputs),
        },
        "graph": {
            "cache_name": f"{cache_stem}.{variant_name}{cache_suffix}",
            "debug_sample_size": 1,
        },
        "features": dict(variant_payload.get("features", {})),
        "training": {
            "repeat_runs": 1,
            "seed_stride": 1,
        },
    }
    variant_cfg = deep_merge_dict(base_cfg, override)
    config_path = configs_dir / f"{variant_name}.yaml"
    dump_yaml(variant_cfg, config_path)
    variant_specs.append(
        {
            "variant_name": variant_name,
            "description": str(variant_payload.get("description", "")),
            "config_path": str(config_path.resolve()),
            "root": str(variant_root.resolve()),
            "graph_cache_dir": str(variant_graph_cache.resolve()),
            "outputs_dir": str(variant_outputs.resolve()),
            "run_name": run_name,
            "text_encoder": variant_cfg.get("features", {}).get("text_encoder", {}),
        }
    )

manifest = {
    "base_config_path": str(base_config_path),
    "artifact_root": str(artifact_root),
    "run_prefix": run_prefix,
    "selected_variants": selected_variants,
    "variant_specs": variant_specs,
}
dump_json(manifest, artifact_root / "configs" / "run_manifest.json")
print(f"Generated configs under {configs_dir}")
for spec in variant_specs:
    print(f"  - {spec['variant_name']}: {spec['config_path']}")
PY
echo ""

BGE_CONFIG="$ARTIFACT_ROOT/configs/bge_m3.yaml"
HASHING_CONFIG="$ARTIFACT_ROOT/configs/hashing.yaml"
HASHING_RUN_NAME="${RUN_PREFIX}__hashing"
BGE_RUN_NAME="${RUN_PREFIX}__bge_m3"

if [ "$SKIP_BGE_BUILD" -eq 0 ]; then
    echo ">>> Step 1/4: build BGE-M3 graph with all visible GPUs"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PY_BIN" "$BUILD_SCRIPT" --config "$BGE_CONFIG"
    echo ""
fi

if [ "$SKIP_HASHING_BUILD" -eq 0 ]; then
    echo ">>> Step 2/4: build hashing graph"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PY_BIN" "$BUILD_SCRIPT" --config "$HASHING_CONFIG"
    echo ""
fi

IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
HASHING_TRAIN_GPU="${GPU_IDS[0]}"
BGE_TRAIN_GPU="${GPU_IDS[0]}"
if [ "${#GPU_IDS[@]}" -ge 2 ]; then
    BGE_TRAIN_GPU="${GPU_IDS[1]}"
fi

if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo ">>> Step 3/4: train hashing and BGE-M3 models"
    echo "Hashing trainer GPU: $HASHING_TRAIN_GPU"
    echo "BGE-M3 trainer GPU : $BGE_TRAIN_GPU"

    if [ "$SEQUENTIAL_TRAIN" -eq 1 ] || [ "$HASHING_TRAIN_GPU" = "$BGE_TRAIN_GPU" ]; then
        CUDA_VISIBLE_DEVICES="$HASHING_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$HASHING_CONFIG" \
            --run-name "$HASHING_RUN_NAME"

        CUDA_VISIBLE_DEVICES="$BGE_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$BGE_CONFIG" \
            --run-name "$BGE_RUN_NAME"
    else
        CUDA_VISIBLE_DEVICES="$HASHING_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$HASHING_CONFIG" \
            --run-name "$HASHING_RUN_NAME" &
        HASHING_PID=$!

        CUDA_VISIBLE_DEVICES="$BGE_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$BGE_CONFIG" \
            --run-name "$BGE_RUN_NAME" &
        BGE_PID=$!

        wait "$HASHING_PID"
        wait "$BGE_PID"
    fi
    echo ""
fi

if [ "$SKIP_SUMMARY" -eq 0 ]; then
    echo ">>> Step 4/4: write comparison summary"
    "$PY_BIN" - "$ARTIFACT_ROOT" "$SUMMARY_HELPER" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

artifact_root = Path(sys.argv[1]).resolve()
helper_path = Path(sys.argv[2]).resolve()

spec = importlib.util.spec_from_file_location("multi_embed_helper", helper_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

manifest = json.loads((artifact_root / "configs" / "run_manifest.json").read_text(encoding="utf-8"))
variant_specs = manifest["variant_specs"]

rows, checks = module.collect_variant_results(variant_specs)
module.validate_checks(checks)

summary_dir = artifact_root / "summary"
module.write_csv(rows, summary_dir / "encoder_macro_f1_comparison.csv")
module.write_markdown(rows, summary_dir / "encoder_macro_f1_comparison.md")
module.dump_json(rows, summary_dir / "encoder_macro_f1_comparison.json")
module.dump_json(checks, summary_dir / "consistency_checks.json")

print(f"Summary written to {summary_dir}")
for row in rows:
    print(
        f"{row['variant']}: "
        f"val_macro_f1={row['val_macro_f1_mean_std']} "
        f"test_macro_f1={row['test_macro_f1_mean_std']} "
        f"best_run={row['best_run_label']}"
    )
PY
    echo ""
fi

echo "Pipeline complete."
