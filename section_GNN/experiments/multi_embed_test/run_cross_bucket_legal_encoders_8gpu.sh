#!/usr/bin/env bash
# Run InCaseLawBERT, InLegalBERT, and E5-large-v2 variants against the same
# cross-bucket reasoning artifact root used for bge_m3 / hashing.
#
# Build order (sequential, avoids OOM):
#   1. e5_large_v2      – sentence_transformers, all visible GPUs (multi_process)
#   2. incaselawbert    – hf_encoder, single GPU (cuda:0 of visible set)
#   3. inlegalbert      – hf_encoder, single GPU (cuda:0 of visible set)
#
# Training (parallel, one GPU each):
#   GPU[2]: e5_large_v2
#   GPU[3]: incaselawbert
#   GPU[4]: inlegalbert
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
SKIP_E5_BUILD=0
SKIP_INCASELAWBERT_BUILD=0
SKIP_INLEGALBERT_BUILD=0
SKIP_TRAIN=0
SKIP_SUMMARY=0
SEQUENTIAL_TRAIN=0

usage() {
    cat <<'EOF'
Usage:
  bash run_cross_bucket_legal_encoders_8gpu.sh
  bash run_cross_bucket_legal_encoders_8gpu.sh --cuda-visible-devices 0,1,2,3,4,5,6,7
  bash run_cross_bucket_legal_encoders_8gpu.sh --skip-e5-build
  bash run_cross_bucket_legal_encoders_8gpu.sh --skip-incaselawbert-build
  bash run_cross_bucket_legal_encoders_8gpu.sh --skip-inlegalbert-build
  bash run_cross_bucket_legal_encoders_8gpu.sh --skip-train
  bash run_cross_bucket_legal_encoders_8gpu.sh --skip-summary
  bash run_cross_bucket_legal_encoders_8gpu.sh --sequential-train

What it does:
  1. Generates three variant configs under the shared multi_embed_test artifact root.
  2. Builds e5_large_v2 graph with all visible GPUs (sentence_transformers multi_process).
  3. Builds incaselawbert graph on a single GPU (hf_encoder).
  4. Builds inlegalbert graph on a single GPU (hf_encoder).
  5. Trains all three models in parallel on separate GPUs.
  6. Writes comparison summary for these three variants.

Notes:
  - Assumes cross-bucket cleaned cases already exist (run preprocess first).
  - e5_large_v2 embedding uses all visible GPUs via SentenceTransformers multi_process.
  - InCaseLawBERT and InLegalBERT use single-GPU hf_encoder; built sequentially to avoid OOM.
  - Training assigns GPU[2] to e5, GPU[3] to incaselawbert, GPU[4] to inlegalbert.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --base-config)           BASE_CONFIG="$2";          shift 2 ;;
        --variants-config)       VARIANTS_CONFIG="$2";      shift 2 ;;
        --artifact-root)         ARTIFACT_ROOT="$2";        shift 2 ;;
        --mamba-prefix)          MAMBA_PREFIX="$2";         shift 2 ;;
        --cuda-visible-devices)  CUDA_DEVICES="$2";         shift 2 ;;
        --run-prefix)            RUN_PREFIX="$2";           shift 2 ;;
        --skip-e5-build)         SKIP_E5_BUILD=1;           shift   ;;
        --skip-incaselawbert-build) SKIP_INCASELAWBERT_BUILD=1; shift ;;
        --skip-inlegalbert-build)   SKIP_INLEGALBERT_BUILD=1;   shift ;;
        --skip-train)            SKIP_TRAIN=1;              shift   ;;
        --skip-summary)          SKIP_SUMMARY=1;            shift   ;;
        --sequential-train)      SEQUENTIAL_TRAIN=1;        shift   ;;
        --help|-h)               usage; exit 0              ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
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
    echo "Python interpreter not found: $PY_BIN" >&2
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
    echo "Cleaned case dir is missing. Run preprocess first." >&2
    echo "Expected cleaned_case_dir: $CLEANED_CASE_DIR" >&2
    exit 1
fi

echo "============================================================"
echo "  Cross-Bucket Legal Encoders Pipeline"
echo "  Python        : $PY_BIN"
echo "  Base Config   : $BASE_CONFIG"
echo "  Variants YAML : $VARIANTS_CONFIG"
echo "  Artifact Root : $ARTIFACT_ROOT"
echo "  Visible GPUs  : $CUDA_DEVICES"
echo "  Run Prefix    : $RUN_PREFIX"
echo "  Cleaned Cases : $CLEANED_CASE_DIR"
echo "  Variants      : incaselawbert, inlegalbert, e5_large_v2"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Generate variant configs
# ---------------------------------------------------------------------------
echo ">>> Step 0/5: generate variant configs"
"$PY_BIN" - "$BASE_CONFIG" "$VARIANTS_CONFIG" "$ARTIFACT_ROOT" "$RUN_PREFIX" <<'PY'
import json
import sys
from pathlib import Path
import yaml


def load_yaml(path):
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML at {path} must parse to a mapping.")
    return data


def dump_yaml(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def dump_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
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
selected_variants = ["incaselawbert", "inlegalbert", "e5_large_v2"]

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
    variant_cfg = deep_merge(base_cfg, override)
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

E5_CONFIG="$ARTIFACT_ROOT/configs/e5_large_v2.yaml"
INCASELAWBERT_CONFIG="$ARTIFACT_ROOT/configs/incaselawbert.yaml"
INLEGALBERT_CONFIG="$ARTIFACT_ROOT/configs/inlegalbert.yaml"

E5_RUN_NAME="${RUN_PREFIX}__e5_large_v2"
INCASELAWBERT_RUN_NAME="${RUN_PREFIX}__incaselawbert"
INLEGALBERT_RUN_NAME="${RUN_PREFIX}__inlegalbert"

# Split GPU list for build (single-GPU) and training assignments.
IFS=',' read -r -a GPU_IDS <<< "$CUDA_DEVICES"
NUM_GPUS="${#GPU_IDS[@]}"

# Assign build GPUs for the hf_encoder variants (index 0 of visible set).
HF_BUILD_GPU="${GPU_IDS[0]}"

# Assign training GPUs: spread across the available set.
E5_TRAIN_GPU="${GPU_IDS[0]}"
INCASELAWBERT_TRAIN_GPU="${GPU_IDS[0]}"
INLEGALBERT_TRAIN_GPU="${GPU_IDS[0]}"
if [ "$NUM_GPUS" -ge 2 ]; then INCASELAWBERT_TRAIN_GPU="${GPU_IDS[1]}"; fi
if [ "$NUM_GPUS" -ge 3 ]; then INLEGALBERT_TRAIN_GPU="${GPU_IDS[2]}"; fi
if [ "$NUM_GPUS" -ge 4 ]; then E5_TRAIN_GPU="${GPU_IDS[3]}"; fi

# ---------------------------------------------------------------------------
# Step 1: Build e5_large_v2 (all GPUs via sentence_transformers multi_process)
# ---------------------------------------------------------------------------
if [ "$SKIP_E5_BUILD" -eq 0 ]; then
    echo ">>> Step 1/5: build e5_large_v2 graph (all visible GPUs: $CUDA_DEVICES)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PY_BIN" "$BUILD_SCRIPT" --config "$E5_CONFIG"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 2: Build incaselawbert (single GPU — hf_encoder)
# ---------------------------------------------------------------------------
if [ "$SKIP_INCASELAWBERT_BUILD" -eq 0 ]; then
    echo ">>> Step 2/5: build incaselawbert graph (all GPUs via DataParallel: $CUDA_DEVICES)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PY_BIN" "$BUILD_SCRIPT" --config "$INCASELAWBERT_CONFIG"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 3: Build inlegalbert (all GPUs — hf_encoder with DataParallel)
# ---------------------------------------------------------------------------
if [ "$SKIP_INLEGALBERT_BUILD" -eq 0 ]; then
    echo ">>> Step 3/5: build inlegalbert graph (all GPUs via DataParallel: $CUDA_DEVICES)"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PY_BIN" "$BUILD_SCRIPT" --config "$INLEGALBERT_CONFIG"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 4: Train all three in parallel on separate GPUs
# ---------------------------------------------------------------------------
if [ "$SKIP_TRAIN" -eq 0 ]; then
    echo ">>> Step 4/5: train models"
    echo "  e5_large_v2     GPU: $E5_TRAIN_GPU"
    echo "  incaselawbert   GPU: $INCASELAWBERT_TRAIN_GPU"
    echo "  inlegalbert     GPU: $INLEGALBERT_TRAIN_GPU"
    echo ""

    if [ "$SEQUENTIAL_TRAIN" -eq 1 ]; then
        CUDA_VISIBLE_DEVICES="$E5_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$E5_CONFIG" \
            --run-name "$E5_RUN_NAME"

        CUDA_VISIBLE_DEVICES="$INCASELAWBERT_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$INCASELAWBERT_CONFIG" \
            --run-name "$INCASELAWBERT_RUN_NAME"

        CUDA_VISIBLE_DEVICES="$INLEGALBERT_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$INLEGALBERT_CONFIG" \
            --run-name "$INLEGALBERT_RUN_NAME"
    else
        CUDA_VISIBLE_DEVICES="$E5_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$E5_CONFIG" \
            --run-name "$E5_RUN_NAME" &
        E5_PID=$!

        CUDA_VISIBLE_DEVICES="$INCASELAWBERT_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$INCASELAWBERT_CONFIG" \
            --run-name "$INCASELAWBERT_RUN_NAME" &
        INCASELAWBERT_PID=$!

        CUDA_VISIBLE_DEVICES="$INLEGALBERT_TRAIN_GPU" "$PY_BIN" "$TRAIN_SCRIPT" \
            --config "$INLEGALBERT_CONFIG" \
            --run-name "$INLEGALBERT_RUN_NAME" &
        INLEGALBERT_PID=$!

        wait "$E5_PID"
        wait "$INCASELAWBERT_PID"
        wait "$INLEGALBERT_PID"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------
if [ "$SKIP_SUMMARY" -eq 0 ]; then
    echo ">>> Step 5/5: write comparison summary"
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
