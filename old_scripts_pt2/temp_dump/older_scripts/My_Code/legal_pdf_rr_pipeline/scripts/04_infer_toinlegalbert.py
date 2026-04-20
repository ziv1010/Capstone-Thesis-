#!/usr/bin/env python3
"""
04_infer_toinlegalbert.py
Run sentence-level rhetorical-role inference with the released LegalSeg
ToInLegalBERT checkpoint.

This stage validates the published checkpoint before inference. The public
`L-NLProc/LegalSeg_ToInLegalBERT` release is stored in PyTorch's old
directory format (`data.pkl` + `data/`). As of March 10, 2026, the published
repo exposes only 100 of the 225 tensor storages referenced by `data.pkl`.
Most importantly, `classifier.weight` and `classifier.bias` are missing.

The earlier pipeline silently fell back to a freshly initialized classifier
head, which produced misleading all-zero predictions. This version refuses to
run if the released checkpoint is missing the label head.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


LEGALSEG_REPO_URL = "https://github.com/ShubhamKumarNigam/LegalSeg.git"
LEGALSEG_HF_REPO = "L-NLProc/LegalSeg_ToInLegalBERT"
BASE_ENCODER_NAME = "law-ai/InLegalBERT"
NUM_LABELS = 7
MAX_SEGMENT_LENGTH = 130
MAX_DOCUMENT_LENGTH = 601
NUM_ENCODER_LAYERS = 2
TRANSFORMER_DROPOUT = 0.0
DEFAULT_EMBEDDER = "absolute"
STRICT_RELEASE_DATE = "March 10, 2026"

LABEL2ROLE = {
    0: "None",
    1: "Facts",
    2: "Issue",
    3: "Arguments of Petitioner",
    4: "Arguments of Respondent",
    5: "Reasoning",
    6: "Decision",
}

CRITICAL_CHECKPOINT_KEYS = (
    "classifier.weight",
    "classifier.bias",
)


class FastAbsoluteSinusoidalEmbedder:
    """Vectorized equivalent of LegalSeg's absolute sinusoidal embedder."""

    def __init__(self, max_document_length: int, embedding_dimension: int):
        self._embeddings = self._build_embeddings(max_document_length, embedding_dimension)

    @staticmethod
    def _build_embeddings(max_document_length: int, embedding_dimension: int) -> torch.Tensor:
        positions = torch.arange(1, max_document_length + 1, dtype=torch.float32).unsqueeze(1)
        dimensions = torch.arange(embedding_dimension, dtype=torch.float32).unsqueeze(0)
        angles = positions / torch.pow(10_000.0, 2 * dimensions / embedding_dimension)
        angles[:, 0::2] = torch.sin(angles[:, 0::2])
        angles[:, 1::2] = torch.cos(angles[:, 1::2])
        return angles

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self._embeddings.to(device=x.device, dtype=x.dtype)


class FastRelativeSinusoidalEmbedder:
    """Vectorized equivalent of LegalSeg's relative sinusoidal embedder."""

    def __init__(self, max_document_length: int, embedding_dimension: int):
        self._positions = torch.arange(
            1,
            max_document_length + 1,
            dtype=torch.float32,
        ).view(1, max_document_length, 1)
        self._divisors = torch.pow(
            10_000.0,
            2 * torch.arange(embedding_dimension, dtype=torch.float32) / embedding_dimension,
        ).view(1, 1, embedding_dimension)

    def __call__(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        document_lengths = mask.sum(dim=1).clamp(min=1).to(dtype=torch.float32).view(-1, 1, 1)
        angles = (1_000.0 * self._positions.to(x.device) / document_lengths) / self._divisors.to(x.device)
        angles[..., 0::2] = torch.sin(angles[..., 0::2])
        angles[..., 1::2] = torch.cos(angles[..., 1::2])
        return x + angles.to(dtype=x.dtype)


@dataclass
class StorageSpec:
    storage_type_name: str
    key: str
    location: str
    numel: int


@dataclass
class TensorSpec:
    storage: StorageSpec
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool


STORAGE_TYPE_TO_DTYPE = {
    "FloatStorage": (np.float32, torch.float32),
    "HalfStorage": (np.float16, torch.float16),
    "DoubleStorage": (np.float64, torch.float64),
    "LongStorage": (np.int64, torch.int64),
    "IntStorage": (np.int32, torch.int32),
    "ShortStorage": (np.int16, torch.int16),
    "ByteStorage": (np.uint8, torch.uint8),
    "CharStorage": (np.int8, torch.int8),
    "BoolStorage": (np.bool_, torch.bool),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_stale_outputs(output_dir: Path) -> None:
    for pattern in ("*_predictions.json", "all_predictions.csv", "toinlegalbert_loader_report.json"):
        for path in output_dir.glob(pattern):
            path.unlink()


def clone_official_legalseg_repo(models_dir: Path) -> Path:
    repo_root = models_dir / "official_LegalSeg"
    marker = repo_root / "code" / "ToInLegalBERT" / "rhetorical_roles_classification" / "transformer_over_bert.py"
    if marker.exists():
        return repo_root

    ensure_dir(models_dir)
    if repo_root.exists():
        raise RuntimeError(
            f"Expected LegalSeg repo at {repo_root}, but required ToInLegalBERT files are missing."
        )

    logger.info("Cloning official LegalSeg repo into %s", repo_root)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", LEGALSEG_REPO_URL, str(repo_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Failed to clone {LEGALSEG_REPO_URL}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return repo_root


def load_official_toinlegalbert_factory(repo_root: Path):
    rr_dir = repo_root / "code" / "ToInLegalBERT" / "rhetorical_roles_classification"
    if str(rr_dir) not in sys.path:
        sys.path.insert(0, str(rr_dir))
    from transformer_over_bert import AutoTransformerOverBERTForTokenClassification

    return AutoTransformerOverBERTForTokenClassification


def extract_zip_checkpoint(zip_path: Path, models_dir: Path) -> Path:
    target_root = ensure_dir(models_dir / "manual_checkpoint" / zip_path.stem)
    if not any(target_root.iterdir()):
        logger.info("Extracting manual checkpoint %s into %s", zip_path, target_root)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_root)
    checkpoint_dir = find_checkpoint_dir(target_root)
    if checkpoint_dir is None:
        raise RuntimeError(f"Could not find a ToInLegalBERT checkpoint inside {zip_path}")
    return checkpoint_dir


def find_checkpoint_dir(root: Path) -> Path | None:
    if (root / "data.pkl").is_file() and (root / "data").is_dir():
        return root

    for data_pkl in root.rglob("data.pkl"):
        candidate = data_pkl.parent
        if (candidate / "data").is_dir():
            return candidate
    return None


def ensure_checkpoint_dir(models_dir: Path, checkpoint_path: str | None) -> Path:
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser().resolve()
        if path.is_dir():
            checkpoint_dir = find_checkpoint_dir(path)
            if checkpoint_dir is None:
                raise RuntimeError(
                    f"Checkpoint directory {path} does not contain data.pkl + data/."
                )
            return checkpoint_dir
        if path.is_file() and path.suffix.lower() == ".zip":
            return extract_zip_checkpoint(path, models_dir)
        raise RuntimeError(
            f"Unsupported checkpoint path {path}. Provide a checkpoint directory or .zip archive."
        )

    download_root = ensure_dir(models_dir / "hf_download")
    checkpoint_dir = download_root / "ToInlegalBERT"
    if checkpoint_dir.is_dir() and (checkpoint_dir / "data.pkl").is_file():
        return checkpoint_dir

    logger.info("Downloading released checkpoint from %s", LEGALSEG_HF_REPO)
    snapshot_download(repo_id=LEGALSEG_HF_REPO, local_dir=str(download_root))
    checkpoint_dir = download_root / "ToInlegalBERT"
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"Expected checkpoint directory at {checkpoint_dir}")
    return checkpoint_dir


def build_embedder(embedder_name: str):
    if embedder_name == "absolute":
        return FastAbsoluteSinusoidalEmbedder(
            max_document_length=MAX_DOCUMENT_LENGTH,
            embedding_dimension=768,
        )
    if embedder_name == "relative":
        return FastRelativeSinusoidalEmbedder(
            max_document_length=MAX_DOCUMENT_LENGTH,
            embedding_dimension=768,
        )
    raise ValueError(f"Unsupported embedder '{embedder_name}'")


def storage_type_name(storage_type) -> str:
    return storage_type.__name__ if hasattr(storage_type, "__name__") else str(storage_type)


def rebuild_tensor_spec(
    storage: StorageSpec,
    storage_offset: int,
    size,
    stride,
    requires_grad: bool,
    backward_hooks,
) -> TensorSpec:
    del backward_hooks
    return TensorSpec(
        storage=storage,
        storage_offset=int(storage_offset),
        size=tuple(int(v) for v in size),
        stride=tuple(int(v) for v in stride),
        requires_grad=bool(requires_grad),
    )


class CheckpointSpecUnpickler(pickle.Unpickler):
    def persistent_load(self, saved_id):
        if saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id}")
        return StorageSpec(
            storage_type_name=storage_type_name(saved_id[1]),
            key=str(saved_id[2]),
            location=str(saved_id[3]),
            numel=int(saved_id[4]),
        )

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_spec
        if module == "torch" and hasattr(torch, name):
            return getattr(torch, name)
        raise pickle.UnpicklingError(f"Blocked class load: {module}.{name}")


def read_checkpoint_tensor_specs(checkpoint_dir: Path) -> OrderedDict[str, TensorSpec]:
    with open(checkpoint_dir / "data.pkl", "rb") as handle:
        state_dict = CheckpointSpecUnpickler(handle).load()
    if not isinstance(state_dict, OrderedDict):
        raise RuntimeError(f"Unexpected checkpoint root object: {type(state_dict)}")
    return state_dict


def load_storage_tensor(data_dir: Path, storage: StorageSpec) -> torch.Tensor:
    np_dtype, torch_dtype = STORAGE_TYPE_TO_DTYPE[storage.storage_type_name]
    file_path = data_dir / storage.key
    raw = np.fromfile(file_path, dtype=np_dtype)
    if raw.size != storage.numel:
        raise RuntimeError(
            f"Storage {file_path} has {raw.size} values, expected {storage.numel}."
        )
    return torch.from_numpy(raw.copy()).to(dtype=torch_dtype)


def materialize_available_tensors(
    specs: OrderedDict[str, TensorSpec],
    checkpoint_dir: Path,
) -> tuple[OrderedDict[str, torch.Tensor], list[str]]:
    data_dir = checkpoint_dir / "data"
    storage_cache: dict[str, torch.Tensor] = {}
    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    missing_keys: list[str] = []

    for key, spec in specs.items():
        storage_path = data_dir / spec.storage.key
        if not storage_path.exists():
            missing_keys.append(key)
            continue

        if spec.storage.key not in storage_cache:
            storage_cache[spec.storage.key] = load_storage_tensor(data_dir, spec.storage)

        storage_tensor = storage_cache[spec.storage.key]
        tensor = torch.as_strided(
            storage_tensor,
            size=spec.size,
            stride=spec.stride,
            storage_offset=spec.storage_offset,
        ).clone()
        if tensor.is_floating_point():
            tensor = tensor.requires_grad_(spec.requires_grad)
        tensors[key] = tensor

    return tensors, missing_keys


def infer_dim_feedforward(specs: OrderedDict[str, TensorSpec]) -> int:
    key = "bert._transformer.layers.0.linear1.weight"
    if key not in specs:
        raise RuntimeError(f"Checkpoint does not define required tensor {key}")
    return specs[key].size[0]


def build_model(
    models_dir: Path,
    device: str,
    embedder_name: str,
    dim_feedforward: int,
):
    repo_root = clone_official_legalseg_repo(models_dir)
    factory = load_official_toinlegalbert_factory(repo_root)
    embedder = build_embedder(embedder_name)
    transformer = torch.nn.Transformer(
        d_model=768,
        nhead=12,
        batch_first=True,
        dim_feedforward=dim_feedforward,
        activation="gelu",
        dropout=TRANSFORMER_DROPOUT,
        layer_norm_eps=1e-12,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=0,
    ).encoder
    model = factory(
        model_name=BASE_ENCODER_NAME,
        embedder=embedder,
        transformer=transformer,
        num_labels=NUM_LABELS,
        max_document_length=MAX_DOCUMENT_LENGTH,
        max_segment_length=MAX_SEGMENT_LENGTH,
        device=device,
    )
    return model, repo_root


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def write_loader_report(report_path: Path, report: dict) -> None:
    ensure_dir(report_path.parent)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def load_model(
    models_dir: Path,
    device: str,
    output_dir: Path,
    embedder_name: str,
    checkpoint_path: str | None,
):
    checkpoint_dir = ensure_checkpoint_dir(models_dir, checkpoint_path)
    specs = read_checkpoint_tensor_specs(checkpoint_dir)
    available_tensors, missing_checkpoint_keys = materialize_available_tensors(specs, checkpoint_dir)
    dim_feedforward = infer_dim_feedforward(specs)

    model, repo_root = build_model(
        models_dir=models_dir,
        device=device,
        embedder_name=embedder_name,
        dim_feedforward=dim_feedforward,
    )

    base_state = model.state_dict()
    merged_state = OrderedDict(base_state)
    shape_mismatches: list[dict[str, object]] = []
    loaded_keys: set[str] = set()

    for key, tensor in available_tensors.items():
        if key not in base_state:
            continue
        if tensor.shape != base_state[key].shape:
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(base_state[key].shape),
                }
            )
            continue
        merged_state[key] = tensor
        loaded_keys.add(key)

    base_fallback_keys = [key for key in base_state if key not in loaded_keys]
    critical_missing_keys = [key for key in CRITICAL_CHECKPOINT_KEYS if key in base_fallback_keys]

    report = {
        "checkpoint_dir": str(checkpoint_dir),
        "official_legalseg_repo": str(repo_root),
        "checkpoint_tensor_specs": len(specs),
        "checkpoint_available_tensors": len(available_tensors),
        "checkpoint_missing_tensors": len(missing_checkpoint_keys),
        "base_fallback_tensors": len(base_fallback_keys),
        "critical_missing_keys": critical_missing_keys,
        "shape_mismatches": shape_mismatches,
        "embedder": embedder_name,
        "dim_feedforward": dim_feedforward,
        "missing_checkpoint_keys": missing_checkpoint_keys,
        "base_fallback_keys": base_fallback_keys,
    }
    report_path = output_dir / "toinlegalbert_loader_report.json"
    write_loader_report(report_path, report)

    logger.info(
        "Checkpoint validation: %d/%d tensors available, %d fallback to base model",
        len(available_tensors),
        len(specs),
        len(base_fallback_keys),
    )

    if critical_missing_keys:
        missing_text = ", ".join(critical_missing_keys)
        raise RuntimeError(
            "Released ToInLegalBERT checkpoint is incomplete and cannot produce valid "
            "sentence labels. "
            f"As of {STRICT_RELEASE_DATE}, {LEGALSEG_HF_REPO} exposes only "
            f"{len(available_tensors)} of {len(specs)} tensors, and critical label-head "
            f"weights are missing: {missing_text}. "
            "The pipeline now stops here instead of falling back to a random classifier. "
            f"See {report_path} for the full loader report. "
            "If you obtain a complete checkpoint from the authors, rerun with "
            "--checkpoint_path pointing to that directory or zip."
        )

    model.load_state_dict(merged_state, strict=True)
    model.to(device)
    model.eval()
    return model


def tokenize_document(segments: list[str], tokenizer) -> torch.Tensor:
    encoded = tokenizer(
        segments,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEGMENT_LENGTH,
        return_tensors="pt",
    )["input_ids"]

    if encoded.size(0) < MAX_DOCUMENT_LENGTH:
        pad = torch.zeros(
            MAX_DOCUMENT_LENGTH - encoded.size(0),
            MAX_SEGMENT_LENGTH,
            dtype=torch.long,
        )
        encoded = torch.cat([encoded, pad], dim=0)
    else:
        encoded = encoded[:MAX_DOCUMENT_LENGTH]

    return encoded.unsqueeze(0)


def infer_document(
    model,
    tokenizer,
    segments: list[str],
    device: str,
) -> list[int]:
    if len(segments) <= MAX_DOCUMENT_LENGTH:
        input_ids = tokenize_document(segments, tokenizer).to(device)
        with torch.inference_mode():
            logits = model(input_ids).logits
        return logits.argmax(dim=-1).flatten().tolist()[: len(segments)]

    logger.info(
        "Document has %d segments (> %d); chunking for inference",
        len(segments),
        MAX_DOCUMENT_LENGTH,
    )
    overlap = 50
    stride = MAX_DOCUMENT_LENGTH - overlap
    predictions: list[int | None] = [None] * len(segments)

    for start in range(0, len(segments), stride):
        end = min(start + MAX_DOCUMENT_LENGTH, len(segments))
        chunk = segments[start:end]
        input_ids = tokenize_document(chunk, tokenizer).to(device)
        with torch.inference_mode():
            chunk_logits = model(input_ids).logits
        chunk_preds = chunk_logits.argmax(dim=-1).flatten().tolist()[: len(chunk)]

        for offset, pred in enumerate(chunk_preds):
            index = start + offset
            if predictions[index] is None:
                predictions[index] = pred

        if end >= len(segments):
            break

    return [int(pred) if pred is not None else 0 for pred in predictions]


def run(
    input_path: str,
    output_dir: str,
    models_dir: str,
    device: str | None = None,
    embedder: str = DEFAULT_EMBEDDER,
    checkpoint_path: str | None = None,
) -> list[str]:
    input_path = str(Path(input_path).resolve())
    output_dir_path = ensure_dir(Path(output_dir).resolve())
    models_dir_path = ensure_dir(Path(models_dir).resolve())
    remove_stale_outputs(output_dir_path)

    resolved_device = resolve_device(device)
    logger.info("Using device: %s", resolved_device)
    logger.info("Using embedder: %s", embedder)

    model = load_model(
        models_dir=models_dir_path,
        device=resolved_device,
        output_dir=output_dir_path,
        embedder_name=embedder,
        checkpoint_path=checkpoint_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_ENCODER_NAME)

    with open(input_path, "r", encoding="utf-8") as handle:
        documents = json.load(handle)

    logger.info("Running inference on %d document(s)", len(documents))

    output_paths: list[str] = []
    combined_rows: list[dict[str, object]] = []

    for document in documents:
        doc_id = document["doc_id"]
        segments = document["segments"]
        logger.info("Doc %s: %d segments", doc_id, len(segments))

        predictions = infer_document(
            model=model,
            tokenizer=tokenizer,
            segments=segments,
            device=resolved_device,
        )

        records = []
        for sentence_id, (text, label_id) in enumerate(zip(segments, predictions)):
            row = {
                "sentence_id": sentence_id,
                "sent_id": sentence_id,
                "text": text,
                "label_id": int(label_id),
                "label": LABEL2ROLE[int(label_id)],
            }
            records.append(row)
            combined_rows.append({"doc_id": doc_id, **row})

        pred_path = output_dir_path / f"{doc_id}_predictions.json"
        with open(pred_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"doc_id": doc_id, "predictions": records},
                handle,
                indent=2,
                ensure_ascii=False,
            )
        output_paths.append(str(pred_path))

    csv_path = output_dir_path / "all_predictions.csv"
    pd.DataFrame(combined_rows).to_csv(csv_path, index=False)
    output_paths.append(str(csv_path))
    logger.info("Saved prediction outputs to %s", output_dir_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ToInLegalBERT inference on segmented legal documents."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the combined inference_input.json file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for per-document predictions and all_predictions.csv.",
    )
    parser.add_argument(
        "--models_dir",
        required=True,
        help="Directory used for model downloads and local repo clones.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, for example cpu or cuda:0. Auto-detect if omitted.",
    )
    parser.add_argument(
        "--embedder",
        choices=("absolute", "relative"),
        default=DEFAULT_EMBEDDER,
        help="Positional embedder to attach to ToInLegalBERT. The released checkpoint does not serialize this choice.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional path to a complete checkpoint directory or zip provided outside the public HF release.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run(
        input_path=args.input_path,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        device=args.device,
        embedder=args.embedder,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
