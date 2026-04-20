#!/usr/bin/env python3
"""
04_infer_hier_bilstm_crf.py
Run sentence-level rhetorical-role inference with the released LegalSeg
Hier_BiLSTM-CRF checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


LEGALSEG_REPO_URL = "https://github.com/ShubhamKumarNigam/LegalSeg.git"
HF_REPO_ID = "L-NLProc/LegalSeg_Hier_BiLSTM-CRF"
HF_MODEL_SUBDIR = "Hier_BiLSTM-CRF"
MY_CODE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FINETUNED_DIR = (
    MY_CODE_DIR
    / "BENCHMARKING_OPENNYAI"
    / "models"
    / "LegalSeg_Hier_BiLSTM_CRF_finetuned"
)
DEFAULT_FINETUNED_CHECKPOINT = DEFAULT_FINETUNED_DIR / "model_state_best.tar"
DEFAULT_FINETUNED_ASSETS_DIR = DEFAULT_FINETUNED_DIR / "hf_download" / HF_MODEL_SUBDIR

ROLE_TO_LABEL_ID = {
    "None": 0,
    "Facts": 1,
    "Issue": 2,
    "Arguments of Petitioner": 3,
    "Arguments of Respondent": 4,
    "Reasoning": 5,
    "Decision": 6,
}

SPECIAL_TAGS = {"<pad>", "<start>", "<end>"}
PUNCT_TRANSLATOR = str.maketrans(string.punctuation, " " * len(string.punctuation))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_stale_outputs(output_dir: Path) -> None:
    for pattern in ("*_predictions.json", "all_predictions.csv", "hier_bilstm_crf_report.json"):
        for path in output_dir.glob(pattern):
            path.unlink()


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def clone_official_legalseg_repo(models_dir: Path) -> Path:
    repo_root = models_dir / "official_LegalSeg"
    marker = repo_root / "code" / "Hier_BiLSTM CRF" / "model" / "Hier_BiLSTM_CRF.py"
    if marker.exists():
        return repo_root

    ensure_dir(models_dir)
    if repo_root.exists():
        raise RuntimeError(
            f"Expected LegalSeg repo at {repo_root}, but required Hier_BiLSTM-CRF files are missing."
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


def load_official_hier_symbols(repo_root: Path):
    code_root = repo_root / "code" / "Hier_BiLSTM CRF"
    model_root = code_root / "model"
    for path in (code_root, model_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from model.Hier_BiLSTM_CRF import Hier_LSTM_CRF_Classifier

    return Hier_LSTM_CRF_Classifier


def ensure_checkpoint_dir(models_dir: Path) -> Path:
    download_root = ensure_dir(models_dir / "hf_download")
    checkpoint_dir = download_root / HF_MODEL_SUBDIR
    required = (
        checkpoint_dir / "model_state4.tar",
        checkpoint_dir / "word2idx.json",
        checkpoint_dir / "tag2idx.json",
    )
    if all(path.is_file() for path in required):
        return checkpoint_dir

    logger.info("Downloading Hier_BiLSTM-CRF checkpoint from %s", HF_REPO_ID)
    snapshot_download(repo_id=HF_REPO_ID, local_dir=str(download_root))

    if not all(path.is_file() for path in required):
        raise RuntimeError(
            f"Expected Hier_BiLSTM-CRF files under {checkpoint_dir}, but download is incomplete."
    )
    return checkpoint_dir


def resolve_checkpoint_bundle(
    models_dir: Path,
    checkpoint_path: str | None = None,
    assets_dir: str | None = None,
) -> tuple[Path, Path, str]:
    released_assets_dir = ensure_checkpoint_dir(models_dir)

    if checkpoint_path:
        resolved_checkpoint = Path(checkpoint_path).resolve()
        if not resolved_checkpoint.is_file():
            raise RuntimeError(f"Hier_BiLSTM-CRF checkpoint file not found: {resolved_checkpoint}")
        resolved_assets_dir = (
            Path(assets_dir).resolve() if assets_dir else released_assets_dir
        )
        if not (resolved_assets_dir / "word2idx.json").is_file() or not (resolved_assets_dir / "tag2idx.json").is_file():
            raise RuntimeError(
                f"Hier_BiLSTM-CRF assets dir must contain word2idx.json and tag2idx.json: {resolved_assets_dir}"
            )
        return resolved_checkpoint, resolved_assets_dir, "override"

    if DEFAULT_FINETUNED_CHECKPOINT.is_file():
        if not (DEFAULT_FINETUNED_ASSETS_DIR / "word2idx.json").is_file() or not (DEFAULT_FINETUNED_ASSETS_DIR / "tag2idx.json").is_file():
            raise RuntimeError(
                f"Default fine-tuned Hier_BiLSTM-CRF checkpoint exists but its assets dir is incomplete: {DEFAULT_FINETUNED_ASSETS_DIR}"
            )
        logger.info("Using default fine-tuned Hier_BiLSTM-CRF checkpoint at %s", DEFAULT_FINETUNED_CHECKPOINT)
        return DEFAULT_FINETUNED_CHECKPOINT, DEFAULT_FINETUNED_ASSETS_DIR, "default_finetuned"

    return released_assets_dir / "model_state4.tar", released_assets_dir, "released_hf"


def normalize_text_for_model(text: str, word2idx: dict[str, int]) -> list[int]:
    tokens = text.lower().translate(PUNCT_TRANSLATOR).split()
    indices = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
    # Preserve sentence alignment even for punctuation-only segments.
    return indices or [word2idx["<unk>"]]


def prepare_documents(
    inference_docs: list[dict[str, object]],
    word2idx: dict[str, int],
) -> list[list[list[int]]]:
    encoded_docs: list[list[list[int]]] = []
    for doc in inference_docs:
        segments = doc["segments"]
        encoded_docs.append([normalize_text_for_model(segment, word2idx) for segment in segments])
    return encoded_docs


def infer_batches(model, documents: list[list[list[int]]], batch_size: int) -> list[list[int]]:
    predictions: list[list[int]] = []
    with torch.no_grad():
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            predictions.extend(model(batch_docs))
    return predictions


def write_report(report_path: Path, report: dict) -> None:
    ensure_dir(report_path.parent)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def load_model(
    models_dir: Path,
    device: str,
    checkpoint_path: str | None = None,
    assets_dir: str | None = None,
):
    repo_root = clone_official_legalseg_repo(models_dir)
    checkpoint_file, checkpoint_dir, checkpoint_source = resolve_checkpoint_bundle(
        models_dir=models_dir,
        checkpoint_path=checkpoint_path,
        assets_dir=assets_dir,
    )
    HierClassifier = load_official_hier_symbols(repo_root)

    with open(checkpoint_dir / "word2idx.json", "r", encoding="utf-8") as handle:
        word2idx = json.load(handle)
    with open(checkpoint_dir / "tag2idx.json", "r", encoding="utf-8") as handle:
        tag2idx = json.load(handle)

    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    model = HierClassifier(
        len(tag2idx),
        200,
        tag2idx["<start>"],
        tag2idx["<end>"],
        tag2idx["<pad>"],
        vocab_size=len(word2idx),
        word_emb_dim=100,
        pretrained=False,
        device=device,
    ).to(device)

    # The released LegalSeg implementation leaves these submodule device fields
    # at their constructor defaults. Override them explicitly for CPU support.
    if model.sent_encoder is not None:
        model.sent_encoder.device = device
    model.emitter.device = device

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Hier_BiLSTM-CRF checkpoint mismatch. missing={missing_keys}, unexpected={unexpected_keys}"
        )

    model.eval()
    return model, word2idx, tag2idx, checkpoint, checkpoint_file, checkpoint_dir, checkpoint_source, repo_root


def run(
    input_path: str,
    output_dir: str,
    models_dir: str,
    device: str | None = None,
    batch_size: int = 8,
    checkpoint_path: str | None = None,
    assets_dir: str | None = None,
) -> list[str]:
    input_path = str(Path(input_path).resolve())
    output_dir_path = ensure_dir(Path(output_dir).resolve())
    models_dir_path = ensure_dir(Path(models_dir).resolve())
    remove_stale_outputs(output_dir_path)

    resolved_device = resolve_device(device)
    logger.info("Using device: %s", resolved_device)

    model, word2idx, tag2idx, checkpoint, checkpoint_file, checkpoint_dir, checkpoint_source, repo_root = load_model(
        models_dir=models_dir_path,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        assets_dir=assets_dir,
    )

    idx2tag = {value: key for key, value in tag2idx.items()}
    model_tags = [idx2tag[idx] for idx in sorted(idx2tag) if idx2tag[idx] not in SPECIAL_TAGS]

    report = {
        "hf_repo_id": HF_REPO_ID,
        "checkpoint_source": checkpoint_source,
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_dir": str(checkpoint_dir),
        "official_legalseg_repo": str(repo_root),
        "device": resolved_device,
        "batch_size": batch_size,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_f1": checkpoint.get("best_f1"),
        "model_name": checkpoint.get("name"),
        "model_tags": model_tags,
        "vocab_size": len(word2idx),
        "tag2idx": tag2idx,
    }
    report_path = output_dir_path / "hier_bilstm_crf_report.json"
    write_report(report_path, report)

    with open(input_path, "r", encoding="utf-8") as handle:
        inference_docs = json.load(handle)

    encoded_docs = prepare_documents(inference_docs, word2idx)
    raw_predictions = infer_batches(model, encoded_docs, batch_size=batch_size)

    output_paths: list[str] = [str(report_path)]
    combined_rows: list[dict[str, object]] = []

    for doc, pred_indices in zip(inference_docs, raw_predictions):
        doc_id = doc["doc_id"]
        segments = doc["segments"]

        if len(pred_indices) != len(segments):
            raise RuntimeError(
                f"Prediction length mismatch for {doc_id}: got {len(pred_indices)}, expected {len(segments)}"
            )

        records = []
        for sentence_id, (text, pred_idx) in enumerate(zip(segments, pred_indices)):
            role_name = idx2tag[pred_idx]
            if role_name in SPECIAL_TAGS:
                raise RuntimeError(f"Model predicted special tag {role_name} for {doc_id}")

            label_id = ROLE_TO_LABEL_ID[role_name]
            row = {
                "sentence_id": sentence_id,
                "sent_id": sentence_id,
                "text": text,
                "label_id": label_id,
                "label": role_name,
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

        distribution = Counter(record["label"] for record in records)
        logger.info("Doc %s label distribution: %s", doc_id, dict(sorted(distribution.items())))

    csv_path = output_dir_path / "all_predictions.csv"
    pd.DataFrame(combined_rows).to_csv(csv_path, index=False)
    output_paths.append(str(csv_path))
    logger.info("Saved prediction outputs to %s", output_dir_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Hier_BiLSTM-CRF inference on segmented legal documents."
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
        "--batch_size",
        type=int,
        default=8,
        help="Number of documents per inference batch.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Optional path to a Hier_BiLSTM-CRF checkpoint file. Defaults to the local fine-tuned OpenNyAI checkpoint if present.",
    )
    parser.add_argument(
        "--assets_dir",
        default=None,
        help="Optional directory containing word2idx.json and tag2idx.json for the selected checkpoint.",
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
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        assets_dir=args.assets_dir,
    )


if __name__ == "__main__":
    main()
