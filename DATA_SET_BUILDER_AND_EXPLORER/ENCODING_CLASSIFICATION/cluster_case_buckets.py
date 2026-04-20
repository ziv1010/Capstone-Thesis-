#!/usr/bin/env python3
"""
Embed, cluster, and visualize legal case buckets.

This script reads case texts from the five *_text bucket folders, builds one
embedding per case, discovers cross-bucket clusters, and writes plots/CSVs that
make it easier to inspect overlap between buckets.

Default bucket folders:
  - family_matrimonial_text
  - land_property_text
  - motor_accidents_text
  - sexual_offences_text
  - financial_fraud_text

Main outputs:
  - cases_with_clusters.csv
  - case_embeddings.npy
  - cluster_bucket_counts.csv
  - cluster_bucket_proportions.csv
  - cluster_exemplars.csv
  - bucket_centroid_similarity.csv
  - bucket_neighbor_overlap_fraction.csv
  - bucket_neighbor_overlap_similarity.csv
  - top_cross_bucket_pairs.csv
  - run_report.json
  - figures/*.png

Example:
  python3 cluster_case_buckets.py --device cuda

Smoke test:
  python3 cluster_case_buckets.py \
      --max-files-per-bucket 200 \
      --model-name sentence-transformers/all-MiniLM-L6-v2 \
      --device cpu
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
    category=UserWarning,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except ImportError as exc:
    raise SystemExit(
        "Missing Python packages. Run this script inside the existing micromamba env, for example:\n"
        "  micromamba run -p "
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/old_scripts_pt2/GNN/.micromamba/gnn_case_star "
        "python "
        "/scratch/ziv_baretto/Thesis_Ziv/Capstone-Thesis-/DATA_SET_BUILDER_AND_EXPLORER/ENCODING_CLASSIFICATION/cluster_case_buckets.py\n"
        f"Original import error: {exc}"
    ) from exc

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration only
    faiss = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - keep script usable without tqdm
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "INPUT_DATA"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_BUCKET_DIRS = [
    "family_matrimonial_text",
    "land_property_text",
    "motor_accidents_text",
    "sexual_offences_text",
    "financial_fraud_text",
]
DEFAULT_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
DEFAULT_FASTER_MODEL = "BAAI/bge-base-en-v1.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed all bucket cases, cluster them, and visualize overlap."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory that contains the *_text bucket folders.",
    )
    parser.add_argument(
        "--bucket-dirs",
        nargs="+",
        default=DEFAULT_BUCKET_DIRS,
        help="Bucket directories to load, relative to --input-root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where embeddings, CSVs, and figures will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Sentence-Transformers compatible embedding model. "
            f"Default: {DEFAULT_MODEL_NAME}. Faster fallback: {DEFAULT_FASTER_MODEL}."
        ),
    )
    parser.add_argument(
        "--truncate-dim",
        type=int,
        default=512,
        help=(
            "Optional embedding truncation dimension. Defaults to 512, which works "
            "well with the default Matryoshka-capable model. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Embedding device, e.g. cuda, cuda:0, cpu. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--multi-gpu",
        choices=["auto", "off", "on"],
        default="auto",
        help=(
            "Embedding parallelism mode. 'auto' uses all visible CUDA devices when "
            "--device resolves to 'cuda'."
        ),
    )
    parser.add_argument(
        "--gpu-devices",
        nargs="+",
        default=None,
        help=(
            "Explicit devices for sentence-transformers multi-process encoding, "
            "for example: --gpu-devices cuda:0 cuda:1 cuda:2 cuda:3"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Chunk batch size passed to SentenceTransformer.encode().",
    )
    parser.add_argument(
        "--document-batch-size",
        type=int,
        default=128,
        help="Number of case files grouped before chunk embedding + pooling.",
    )
    parser.add_argument(
        "--multi-process-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional chunk size passed to SentenceTransformer.encode_multi_process. "
            "Leave unset to let the library choose."
        ),
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length exposed to the embedding model.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=220,
        help="Approximate chunk size per case, in whitespace-delimited words.",
    )
    parser.add_argument(
        "--chunk-overlap-words",
        type=int,
        default=40,
        help="Word overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--max-chunks-per-document",
        type=int,
        default=8,
        help="Cap chunks per case to keep full-corpus runs practical.",
    )
    parser.add_argument(
        "--max-files-per-bucket",
        type=int,
        default=None,
        help=(
            "Optional deterministic sample size per bucket for faster iteration. "
            "If omitted, the script uses all files."
        ),
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=None,
        help="Force a fixed number of clusters. If omitted, the script selects k automatically.",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=4,
        help="Minimum cluster count considered during automatic selection.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=14,
        help="Maximum cluster count considered during automatic selection.",
    )
    parser.add_argument(
        "--svd-dim",
        type=int,
        default=50,
        help="Intermediate SVD dimension used before clustering.",
    )
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=5000,
        help="Maximum number of points used while selecting cluster count.",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=20,
        help="Number of nearest neighbors used for overlap analysis.",
    )
    parser.add_argument(
        "--neighbor-analysis-sample-size",
        type=int,
        default=10000,
        help=(
            "If faiss is unavailable, large corpora fall back to a stratified sample "
            "of this size for the overlap stage."
        ),
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=250,
        help="How many strongest cross-bucket case pairs to save.",
    )
    parser.add_argument(
        "--cluster-exemplars",
        type=int,
        default=5,
        help="How many representative case files to save per discovered cluster.",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse output-dir/case_embeddings.npy and cases_metadata.csv if present.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for sampling, clustering, and plots.",
    )
    return parser.parse_args()


def infer_device(explicit_device: str | None) -> str:
    if explicit_device:
        return explicit_device

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def available_cuda_devices() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
    except Exception:
        pass
    return []


def resolve_multi_gpu_devices(
    device: str,
    multi_gpu_mode: str,
    gpu_devices: Sequence[str] | None,
) -> list[str]:
    if multi_gpu_mode == "off":
        return []

    if gpu_devices:
        unique_devices = list(dict.fromkeys(gpu_devices))
        return unique_devices if len(unique_devices) > 1 else []

    if not device.startswith("cuda"):
        return []

    cuda_devices = available_cuda_devices()
    if len(cuda_devices) <= 1:
        return []

    if device == "cuda":
        return cuda_devices

    return []


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"Indian Kanoon\s*-\s*https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"http[s]?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+No\.?\s*\d+\s*(?:of\s*\d+)?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDownloaded on\b.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)

    lines: list[str] = []
    previous = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) <= 1:
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def bucket_label_from_dir(bucket_dir: str) -> str:
    if bucket_dir.endswith("_text"):
        return bucket_dir[: -len("_text")]
    return bucket_dir


def stable_sample(
    files: Sequence[Path],
    max_files: int | None,
    rng: np.random.Generator,
) -> list[Path]:
    if max_files is None or len(files) <= max_files:
        return list(files)

    chosen = rng.choice(len(files), size=max_files, replace=False)
    chosen = np.sort(chosen)
    return [files[idx] for idx in chosen]


def stratified_sample_indices(
    labels: Sequence[str],
    max_items: int,
    random_state: int,
) -> np.ndarray:
    labels_array = np.asarray(labels)
    n_items = labels_array.shape[0]
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items, dtype=int)

    rng = np.random.default_rng(random_state)
    selected: list[int] = []
    remaining: list[int] = []

    unique_labels, counts = np.unique(labels_array, return_counts=True)
    allocations: dict[str, int] = {}
    total_allocated = 0
    for label, count in zip(unique_labels, counts):
        take = max(1, int(round(max_items * (int(count) / n_items))))
        take = min(take, int(count))
        allocations[str(label)] = take
        total_allocated += take

    if total_allocated > max_items:
        while total_allocated > max_items:
            largest_label = max(allocations, key=allocations.get)
            if allocations[largest_label] <= 1:
                break
            allocations[largest_label] -= 1
            total_allocated -= 1
    elif total_allocated < max_items:
        remainder = max_items - total_allocated
        by_size = sorted(
            ((str(label), int(count)) for label, count in zip(unique_labels, counts)),
            key=lambda item: item[1],
            reverse=True,
        )
        pointer = 0
        while remainder > 0 and by_size:
            label, count = by_size[pointer % len(by_size)]
            if allocations[label] < count:
                allocations[label] += 1
                remainder -= 1
            pointer += 1

    for label in unique_labels:
        label_str = str(label)
        label_idx = np.where(labels_array == label)[0]
        take = allocations[label_str]
        chosen = rng.choice(label_idx, size=take, replace=False)
        selected.extend(int(idx) for idx in chosen)

    selected = sorted(set(selected))
    if len(selected) > max_items:
        selected = sorted(rng.choice(selected, size=max_items, replace=False).tolist())
    elif len(selected) < max_items:
        selected_set = set(selected)
        remaining = [idx for idx in range(n_items) if idx not in selected_set]
        if remaining:
            extra_take = min(max_items - len(selected), len(remaining))
            extra = rng.choice(remaining, size=extra_take, replace=False)
            selected.extend(int(idx) for idx in extra)
            selected = sorted(selected)

    return np.asarray(selected, dtype=int)


def load_cases(
    input_root: Path,
    bucket_dirs: Sequence[str],
    max_files_per_bucket: int | None,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)

    for bucket_dir_name in bucket_dirs:
        bucket_dir = input_root / bucket_dir_name
        if not bucket_dir.exists():
            raise FileNotFoundError(f"Bucket directory not found: {bucket_dir}")

        bucket = bucket_label_from_dir(bucket_dir_name)
        files = sorted(bucket_dir.glob("*.txt"))
        files = stable_sample(files, max_files_per_bucket, rng)

        for path in tqdm(files, desc=f"Loading {bucket}", unit="file"):
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw_text)
            if not cleaned:
                cleaned = path.stem

            rows.append(
                {
                    "case_id": f"{bucket}::{path.stem}",
                    "bucket_dir": bucket_dir_name,
                    "bucket": bucket,
                    "filename": path.name,
                    "path": str(path),
                    "text": cleaned,
                    "text_preview": cleaned[:300],
                    "char_count": len(cleaned),
                    "word_count": len(cleaned.split()),
                }
            )

    if not rows:
        raise ValueError("No case files were loaded. Check the input directories.")

    return pd.DataFrame(rows)


def pick_chunk_indices(total_chunks: int, max_chunks: int | None) -> list[int]:
    if max_chunks is None or max_chunks <= 0 or total_chunks <= max_chunks:
        return list(range(total_chunks))

    if max_chunks == 1:
        return [0]

    raw = np.linspace(0, total_chunks - 1, num=max_chunks)
    indices: list[int] = []
    seen: set[int] = set()
    for value in raw:
        idx = int(round(float(value)))
        idx = min(max(idx, 0), total_chunks - 1)
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)

    if len(indices) < max_chunks:
        for idx in range(total_chunks):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) >= max_chunks:
                break
    return sorted(indices)


def split_text(
    text: str,
    chunk_words: int,
    overlap_words: int,
    max_chunks: int | None,
) -> list[str]:
    words = text.split()
    if not words:
        return [text]

    if len(words) <= chunk_words:
        return [" ".join(words)]

    step = max(chunk_words - overlap_words, 1)
    chunks = [
        " ".join(words[start : start + chunk_words])
        for start in range(0, len(words), step)
    ]
    keep = pick_chunk_indices(len(chunks), max_chunks)
    return [chunks[idx] for idx in keep]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def build_model(
    model_name: str,
    device: str,
    truncate_dim: int,
    max_seq_length: int,
    cache_dir: Path,
) -> SentenceTransformer:
    kwargs: dict[str, object] = {
        "device": device,
        "cache_folder": str(cache_dir),
    }
    if truncate_dim > 0:
        kwargs["truncate_dim"] = truncate_dim

    model = SentenceTransformer(model_name, **kwargs)
    model.max_seq_length = max_seq_length
    return model


def embed_cases(
    cases_df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
    document_batch_size: int,
    chunk_words: int,
    overlap_words: int,
    max_chunks_per_document: int,
    multi_process_pool: dict[str, Any] | None = None,
    multi_process_chunk_size: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    texts = cases_df["text"].tolist()
    doc_embeddings: list[np.ndarray] = []
    chunk_counts: list[int] = []

    for start in tqdm(
        range(0, len(texts), document_batch_size),
        desc="Embedding cases",
        unit="batch",
    ):
        batch_texts = texts[start : start + document_batch_size]
        flat_chunks: list[str] = []
        spans: list[tuple[int, int]] = []

        for text in batch_texts:
            chunks = split_text(
                text=text,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
                max_chunks=max_chunks_per_document,
            )
            spans.append((len(flat_chunks), len(chunks)))
            chunk_counts.append(len(chunks))
            flat_chunks.extend(chunks)

        if multi_process_pool is not None:
            chunk_embeddings = model.encode_multi_process(
                flat_chunks,
                pool=multi_process_pool,
                batch_size=batch_size,
                chunk_size=multi_process_chunk_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
        else:
            chunk_embeddings = model.encode(
                flat_chunks,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

        for offset, length in spans:
            pooled = chunk_embeddings[offset : offset + length].mean(axis=0, keepdims=True)
            pooled = normalize_rows(pooled)[0].astype(np.float32)
            doc_embeddings.append(pooled)

    embeddings = np.vstack(doc_embeddings).astype(np.float32)
    return embeddings, chunk_counts


def reduce_embeddings(
    embeddings: np.ndarray,
    svd_dim: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    n_components = min(svd_dim, embeddings.shape[0] - 1, embeddings.shape[1])
    if n_components < 2:
        raise ValueError("Need at least 2 samples to reduce embeddings.")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = svd.fit_transform(embeddings)

    projection = TruncatedSVD(n_components=2, random_state=random_state).fit_transform(embeddings)
    explained = float(np.sum(svd.explained_variance_ratio_))
    return reduced.astype(np.float32), projection.astype(np.float32), explained


def choose_cluster_count(
    reduced_embeddings: np.ndarray,
    num_clusters: int | None,
    min_clusters: int,
    max_clusters: int,
    sample_size: int,
    random_state: int,
) -> tuple[int, list[dict[str, float]]]:
    n_samples = reduced_embeddings.shape[0]
    if num_clusters is not None:
        return num_clusters, []

    upper = min(max_clusters, max(2, n_samples - 1))
    lower = min(min_clusters, upper)
    if lower < 2:
        lower = 2
    if upper < lower:
        upper = lower

    rng = np.random.default_rng(random_state)
    if n_samples > sample_size:
        sample_idx = rng.choice(n_samples, size=sample_size, replace=False)
        sample = reduced_embeddings[np.sort(sample_idx)]
    else:
        sample = reduced_embeddings

    scores: list[dict[str, float]] = []
    best_k = lower
    best_score = -1.0

    for k in tqdm(range(lower, upper + 1), desc="Selecting cluster count", unit="k"):
        if sample.shape[0] <= k:
            break
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=min(4096, sample.shape[0]),
            n_init=10,
        )
        labels = model.fit_predict(sample)
        unique = np.unique(labels)
        if unique.size < 2:
            score = -1.0
        else:
            score = float(silhouette_score(sample, labels))

        scores.append({"k": float(k), "silhouette": score})
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, scores


def cluster_cases(
    reduced_embeddings: np.ndarray,
    num_clusters: int,
    random_state: int,
) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=random_state,
        batch_size=min(4096, reduced_embeddings.shape[0]),
        n_init=10,
    )
    return model.fit_predict(reduced_embeddings).astype(int)


def centroid_similarity(cases_df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    bucket_vectors: list[np.ndarray] = []
    bucket_names: list[str] = []

    for bucket in sorted(cases_df["bucket"].unique()):
        mask = cases_df["bucket"] == bucket
        centroid = embeddings[mask.to_numpy()].mean(axis=0, keepdims=True)
        centroid = normalize_rows(centroid)[0]
        bucket_vectors.append(centroid)
        bucket_names.append(bucket)

    matrix = np.vstack(bucket_vectors)
    similarity = matrix @ matrix.T
    return pd.DataFrame(similarity, index=bucket_names, columns=bucket_names)


def search_neighbors(
    embeddings: np.ndarray,
    labels: Sequence[str],
    neighbor_k: int,
    sample_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    n_neighbors = min(neighbor_k + 1, embeddings.shape[0])

    if faiss is not None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        scores, indices = index.search(embeddings.astype(np.float32), n_neighbors)
        query_indices = np.arange(embeddings.shape[0], dtype=int)
        return scores, indices, query_indices, "faiss"

    if embeddings.shape[0] <= sample_size:
        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings, return_distance=True)
        scores = 1.0 - distances
        query_indices = np.arange(embeddings.shape[0], dtype=int)
        return scores.astype(np.float32), indices.astype(int), query_indices, "sklearn"

    sample_indices = stratified_sample_indices(
        labels=labels,
        max_items=sample_size,
        random_state=random_state,
    )
    sample_embeddings = embeddings[sample_indices]
    sample_neighbors = min(neighbor_k + 1, sample_embeddings.shape[0])

    nn = NearestNeighbors(
        n_neighbors=sample_neighbors,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    nn.fit(sample_embeddings)
    distances, local_indices = nn.kneighbors(sample_embeddings, return_distance=True)
    scores = 1.0 - distances
    indices = sample_indices[local_indices]
    backend = f"sklearn_sample[{sample_indices.shape[0]}]"
    return scores.astype(np.float32), indices.astype(int), sample_indices, backend


def neighbor_overlap_analysis(
    cases_df: pd.DataFrame,
    scores: np.ndarray,
    indices: np.ndarray,
    query_indices: np.ndarray,
    top_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buckets = cases_df["bucket"].to_numpy()
    case_ids = cases_df["case_id"].to_numpy()
    filenames = cases_df["filename"].to_numpy()

    overlap_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    overlap_scores: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    per_case_rows: list[dict[str, object]] = []
    best_pairs: dict[tuple[int, int], dict[str, object]] = {}

    for query_position, row_idx in enumerate(query_indices):
        source_bucket = str(buckets[row_idx])
        cross_bucket_match: dict[str, object] | None = None

        for rank in range(1, indices.shape[1]):
            neighbor_idx = int(indices[query_position, rank])
            if neighbor_idx < 0 or neighbor_idx == row_idx:
                continue

            target_bucket = str(buckets[neighbor_idx])
            similarity = float(scores[query_position, rank])
            if target_bucket == source_bucket:
                continue

            overlap_counts[(source_bucket, target_bucket)] += 1
            overlap_scores[(source_bucket, target_bucket)].append(similarity)

            pair_key = tuple(sorted((row_idx, neighbor_idx)))
            existing = best_pairs.get(pair_key)
            if existing is None or similarity > float(existing["similarity"]):
                best_pairs[pair_key] = {
                    "case_id_a": str(case_ids[pair_key[0]]),
                    "case_id_b": str(case_ids[pair_key[1]]),
                    "bucket_a": str(buckets[pair_key[0]]),
                    "bucket_b": str(buckets[pair_key[1]]),
                    "filename_a": str(filenames[pair_key[0]]),
                    "filename_b": str(filenames[pair_key[1]]),
                    "similarity": similarity,
                }

            cross_bucket_match = {
                "case_id": str(case_ids[row_idx]),
                "nearest_other_bucket": target_bucket,
                "nearest_other_bucket_case_id": str(case_ids[neighbor_idx]),
                "nearest_other_bucket_filename": str(filenames[neighbor_idx]),
                "nearest_other_bucket_similarity": similarity,
            }
            break

        if cross_bucket_match is None:
            cross_bucket_match = {
                "case_id": str(case_ids[row_idx]),
                "nearest_other_bucket": None,
                "nearest_other_bucket_case_id": None,
                "nearest_other_bucket_filename": None,
                "nearest_other_bucket_similarity": math.nan,
            }
        per_case_rows.append(cross_bucket_match)

    bucket_names = sorted(cases_df["bucket"].unique())
    fraction_matrix = pd.DataFrame(0.0, index=bucket_names, columns=bucket_names)
    similarity_matrix = pd.DataFrame(0.0, index=bucket_names, columns=bucket_names)

    cases_per_bucket = cases_df["bucket"].value_counts().to_dict()
    for source in bucket_names:
        for target in bucket_names:
            count = overlap_counts.get((source, target), 0)
            denom = max(int(cases_per_bucket.get(source, 0)), 1)
            fraction_matrix.loc[source, target] = count / denom

            sims = overlap_scores.get((source, target), [])
            similarity_matrix.loc[source, target] = float(np.mean(sims)) if sims else 0.0

    top_pairs_df = pd.DataFrame(best_pairs.values())
    if not top_pairs_df.empty:
        top_pairs_df = (
            top_pairs_df.sort_values("similarity", ascending=False)
            .head(top_pairs)
            .reset_index(drop=True)
        )

    per_case_df = pd.DataFrame(per_case_rows)
    return fraction_matrix, similarity_matrix, top_pairs_df, per_case_df


def cluster_bucket_tables(cases_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        cases_df.groupby(["cluster", "bucket"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    proportions = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    return counts, proportions


def cluster_exemplars(
    cases_df: pd.DataFrame,
    embeddings: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clusters = sorted(int(value) for value in cases_df["cluster"].unique())

    for cluster in clusters:
        cluster_mask = cases_df["cluster"].to_numpy() == cluster
        cluster_indices = np.where(cluster_mask)[0]
        centroid = embeddings[cluster_indices].mean(axis=0, keepdims=True)
        centroid = normalize_rows(centroid)[0]
        similarities = embeddings[cluster_indices] @ centroid
        order = np.argsort(-similarities)[:top_n]

        for rank, local_idx in enumerate(order, start=1):
            global_idx = int(cluster_indices[local_idx])
            row = cases_df.iloc[global_idx]
            rows.append(
                {
                    "cluster": cluster,
                    "rank": rank,
                    "bucket": row["bucket"],
                    "filename": row["filename"],
                    "path": row["path"],
                    "similarity_to_cluster_centroid": float(similarities[local_idx]),
                    "nearest_other_bucket": row.get("nearest_other_bucket"),
                    "nearest_other_bucket_similarity": row.get(
                        "nearest_other_bucket_similarity"
                    ),
                    "text_preview": row.get("text_preview"),
                }
            )

    return pd.DataFrame(rows)


def plot_scatter(
    frame: pd.DataFrame,
    label_column: str,
    title: str,
    output_path: Path,
) -> None:
    labels = frame[label_column].astype(str)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab20" if len(unique_labels) <= 20 else "gist_ncar")

    plt.figure(figsize=(12, 9))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            frame.loc[mask, "projection_x"],
            frame.loc[mask, "projection_y"],
            s=9,
            alpha=0.55,
            rasterized=True,
            label=label,
            color=cmap(idx % cmap.N),
        )

    plt.title(title)
    plt.xlabel("Projection 1")
    plt.ylabel("Projection 2")
    if len(unique_labels) <= 20:
        plt.legend(
            loc="best",
            fontsize=8,
            markerscale=2,
            frameon=False,
            ncol=2 if len(unique_labels) > 8 else 1,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    value_fmt: str = ".2f",
    cmap: str = "viridis",
) -> None:
    values = matrix.to_numpy(dtype=float)

    fig_width = max(8.0, matrix.shape[1] * 1.2)
    fig_height = max(6.0, matrix.shape[0] * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)

    if matrix.shape[0] * matrix.shape[1] <= 120:
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(
                    col,
                    row,
                    format(values[row, col], value_fmt),
                    ha="center",
                    va="center",
                    color="white" if values[row, col] < values.max() * 0.6 else "black",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_silhouette(scores: pd.DataFrame, output_path: Path) -> None:
    if scores.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(scores["k"], scores["silhouette"], marker="o", linewidth=1.5)
    plt.title("Cluster Count Selection")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    cases_df: pd.DataFrame,
    embeddings: np.ndarray,
    selected_k: int,
    svd_explained_variance: float,
    neighbor_backend: str,
    neighbor_query_count: int,
    multi_gpu_devices: Sequence[str],
) -> None:
    bucket_counts = {
        bucket: int(count)
        for bucket, count in cases_df["bucket"].value_counts().sort_index().items()
    }
    cluster_counts = {
        str(cluster): int(count)
        for cluster, count in cases_df["cluster"].value_counts().sort_index().items()
    }

    report = {
        "input_root": str(args.input_root),
        "bucket_dirs": list(args.bucket_dirs),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "truncate_dim": args.truncate_dim,
        "device": args.device,
        "multi_gpu_mode": args.multi_gpu,
        "multi_gpu_devices": list(multi_gpu_devices),
        "max_seq_length": args.max_seq_length,
        "documents": int(len(cases_df)),
        "embedding_dim": int(embeddings.shape[1]),
        "bucket_counts": bucket_counts,
        "selected_clusters": int(selected_k),
        "cluster_counts": cluster_counts,
        "svd_dim": int(min(args.svd_dim, embeddings.shape[1])),
        "svd_explained_variance_ratio_sum": svd_explained_variance,
        "neighbor_backend": neighbor_backend,
        "neighbor_query_count": int(neighbor_query_count),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.device = infer_device(args.device)
    multi_gpu_devices = resolve_multi_gpu_devices(
        device=args.device,
        multi_gpu_mode=args.multi_gpu,
        gpu_devices=args.gpu_devices,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_dir / "cases_metadata.csv"
    embeddings_path = args.output_dir / "case_embeddings.npy"

    if args.reuse_embeddings and metadata_path.exists() and embeddings_path.exists():
        cases_df = pd.read_csv(metadata_path)
        embeddings = np.load(embeddings_path)
    else:
        cases_df = load_cases(
            input_root=args.input_root,
            bucket_dirs=args.bucket_dirs,
            max_files_per_bucket=args.max_files_per_bucket,
            random_state=args.random_state,
        )
        model_device = "cpu" if multi_gpu_devices else args.device
        model = build_model(
            model_name=args.model_name,
            device=model_device,
            truncate_dim=args.truncate_dim,
            max_seq_length=args.max_seq_length,
            cache_dir=args.output_dir / "hf_cache",
        )
        multi_process_pool: dict[str, Any] | None = None
        try:
            if multi_gpu_devices:
                print(
                    "Starting multi-GPU embedding pool on: "
                    + ", ".join(multi_gpu_devices)
                )
                multi_process_pool = model.start_multi_process_pool(
                    target_devices=multi_gpu_devices
                )

            embeddings, chunk_counts = embed_cases(
                cases_df=cases_df,
                model=model,
                batch_size=args.batch_size,
                document_batch_size=args.document_batch_size,
                chunk_words=args.chunk_words,
                overlap_words=args.chunk_overlap_words,
                max_chunks_per_document=args.max_chunks_per_document,
                multi_process_pool=multi_process_pool,
                multi_process_chunk_size=args.multi_process_chunk_size,
            )
        finally:
            if multi_process_pool is not None:
                model.stop_multi_process_pool(multi_process_pool)
                for queue_name in ("input", "output"):
                    queue_obj = multi_process_pool.get(queue_name)
                    if queue_obj is None:
                        continue
                    try:
                        queue_obj.join_thread()
                    except Exception:
                        pass

        cases_df["chunk_count"] = chunk_counts
        metadata_df = cases_df.drop(columns=["text"])
        metadata_df.to_csv(metadata_path, index=False)
        np.save(embeddings_path, embeddings.astype(np.float32))

    reduced, projection, svd_explained = reduce_embeddings(
        embeddings=embeddings,
        svd_dim=args.svd_dim,
        random_state=args.random_state,
    )
    cases_df["projection_x"] = projection[:, 0]
    cases_df["projection_y"] = projection[:, 1]

    selected_k, silhouette_rows = choose_cluster_count(
        reduced_embeddings=reduced,
        num_clusters=args.num_clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        sample_size=args.silhouette_sample_size,
        random_state=args.random_state,
    )
    cases_df["cluster"] = cluster_cases(
        reduced_embeddings=reduced,
        num_clusters=selected_k,
        random_state=args.random_state,
    )

    centroid_similarity_df = centroid_similarity(cases_df, embeddings)
    neighbor_scores, neighbor_indices, neighbor_query_indices, neighbor_backend = search_neighbors(
        embeddings=embeddings,
        labels=cases_df["bucket"].tolist(),
        neighbor_k=args.neighbor_k,
        sample_size=args.neighbor_analysis_sample_size,
        random_state=args.random_state,
    )
    (
        neighbor_fraction_df,
        neighbor_similarity_df,
        top_pairs_df,
        per_case_overlap_df,
    ) = neighbor_overlap_analysis(
        cases_df=cases_df,
        scores=neighbor_scores,
        indices=neighbor_indices,
        query_indices=neighbor_query_indices,
        top_pairs=args.top_pairs,
    )

    cases_df = cases_df.merge(per_case_overlap_df, on="case_id", how="left")
    cluster_counts_df, cluster_proportions_df = cluster_bucket_tables(cases_df)
    cluster_exemplars_df = cluster_exemplars(
        cases_df=cases_df,
        embeddings=embeddings,
        top_n=args.cluster_exemplars,
    )
    silhouette_df = pd.DataFrame(silhouette_rows)

    cases_output_df = cases_df.drop(columns=["text"], errors="ignore")
    cases_output_df.to_csv(args.output_dir / "cases_with_clusters.csv", index=False)
    centroid_similarity_df.to_csv(args.output_dir / "bucket_centroid_similarity.csv")
    neighbor_fraction_df.to_csv(args.output_dir / "bucket_neighbor_overlap_fraction.csv")
    neighbor_similarity_df.to_csv(args.output_dir / "bucket_neighbor_overlap_similarity.csv")
    top_pairs_df.to_csv(args.output_dir / "top_cross_bucket_pairs.csv", index=False)
    cluster_counts_df.to_csv(args.output_dir / "cluster_bucket_counts.csv")
    cluster_proportions_df.to_csv(args.output_dir / "cluster_bucket_proportions.csv")
    cluster_exemplars_df.to_csv(args.output_dir / "cluster_exemplars.csv", index=False)
    silhouette_df.to_csv(args.output_dir / "cluster_selection_scores.csv", index=False)

    plot_scatter(
        cases_df,
        label_column="bucket",
        title="Case Embeddings Colored by Original Bucket",
        output_path=figures_dir / "scatter_by_bucket.png",
    )
    plot_scatter(
        cases_df,
        label_column="cluster",
        title="Case Embeddings Colored by Discovered Cluster",
        output_path=figures_dir / "scatter_by_cluster.png",
    )
    plot_heatmap(
        cluster_counts_df,
        title="Cluster x Bucket Counts",
        output_path=figures_dir / "cluster_bucket_counts.png",
        value_fmt=".0f",
        cmap="magma",
    )
    plot_heatmap(
        cluster_proportions_df,
        title="Cluster x Bucket Proportions",
        output_path=figures_dir / "cluster_bucket_proportions.png",
        value_fmt=".2f",
        cmap="viridis",
    )
    plot_heatmap(
        centroid_similarity_df,
        title="Bucket Centroid Cosine Similarity",
        output_path=figures_dir / "bucket_centroid_similarity.png",
        value_fmt=".2f",
        cmap="cividis",
    )
    plot_heatmap(
        neighbor_fraction_df,
        title="Nearest Cross-Bucket Overlap Fraction",
        output_path=figures_dir / "bucket_neighbor_overlap_fraction.png",
        value_fmt=".2f",
        cmap="plasma",
    )
    plot_heatmap(
        neighbor_similarity_df,
        title="Nearest Cross-Bucket Mean Similarity",
        output_path=figures_dir / "bucket_neighbor_overlap_similarity.png",
        value_fmt=".2f",
        cmap="plasma",
    )
    plot_silhouette(
        silhouette_df,
        output_path=figures_dir / "cluster_selection_scores.png",
    )

    write_report(
        output_path=args.output_dir / "run_report.json",
        args=args,
        cases_df=cases_df,
        embeddings=embeddings,
        selected_k=selected_k,
        svd_explained_variance=svd_explained,
        neighbor_backend=neighbor_backend,
        neighbor_query_count=neighbor_query_indices.shape[0],
        multi_gpu_devices=multi_gpu_devices,
    )

    print(f"Wrote outputs to: {args.output_dir}")
    print(f"Cases processed: {len(cases_df):,}")
    print(f"Embedding model: {args.model_name}")
    if multi_gpu_devices:
        print(f"Embedding GPUs: {', '.join(multi_gpu_devices)}")
    else:
        print(f"Embedding device: {args.device}")
    print(f"Selected clusters: {selected_k}")
    print(f"Neighbor backend: {neighbor_backend}")


if __name__ == "__main__":
    main()
