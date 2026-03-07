from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as exc:
        raise RuntimeError(
            "faiss is not available. Install faiss-cpu (or faiss-gpu) to run RAG."
        ) from exc


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return (vectors / norms).astype(np.float32)


def build_faiss_index(
    embeddings: np.ndarray,
    case_ids: list[str],
    metadata: list[dict[str, Any]],
    index_type: str = "flat_ip",
) -> tuple[Any, list[str], list[dict[str, Any]]]:
    faiss = _import_faiss()

    emb = l2_normalize(embeddings)
    dim = emb.shape[1]

    if index_type != "flat_ip":
        raise ValueError(f"Unsupported FAISS index type: {index_type}")

    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    return index, case_ids, metadata


def save_faiss_artifacts(
    index: Any,
    index_path: str | Path,
    case_ids: list[str],
    metadata: list[dict[str, Any]],
    case_ids_path: str | Path,
    metadata_path: str | Path,
) -> None:
    faiss = _import_faiss()
    ipath = Path(index_path)
    ipath.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(ipath))

    cpath = Path(case_ids_path)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(case_ids, ensure_ascii=False, indent=2), encoding="utf-8")

    mpath = Path(metadata_path)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def save_faiss_index(index: Any, path: str | Path) -> None:
    faiss = _import_faiss()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p))


def load_faiss_index(path: str | Path) -> Any:
    faiss = _import_faiss()
    return faiss.read_index(str(Path(path)))


def load_faiss_artifacts(
    index_path: str | Path,
    case_ids_path: str | Path,
    metadata_path: str | Path,
) -> tuple[Any, list[str], list[dict[str, Any]]]:
    faiss = _import_faiss()
    index = faiss.read_index(str(Path(index_path)))
    case_ids = json.loads(Path(case_ids_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return index, [str(x) for x in case_ids], metadata


def search_faiss(index: Any, query_vec: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    q = query_vec.astype(np.float32)
    if q.ndim == 1:
        q = q[None, :]
    distances, indices = index.search(q, int(top_k))
    return distances[0], indices[0]
