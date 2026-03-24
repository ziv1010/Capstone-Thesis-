from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.utils.io import ensure_dir


class TextEncoderProtocol(Protocol):
    output_dim: int

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        ...


@dataclass
class SentenceTransformerEncoder:
    model_name: str
    device: str | None = None
    output_dim: int | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.output_dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)


@dataclass
class HFEncoder:
    model_name: str
    pooling: str = "mean"
    device: str = "cpu"
    output_dim: int | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.output_dim = int(getattr(self.model.config, "hidden_size"))

    def _mean_pool(self, hidden: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        torch = self.torch
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                model_out = self.model(**encoded)
                if self.pooling == "cls":
                    pooled = model_out.last_hidden_state[:, 0]
                else:
                    pooled = self._mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
                outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, int(self.output_dim or 1)), dtype=np.float32)


@dataclass
class HashingEncoder:
    dim: int = 384
    output_dim: int = 384

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        del batch_size, max_length
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = (text or "").lower().split()
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row_idx, bucket] += sign
            norm = float(np.linalg.norm(matrix[row_idx]))
            if norm > 0:
                matrix[row_idx] /= norm
        return matrix


def create_text_encoder(cfg: dict[str, Any]) -> TextEncoderProtocol:
    backend = str(cfg.get("backend", "sentence_transformers"))
    if backend == "sentence_transformers":
        return SentenceTransformerEncoder(
            model_name=str(cfg.get("model_name")),
            device=cfg.get("device"),
        )
    if backend == "hf_encoder":
        import torch

        device = str(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        return HFEncoder(
            model_name=str(cfg.get("model_name")),
            pooling=str(cfg.get("pooling", "mean")),
            device=device,
        )
    if backend == "hashing":
        dim = int(cfg.get("dim", 384))
        return HashingEncoder(dim=dim, output_dim=dim)
    raise ValueError(f"Unsupported text encoder backend: {backend}")


def _cache_key(ids: list[str], texts: list[str], cfg: dict[str, Any], namespace: str) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for field in ("backend", "model_name", "pooling", "max_length", "dim"):
        digest.update(str(cfg.get(field, "")).encode("utf-8"))
    for item_id, text in zip(ids, texts):
        digest.update(item_id.encode("utf-8"))
        digest.update(str(len(text)).encode("utf-8"))
    return digest.hexdigest()[:20]


def load_or_compute_embeddings(
    ids: list[str],
    texts: list[str],
    encoder_cfg: dict[str, Any],
    cache_dir: str | Path,
    namespace: str,
    logger: Any | None = None,
) -> tuple[np.ndarray, int]:
    cache_root = ensure_dir(cache_dir)
    cache_path = cache_root / f"{namespace}_{_cache_key(ids, texts, encoder_cfg, namespace)}.npz"
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["ids"].astype(str).tolist()
        if cached_ids == [str(item) for item in ids]:
            embeddings = cached["embeddings"].astype(np.float32)
            if logger is not None:
                logger.info("Loaded embedding cache %s shape=%s", cache_path, embeddings.shape)
            return embeddings, int(embeddings.shape[1])

    encoder = create_text_encoder(encoder_cfg)
    embeddings = encoder.encode(
        texts=texts,
        batch_size=int(encoder_cfg.get("batch_size", 16)),
        max_length=int(encoder_cfg.get("max_length", 512)),
    ).astype(np.float32)
    np.savez_compressed(cache_path, ids=np.array(ids), embeddings=embeddings)
    if logger is not None:
        logger.info("Saved embedding cache %s shape=%s", cache_path, embeddings.shape)
    return embeddings, int(embeddings.shape[1])
