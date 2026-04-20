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
    show_progress_bar: bool = True
    multi_process: bool = False
    multi_process_devices: list[str] | None = None
    chunk_size: int | None = None
    precision: str = "float32"
    output_dim: int | None = None
    # Hierarchical encoding: texts longer than this many characters are split into
    # overlapping chunks, each encoded independently, then mean-pooled.
    # Set to 0 or None to disable (default — preserves original behaviour).
    hierarchical_chunk_chars: int = 0
    hierarchical_chunk_overlap_chars: int = 200

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.output_dim = int(self.model.get_sentence_embedding_dimension())

    def _resolve_target_devices(self) -> list[str]:
        if self.multi_process_devices:
            return [str(device) for device in self.multi_process_devices]

        import torch

        if not torch.cuda.is_available():
            return []
        return [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split a long text into overlapping character-level chunks."""
        step = self.hierarchical_chunk_chars - self.hierarchical_chunk_overlap_chars
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.hierarchical_chunk_chars
            # Try to split on a sentence boundary (". ") within the last 200 chars
            if end < len(text):
                boundary = text.rfind(". ", end - 200, end)
                if boundary != -1:
                    end = boundary + 1
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start += step
        return [c for c in chunks if c]

    def _encode_batch(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode a flat list of texts with the underlying model."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            precision=self.precision,
            normalize_embeddings=False,
        ).astype(np.float32)

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        del max_length

        use_hierarchical = bool(self.hierarchical_chunk_chars and self.hierarchical_chunk_chars > 0)

        # ── Identify which texts need chunking ────────────────
        if use_hierarchical:
            needs_chunking = [len(t) > self.hierarchical_chunk_chars for t in texts]
            n_long = sum(needs_chunking)
            if n_long > 0 and self.show_progress_bar:
                print(f"[hierarchical] {n_long}/{len(texts)} texts exceed "
                      f"{self.hierarchical_chunk_chars} chars — chunking and pooling")
        else:
            needs_chunking = [False] * len(texts)

        # ── Fast path: no hierarchical encoding needed ────────
        if not any(needs_chunking):
            if self.multi_process:
                target_devices = self._resolve_target_devices()
                if len(target_devices) > 1:
                    pool = self.model.start_multi_process_pool(target_devices=target_devices)
                    try:
                        embeddings = self.model.encode_multi_process(
                            texts,
                            pool,
                            batch_size=batch_size,
                            chunk_size=self.chunk_size,
                            show_progress_bar=self.show_progress_bar,
                            precision=self.precision,
                            normalize_embeddings=False,
                        )
                    finally:
                        self.model.stop_multi_process_pool(pool)
                    return embeddings.astype(np.float32)

            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=False,
            ).astype(np.float32)

        # ── Mixed path: batch-encode short texts, chunk long ones ──
        assert self.output_dim is not None
        result = np.zeros((len(texts), self.output_dim), dtype=np.float32)

        # Build the pool once if multi_process is enabled — shared by both passes
        pool = None
        target_devices: list[str] = []
        if self.multi_process:
            target_devices = self._resolve_target_devices()
            if len(target_devices) > 1:
                pool = self.model.start_multi_process_pool(target_devices=target_devices)

        def _encode_flat(flat_texts: list[str]) -> np.ndarray:
            if pool is not None:
                return self.model.encode_multi_process(
                    flat_texts, pool,
                    batch_size=batch_size,
                    show_progress_bar=self.show_progress_bar,
                    precision=self.precision,
                    normalize_embeddings=False,
                ).astype(np.float32)
            return self.model.encode(
                flat_texts,
                batch_size=batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_numpy=True,
                precision=self.precision,
                normalize_embeddings=False,
            ).astype(np.float32)

        try:
            # Short texts: one batched encode across all GPUs
            short_indices = [i for i, flag in enumerate(needs_chunking) if not flag]
            if short_indices:
                short_embs = _encode_flat([texts[i] for i in short_indices])
                for out_idx, emb in zip(short_indices, short_embs):
                    result[out_idx] = emb

            # Long texts: flatten all chunks into one batched encode, then reassemble
            long_indices = [i for i, flag in enumerate(needs_chunking) if flag]
            if long_indices:
                all_chunks: list[str] = []
                chunk_counts: list[int] = []
                for i in long_indices:
                    chunks = self._split_into_chunks(texts[i])
                    all_chunks.extend(chunks)
                    chunk_counts.append(len(chunks))
                all_chunk_embs = _encode_flat(all_chunks)
                offset = 0
                for out_idx, n_chunks in zip(long_indices, chunk_counts):
                    result[out_idx] = all_chunk_embs[offset:offset + n_chunks].mean(axis=0)
                    offset += n_chunks
        finally:
            if pool is not None:
                self.model.stop_multi_process_pool(pool)

        return result


@dataclass
class HFEncoder:
    model_name: str
    pooling: str = "mean"
    device: str = "cpu"
    data_parallel: bool = True
    show_progress_bar: bool = True
    output_dim: int | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        base_model = AutoModel.from_pretrained(self.model_name)
        base_model.to(self.device)
        base_model.eval()
        self.output_dim = int(getattr(base_model.config, "hidden_size"))

        # Wrap with DataParallel when multiple GPUs are visible and device is cuda.
        if (
            self.data_parallel
            and self.device.startswith("cuda")
            and torch.cuda.device_count() > 1
        ):
            self.model = torch.nn.DataParallel(base_model)
        else:
            self.model = base_model

    def _mean_pool(self, hidden: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        from tqdm.auto import tqdm

        torch = self.torch
        outputs: list[np.ndarray] = []
        iterator = range(0, len(texts), batch_size)
        if self.show_progress_bar:
            iterator = tqdm(iterator, desc=f"Encoding {self.model_name}", unit="batch")
        for start in iterator:
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
                # DataParallel gathers outputs onto self.device; attention_mask is
                # already there, so mean-pool works without any device adjustment.
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
    show_progress_bar: bool = True

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        from tqdm.auto import tqdm

        del batch_size, max_length
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        iterator = enumerate(texts)
        if self.show_progress_bar:
            iterator = tqdm(iterator, total=len(texts), desc="Encoding hashing", unit="text")
        for row_idx, text in iterator:
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
            show_progress_bar=bool(cfg.get("show_progress_bar", True)),
            multi_process=bool(cfg.get("multi_process", False)),
            multi_process_devices=cfg.get("multi_process_devices"),
            chunk_size=cfg.get("chunk_size"),
            precision=str(cfg.get("precision", "float32")),
            hierarchical_chunk_chars=int(cfg.get("hierarchical_chunk_chars", 0)),
            hierarchical_chunk_overlap_chars=int(cfg.get("hierarchical_chunk_overlap_chars", 200)),
        )
    if backend == "hf_encoder":
        import torch

        device = str(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        return HFEncoder(
            model_name=str(cfg.get("model_name")),
            pooling=str(cfg.get("pooling", "mean")),
            device=device,
            data_parallel=bool(cfg.get("data_parallel", True)),
            show_progress_bar=bool(cfg.get("show_progress_bar", True)),
        )
    if backend == "hashing":
        dim = int(cfg.get("dim", 384))
        return HashingEncoder(
            dim=dim,
            output_dim=dim,
            show_progress_bar=bool(cfg.get("show_progress_bar", True)),
        )
    raise ValueError(f"Unsupported text encoder backend: {backend}")


def _cache_key(ids: list[str], texts: list[str], cfg: dict[str, Any], namespace: str) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for field in ("backend", "model_name", "pooling", "max_length", "dim",
                  "hierarchical_chunk_chars", "hierarchical_chunk_overlap_chars"):
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
    if logger is not None:
        logger.info(
            "Computing embeddings with backend=%s model=%s for %d texts",
            encoder_cfg.get("backend", ""),
            encoder_cfg.get("model_name", ""),
            len(texts),
        )
    embeddings = encoder.encode(
        texts=texts,
        batch_size=int(encoder_cfg.get("batch_size", 16)),
        max_length=int(encoder_cfg.get("max_length", 512)),
    ).astype(np.float32)
    np.savez_compressed(cache_path, ids=np.array(ids), embeddings=embeddings)
    if logger is not None:
        logger.info("Saved embedding cache %s shape=%s", cache_path, embeddings.shape)
    return embeddings, int(embeddings.shape[1])
