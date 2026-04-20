from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class EmbedderProtocol(Protocol):
    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        ...


@dataclass
class SentenceTransformerEmbedder:
    model_name: str
    device: str | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        # sentence-transformers handles truncation internally for many models.
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)


@dataclass
class HFMeanPoolingEmbedder:
    model_name: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked = token_embeddings * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: list[str], batch_size: int, max_length: int) -> np.ndarray:
        torch = self.torch
        all_vecs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                vec = pooled.detach().cpu().numpy().astype(np.float32)
            all_vecs.append(vec)
        return np.vstack(all_vecs) if all_vecs else np.zeros((0, 1), dtype=np.float32)


def create_embedder(cfg: dict[str, Any]) -> EmbedderProtocol:
    backend = str(cfg.get("backend", "sentence_transformers"))
    model_name = str(cfg.get("model_name"))

    if backend == "sentence_transformers":
        device = cfg.get("device")
        return SentenceTransformerEmbedder(model_name=model_name, device=device)

    if backend == "hf_encoder":
        import torch

        if cfg.get("device"):
            device = str(cfg.get("device"))
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return HFMeanPoolingEmbedder(model_name=model_name, device=device)

    raise ValueError(f"Unsupported embedding backend: {backend}")


def _make_cache_key(
    case_ids: list[str],
    texts: list[str],
    embed_cfg: dict[str, Any],
    namespace: str,
) -> str:
    h = hashlib.sha256()
    h.update(namespace.encode("utf-8"))
    h.update(str(embed_cfg.get("backend", "")).encode("utf-8"))
    h.update(str(embed_cfg.get("model_name", "")).encode("utf-8"))
    h.update(str(embed_cfg.get("max_length", "")).encode("utf-8"))
    for cid, txt in zip(case_ids, texts):
        h.update(cid.encode("utf-8"))
        h.update(str(len(txt)).encode("utf-8"))
    return h.hexdigest()[:20]


def load_or_compute_embeddings(
    case_ids: list[str],
    texts: list[str],
    embed_cfg: dict[str, Any],
    cache_dir: str | Path,
    namespace: str,
    logger: Any | None = None,
) -> np.ndarray:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _make_cache_key(case_ids, texts, embed_cfg, namespace)
    cache_path = cache_root / f"{namespace}_{key}.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        ids = data["case_ids"].astype(str).tolist()
        if ids == [str(c) for c in case_ids]:
            emb = data["embeddings"].astype(np.float32)
            if logger is not None:
                logger.info("Loaded embeddings cache: %s shape=%s", cache_path, emb.shape)
            return emb

    embedder = create_embedder(embed_cfg)
    emb = embedder.encode(
        texts=texts,
        batch_size=int(embed_cfg.get("batch_size", 32)),
        max_length=int(embed_cfg.get("max_length", 512)),
    ).astype(np.float32)

    np.savez_compressed(
        cache_path,
        case_ids=np.array(case_ids),
        embeddings=emb,
    )
    if logger is not None:
        logger.info("Saved embeddings cache: %s shape=%s", cache_path, emb.shape)
    return emb
