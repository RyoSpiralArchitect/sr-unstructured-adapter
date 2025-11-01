# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid multi-modal embeddings for document blocks."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - faiss is optional
    faiss = None  # type: ignore[assignment]

from .schema import Block, BBox


def _hash_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _normalise(values: Sequence[float]) -> List[float]:
    vec = [float(value) for value in values]
    norm = math.sqrt(sum(component * component for component in vec))
    if norm <= 0:
        return vec
    return [component / norm for component in vec]


def _layout_vector(bbox: Optional[BBox], page: Optional[int]) -> List[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0, float(page or 0)]
    width = max(1.0, bbox.x1 - bbox.x0)
    height = max(1.0, bbox.y1 - bbox.y0)
    area = width * height
    return [bbox.x0, bbox.y0, width, height, area ** 0.5, float(page or 0)]


class BlockEmbedder:
    """Generate hybrid embeddings combining text, layout, and metadata."""

    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimensions: int = 256,
        seed: int = 13,
    ) -> None:
        self.dimensions = max(32, dimensions)
        self.seed = seed
        self._model: Optional[SentenceTransformer] = None
        if SentenceTransformer is not None:
            try:  # pragma: no cover - optional heavy dependency
                self._model = SentenceTransformer(model_name)
                self.dimensions = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._model = None

    def _text_embedding(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return [0.0] * self.dimensions
        if self._model is not None:
            vector = self._model.encode(text, normalize_embeddings=True)  # type: ignore[assignment]
            return list(vector)
        rng = random.Random(self.seed ^ _hash_seed(text))
        return _normalise([rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)])

    def _metadata_embedding(self, block: Block) -> List[float]:
        attrs = block.attrs or {}
        tokens: List[float] = []
        priority_keys = ["label", "key", "value", "title"]
        for key in priority_keys:
            value = attrs.get(key)
            if isinstance(value, str) and value:
                rng = random.Random(self.seed ^ _hash_seed(f"{key}:{value}"))
                tokens.extend(rng.uniform(-0.5, 0.5) for _ in range(4))
        tokens.extend(_layout_vector(block.prov.bbox, block.prov.page))
        while len(tokens) < 32:
            tokens.append(0.0)
        return tokens[:32]

    def embed_block(self, block: Block) -> List[float]:
        text_vec = self._text_embedding(block.text or "")
        meta_vec = self._metadata_embedding(block)
        combined = text_vec[: self.dimensions]
        # Mix metadata into the tail of the vector to keep deterministic size.
        tail = combined[-len(meta_vec) :]
        padded_meta = _normalise(meta_vec)
        mixed_tail = [0.7 * t + 0.3 * m for t, m in zip(tail, padded_meta)]
        combined[-len(mixed_tail) :] = mixed_tail
        return _normalise(combined)

    def embed(self, blocks: Iterable[Block]) -> List[List[float]]:
        return [self.embed_block(block) for block in blocks]


@dataclass
class EmbeddingHit:
    score: float
    index: int
    metadata: dict[str, object]


class EmbeddingIndex:
    """In-memory similarity search for block embeddings."""

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._vectors: List[List[float]] = []
        self._metadata: List[dict[str, object]] = []
        self._faiss_index = None
        if faiss is not None and _np is not None:  # pragma: no cover - optional path
            self._faiss_index = faiss.IndexFlatIP(self.dimension)

    def add(self, vector: Sequence[float], metadata: Optional[dict[str, object]] = None) -> None:
        dense = list(vector)
        if len(dense) != self.dimension:
            raise ValueError(f"Expected vector of length {self.dimension}")
        norm = math.sqrt(sum(v * v for v in dense)) or 1.0
        dense = [v / norm for v in dense]
        self._vectors.append(dense)
        self._metadata.append(dict(metadata or {}))
        if self._faiss_index is not None:
            array = _np.array([dense], dtype="float32")
            self._faiss_index.add(array)

    def extend(self, vectors: Iterable[Sequence[float]], metadata: Iterable[dict[str, object]]) -> None:
        for vector, meta in zip(vectors, metadata):
            self.add(vector, meta)

    def search(self, query: Sequence[float], *, top_k: int = 5) -> List[EmbeddingHit]:
        dense_query = list(query)
        if len(dense_query) != self.dimension:
            raise ValueError(f"Expected query vector of length {self.dimension}")
        norm = math.sqrt(sum(v * v for v in dense_query)) or 1.0
        dense_query = [v / norm for v in dense_query]
        if self._faiss_index is not None:
            array = _np.array([dense_query], dtype="float32")
            scores, indices = self._faiss_index.search(array, top_k)
            results: List[EmbeddingHit] = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue
                results.append(
                    EmbeddingHit(
                        score=float(score),
                        index=int(index),
                        metadata=dict(self._metadata[index]),
                    )
                )
            return results

        # Fallback cosine similarity implementation
        scored: List[Tuple[float, int]] = []
        for idx, vector in enumerate(self._vectors):
            score = sum(a * b for a, b in zip(vector, dense_query))
            scored.append((score, idx))
        scored.sort(reverse=True)
        hits: List[EmbeddingHit] = []
        for score, idx in scored[:top_k]:
            hits.append(
                EmbeddingHit(score=float(score), index=idx, metadata=dict(self._metadata[idx]))
            )
        return hits


__all__ = [
    "BlockEmbedder",
    "EmbeddingHit",
    "EmbeddingIndex",
]
