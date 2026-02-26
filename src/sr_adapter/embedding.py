# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid multi-modal embeddings for document blocks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
import re
from typing import Any, Iterable, List, Optional, Sequence, Tuple

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
from .confidence import semantic_confidence, structural_confidence


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def _stable_hash64(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _hash64(text: str, *, seed_hash: int) -> int:
    return _stable_hash64(text) ^ seed_hash


def _normalise(values: Sequence[float]) -> List[float]:
    vec = [float(value) for value in values]
    norm = math.sqrt(sum(component * component for component in vec))
    if norm <= 0:
        return vec
    return [component / norm for component in vec]


def _hash_embed_tokens(
    tokens: Sequence[str],
    *,
    dim: int,
    seed_hash: int,
) -> List[float]:
    if dim <= 0:
        return []
    vec = [0.0] * dim
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        value = _hash64(token, seed_hash=seed_hash)
        idx = int(value % dim)
        sign = 1.0 if (value >> 63) == 0 else -1.0
        vec[idx] += sign
    return _normalise(vec)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xAC00 <= code <= 0xD7AF  # Hangul syllables
    )


def _is_semantic_char(char: str) -> bool:
    return char.isalnum() or _is_cjk(char)


def _tokenize(text: str, *, max_tokens: int) -> List[str]:
    lowered = (text or "").strip().lower()
    if not lowered:
        return []

    if any(_is_cjk(ch) for ch in lowered):
        chars = [ch for ch in lowered if _is_semantic_char(ch)]
        if len(chars) <= 2:
            return chars[:max_tokens]
        return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)][:max_tokens]

    words = _WORD_RE.findall(lowered)
    if words:
        return words[:max_tokens]

    chars = [ch for ch in lowered if _is_semantic_char(ch)]
    if len(chars) <= 2:
        return chars[:max_tokens]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)][:max_tokens]


def _layout_vector(bbox: Optional[BBox], page: Optional[int]) -> List[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, float(page or 0)]
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
        text_max_tokens: int = 256,
        metadata_dimensions: int = 32,
        metadata_weight: float = 0.3,
        use_sentence_transformers: bool = True,
    ) -> None:
        self.dimensions = max(32, int(dimensions))
        self.seed = int(seed)
        self._seed_hash = _stable_hash64(f"seed:{self.seed}")
        self.text_max_tokens = max(1, int(text_max_tokens))
        self.metadata_dimensions = max(8, int(metadata_dimensions))
        if self.metadata_dimensions >= self.dimensions:
            self.metadata_dimensions = max(8, self.dimensions // 2)
        self.metadata_weight = float(max(0.0, min(1.0, metadata_weight)))
        self._model: Optional[SentenceTransformer] = None
        if use_sentence_transformers and SentenceTransformer is not None:
            try:  # pragma: no cover - optional heavy dependency
                self._model = SentenceTransformer(model_name)
                self.dimensions = int(self._model.get_sentence_embedding_dimension())
            except Exception:
                self._model = None
        if self.metadata_dimensions >= self.dimensions:
            self.metadata_dimensions = max(8, self.dimensions // 2)

    def _text_embedding(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return [0.0] * self.dimensions
        if self._model is not None:
            vector = self._model.encode(text, normalize_embeddings=True)  # type: ignore[assignment]
            return list(vector)
        tokens = _tokenize(text, max_tokens=self.text_max_tokens)
        return _hash_embed_tokens(tokens, dim=self.dimensions, seed_hash=self._seed_hash)

    def _metadata_embedding(self, block: Block, *, semantic: object | None = None) -> List[float]:
        attrs = block.attrs or {}
        token_dim = max(4, self.metadata_dimensions // 2)
        numeric_dim = max(1, self.metadata_dimensions - token_dim)

        tokens: List[str] = [f"type={block.type}"]
        if block.lang:
            tokens.append(f"lang={block.lang}")

        priority_keys = ["label", "key", "value", "title", "path_r"]
        for key in priority_keys:
            value = attrs.get(key)
            if isinstance(value, str) and value.strip():
                for token in _tokenize(value, max_tokens=64):
                    tokens.append(f"{key}:{token}")

        token_vec = _hash_embed_tokens(tokens, dim=token_dim, seed_hash=self._seed_hash)

        struct_conf = float(structural_confidence(block))
        sem_conf = semantic_confidence(block)
        sem_density = None
        sem_anisotropy = None
        sem_spread = None
        if semantic is not None:
            sem_conf = getattr(semantic, "confidence", sem_conf)
            sem_density = getattr(semantic, "density", None)
            sem_anisotropy = getattr(semantic, "anisotropy", None)
            sem_spread = getattr(semantic, "spread", None)
        if isinstance(attrs, dict):
            sem_density = attrs.get("semantic_density") if sem_density is None else sem_density
            sem_anisotropy = attrs.get("semantic_anisotropy") if sem_anisotropy is None else sem_anisotropy
            sem_spread = attrs.get("semantic_spread") if sem_spread is None else sem_spread

        def _safe_float(value: object) -> float:
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return 0.0
            if math.isnan(number) or math.isinf(number):
                return 0.0
            return float(number)

        numeric: List[float] = [
            struct_conf,
            float(sem_conf) if sem_conf is not None else 0.0,
            _safe_float(sem_density),
            max(-1.0, min(1.0, _safe_float(sem_anisotropy))),
            _safe_float(sem_spread),
            *_layout_vector(block.prov.bbox, block.prov.page),
        ]
        numeric_vec = [float(value) for value in numeric[:numeric_dim]]
        if len(numeric_vec) < numeric_dim:
            numeric_vec.extend([0.0] * (numeric_dim - len(numeric_vec)))
        return _normalise([*token_vec, *numeric_vec])

    def embed_block(self, block: Block, *, semantic: object | None = None) -> List[float]:
        text_vec = self._text_embedding(block.text or "")
        meta_vec = self._metadata_embedding(block, semantic=semantic)
        combined = text_vec[: self.dimensions]
        # Mix metadata into the tail of the vector to keep deterministic size.
        tail = combined[-len(meta_vec) :]
        padded_meta = _normalise(meta_vec)
        text_weight = 1.0 - self.metadata_weight
        mixed_tail = [text_weight * t + self.metadata_weight * m for t, m in zip(tail, padded_meta)]
        combined[-len(mixed_tail) :] = mixed_tail
        return _normalise(combined)

    def embed(self, blocks: Iterable[Block]) -> List[List[float]]:
        return [self.embed_block(block) for block in blocks]

    def embed_with_context(
        self,
        blocks: Sequence[Block],
        *,
        window: int = 24,
        top_k: int = 3,
        dim: int = 256,
        max_tokens: int = 256,
    ) -> List[List[float]]:
        """Embed blocks while injecting semantic field statistics.

        This method uses :func:`sr_adapter.semantic.score_blocks` to compute a
        deterministic semantic proxy and mixes it into the metadata tail.
        """

        from .semantic import score_blocks

        scores = score_blocks(
            blocks,
            window=window,
            top_k=top_k,
            dim=dim,
            max_tokens=max_tokens,
        )
        return [self.embed_block(block, semantic=score) for block, score in zip(blocks, scores)]


@dataclass
class EmbeddingHit:
    score: float
    index: int
    metadata: dict[str, Any]


class EmbeddingIndex:
    """In-memory similarity search for block embeddings."""

    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._vectors: List[List[float]] = []
        self._metadata: List[dict[str, Any]] = []
        self._faiss_index = None
        if faiss is not None and _np is not None:  # pragma: no cover - optional path
            self._faiss_index = faiss.IndexFlatIP(self.dimension)

    def add(self, vector: Sequence[float], metadata: Optional[dict[str, Any]] = None) -> None:
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

    def extend(self, vectors: Iterable[Sequence[float]], metadata: Iterable[dict[str, Any]]) -> None:
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
