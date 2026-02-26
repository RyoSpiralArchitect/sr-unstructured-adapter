# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic semantic scoring helpers.

This module intentionally avoids heavyweight ML dependencies. It provides a
cheap, deterministic proxy for "semantic confidence" that can be used alongside
rule-based confidence when deciding which blocks to escalate to an LLM.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from .schema import Block, clone_model


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


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
    lowered = text.strip().lower()
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


def _stable_hash64(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def _embed_sparse(tokens: Sequence[str], *, dim: int) -> Dict[int, float]:
    if not tokens:
        return {}
    counts: MutableMapping[int, float] = {}
    for token in tokens:
        value = _stable_hash64(token)
        idx = int(value % dim)
        sign = 1.0 if (value >> 63) == 0 else -1.0
        counts[idx] = counts.get(idx, 0.0) + sign
    norm = math.sqrt(sum(val * val for val in counts.values()))
    if norm <= 0.0:
        return {}
    return {idx: val / norm for idx, val in counts.items()}


def _cosine_sparse(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(val * b.get(idx, 0.0) for idx, val in a.items()))


@dataclass(frozen=True)
class SemanticScore:
    confidence: float
    density: float
    anisotropy: float
    spread: float


def score_blocks(
    blocks: Sequence[Block],
    *,
    window: int = 24,
    top_k: int = 3,
    dim: int = 256,
    max_tokens: int = 256,
) -> List[SemanticScore]:
    """Compute semantic scores for *blocks* using a deterministic embedding proxy."""

    vectors: List[Dict[int, float]] = []
    lengths: List[int] = []
    for block in blocks:
        text = block.text or ""
        lengths.append(len(text.strip()))
        tokens = _tokenize(text, max_tokens=max_tokens)
        vectors.append(_embed_sparse(tokens, dim=dim))

    n = len(blocks)
    w = max(1, int(window))
    k = max(1, int(top_k))
    scores: List[SemanticScore] = []
    for idx in range(n):
        vec = vectors[idx]
        sims: List[float] = []
        start = max(0, idx - w)
        end = min(n, idx + w + 1)
        for j in range(start, end):
            if j == idx:
                continue
            sim = _cosine_sparse(vec, vectors[j])
            if sim > 0.0:
                sims.append(sim)
        sims.sort(reverse=True)
        top = sims[:k]
        density = sum(top) / len(top) if top else 0.0
        mean = sum(sims) / len(sims) if sims else 0.0
        spread = 0.0
        if len(sims) >= 2:
            var = sum((val - mean) ** 2 for val in sims) / float(len(sims))
            spread = math.sqrt(max(0.0, var))
        anisotropy = (top[0] - mean) if top else 0.0

        length = lengths[idx]
        if not vec or length <= 0:
            confidence = 0.5
        else:
            length_factor = math.sqrt(min(1.0, length / 200.0))
            confidence = 0.2 + 0.8 * density * length_factor
            confidence = max(0.0, min(1.0, confidence))

        scores.append(
            SemanticScore(
                confidence=float(confidence),
                density=float(min(1.0, max(0.0, density))),
                anisotropy=float(anisotropy),
                spread=float(min(1.0, max(0.0, spread))),
            )
        )
    return scores


def annotate_semantic_confidence(
    blocks: Sequence[Block],
    *,
    window: int = 24,
    top_k: int = 3,
    dim: int = 256,
    max_tokens: int = 256,
    prefix: str = "semantic",
) -> List[Block]:
    """Attach semantic confidence metrics to blocks under ``<prefix>_*`` keys."""

    if not blocks:
        return []
    scores = score_blocks(
        blocks,
        window=window,
        top_k=top_k,
        dim=dim,
        max_tokens=max_tokens,
    )
    enriched: List[Block] = []
    for block, score in zip(blocks, scores, strict=False):
        attrs = dict(block.attrs)
        attrs[f"{prefix}_confidence"] = round(score.confidence, 4)
        attrs[f"{prefix}_density"] = round(score.density, 4)
        attrs[f"{prefix}_anisotropy"] = round(score.anisotropy, 4)
        attrs[f"{prefix}_spread"] = round(score.spread, 4)
        attrs.setdefault("confidence_structural", block.confidence)
        enriched.append(clone_model(block, attrs=attrs))
    return enriched


SemanticAnnotator = Callable[..., List[Block]]


_SEMANTIC_ANNOTATORS: Dict[str, SemanticAnnotator] = {}


def register_semantic_annotator(
    name: str,
    annotator: SemanticAnnotator,
    *,
    overwrite: bool = False,
) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("Semantic annotator name must be non-empty")
    existing = _SEMANTIC_ANNOTATORS.get(key)
    if existing is not None and existing is not annotator and not overwrite:
        raise ValueError(f"Semantic annotator '{name}' is already registered")
    _SEMANTIC_ANNOTATORS[key] = annotator


def list_semantic_annotators() -> tuple[str, ...]:
    _ensure_builtin_semantics()
    _load_entrypoint_semantics()
    return tuple(sorted(_SEMANTIC_ANNOTATORS))


def get_semantic_annotator(name: str | None = None) -> SemanticAnnotator:
    _ensure_builtin_semantics()
    _load_entrypoint_semantics()
    key = (name or "hash-v1").strip().lower()
    if key not in _SEMANTIC_ANNOTATORS:
        raise KeyError(f"Unknown semantic annotator '{name}'. Available: {', '.join(list_semantic_annotators())}")
    return _SEMANTIC_ANNOTATORS[key]


@lru_cache(maxsize=1)
def _load_entrypoint_semantics() -> None:
    """Load semantic annotators registered via packaging entry points."""

    try:
        from importlib.metadata import entry_points
    except Exception:
        return

    try:
        groups = entry_points()
    except Exception:
        return

    if hasattr(groups, "select"):
        candidates = groups.select(group="sr_adapter.semantic_annotators")
    else:  # pragma: no cover - legacy importlib.metadata behavior
        candidates = groups.get("sr_adapter.semantic_annotators", [])  # type: ignore[attr-defined]

    for ep in candidates:
        try:
            loaded = ep.load()
        except Exception:
            continue
        if callable(loaded):
            register_semantic_annotator(ep.name, loaded, overwrite=False)


@lru_cache(maxsize=1)
def _ensure_builtin_semantics() -> None:
    register_semantic_annotator("hash-v1", annotate_semantic_confidence, overwrite=False)


def import_semantic_module(module_path: str) -> None:
    """Import a module for side-effect registration.

    This is a lightweight alternative to entry points when running in ad-hoc
    environments (research notebooks, internal pipelines).
    """

    module_path = str(module_path).strip()
    if not module_path:
        return
    importlib.import_module(module_path)


__all__ = [
    "SemanticScore",
    "annotate_semantic_confidence",
    "get_semantic_annotator",
    "import_semantic_module",
    "list_semantic_annotators",
    "register_semantic_annotator",
    "score_blocks",
]
