# SPDX-License-Identifier: AGPL-3.0-or-later
"""Confidence utilities for escalation gating.

The adapter treats "confidence" as a multi-signal concept:

- Structural confidence: deterministic + layout-derived reliability of the block
  structure/type assignment.
- Semantic confidence: local embedding-density proxy derived from neighbouring
  blocks (see :mod:`sr_adapter.semantic`).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from .schema import Block, clone_model


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return float(number)


def structural_confidence(block: Block) -> float:
    """Compute structural confidence for *block*.

    The baseline is ``Block.confidence`` (recipe/type confidence). When layout
    confidence is available it is treated as a hard upper-bound.
    """

    score = float(block.confidence)
    if isinstance(block.attrs, dict):
        layout = _safe_float(block.attrs.get("layout_confidence"))
        if layout is not None:
            score = min(score, layout)
    return max(0.0, min(1.0, score))


def semantic_confidence(block: Block) -> float | None:
    """Read semantic confidence from ``block.attrs['semantic_confidence']`` if present."""

    if not isinstance(block.attrs, dict):
        return None
    score = _safe_float(block.attrs.get("semantic_confidence"))
    if score is None:
        score = _safe_float(block.attrs.get("confidence_semantic"))
    return score


def annotate_structural_confidence(
    blocks: Sequence[Block],
    *,
    key: str = "confidence_structural",
    overwrite: bool = False,
    precision: int = 4,
) -> List[Block]:
    """Attach structural confidence under *key* in block attrs (non-destructive)."""

    enriched: List[Block] = []
    for block in blocks:
        attrs = dict(block.attrs)
        if overwrite or key not in attrs:
            attrs[key] = round(structural_confidence(block), precision)
        enriched.append(clone_model(block, attrs=attrs))
    return enriched


def annotate_semantic_alias(
    blocks: Sequence[Block],
    *,
    source_key: str = "semantic_confidence",
    target_key: str = "confidence_semantic",
    overwrite: bool = False,
    precision: int = 4,
) -> List[Block]:
    """Optionally mirror semantic confidence under a unified naming key."""

    enriched: List[Block] = []
    for block in blocks:
        if not isinstance(block.attrs, dict):
            enriched.append(block)
            continue
        attrs = dict(block.attrs)
        if not overwrite and target_key in attrs:
            enriched.append(block)
            continue
        score = _safe_float(attrs.get(source_key))
        if score is None:
            enriched.append(block)
            continue
        attrs[target_key] = round(max(0.0, min(1.0, score)), precision)
        enriched.append(clone_model(block, attrs=attrs))
    return enriched


def ensure_confidence_fields(
    blocks: Iterable[Block],
) -> List[Block]:
    """Best-effort helper to ensure both confidence keys exist when possible."""

    block_list = list(blocks)
    block_list = annotate_structural_confidence(block_list, overwrite=False)
    block_list = annotate_semantic_alias(block_list, overwrite=False)
    return block_list


__all__ = [
    "annotate_semantic_alias",
    "annotate_structural_confidence",
    "ensure_confidence_fields",
    "semantic_confidence",
    "structural_confidence",
]
