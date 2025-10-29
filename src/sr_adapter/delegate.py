"""Placeholder implementation of low-confidence block escalation."""

from __future__ import annotations

from typing import Iterable, List

from .recipe import load_recipe
from .schema import Block


def escalate_low_conf(blocks: Iterable[Block], recipe_name: str) -> List[Block]:
    """Return blocks unchanged while marking low-confidence candidates.

    The real implementation would call an external LLM. For now we simply
    annotate blocks that would be escalated so downstream tooling can observe
    the potential hand-off points.
    """

    recipe = load_recipe(recipe_name)
    min_conf = float(recipe.llm.get("min_conf", 0.0)) if recipe.llm else 0.0
    annotated: List[Block] = []
    for block in blocks:
        if block.confidence < min_conf:
            attrs = dict(block.attrs)
            attrs.setdefault("llm_escalated", "true")
            annotated.append(block.with_updates(attrs=attrs, confidence=min_conf))
        else:
            annotated.append(block)
    return annotated
