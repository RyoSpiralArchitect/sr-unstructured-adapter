"""LLM delegation stubs used for low-confidence escalation."""

from __future__ import annotations

from typing import Iterable, List

from .schema import Block
from .recipe import load_recipe


def escalate_low_conf(blocks: Iterable[Block], recipe_name: str) -> List[Block]:
    """Return blocks unchanged while reserving hooks for LLM delegation.

    The initial implementation keeps the interface but does not perform any
    network calls. Recipes may opt-in to LLM escalation by setting
    ``llm.enable``; once a delegate is available we can replace this stub with
    a real implementation.
    """

    recipe = load_recipe(recipe_name)
    if not recipe.llm or not recipe.llm.get("enable"):
        return list(blocks)

    # Placeholder behaviour – simply return the original blocks for now.
    return list(blocks)

