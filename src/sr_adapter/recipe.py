"""Recipe handling for domain specific post-processing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import yaml

from .schema import Block


@dataclass
class Pattern:
    when: re.Pattern[str]
    as_type: str
    attrs: Dict[str, str]
    confidence: Optional[float]

    def matches(self, text: str) -> bool:
        return bool(self.when.search(text))


@dataclass
class Recipe:
    name: str
    patterns: List[Pattern]
    fallback_type: Optional[str]
    fallback_attrs: Dict[str, str]
    fallback_confidence: Optional[float]
    llm: Dict[str, object]


@lru_cache(maxsize=32)
def load_recipe(name: str) -> Recipe:
    data = _load_recipe_data(name)
    patterns = [
        Pattern(
            when=re.compile(item["when"]),
            as_type=item.get("as", "paragraph"),
            attrs=item.get("attrs", {}),
            confidence=item.get("confidence"),
        )
        for item in data.get("patterns", [])
    ]
    fallback = data.get("fallback", {})
    return Recipe(
        name=data.get("name", name),
        patterns=patterns,
        fallback_type=fallback.get("as"),
        fallback_attrs=fallback.get("attrs", {}),
        fallback_confidence=fallback.get("confidence"),
        llm=data.get("llm", {}),
    )


def apply_recipe(blocks: Iterable[Block], recipe_name: str) -> List[Block]:
    recipe = load_recipe(recipe_name)
    result: List[Block] = []
    for block in blocks:
        updated = block
        for pattern in recipe.patterns:
            if pattern.matches(block.text):
                updated = _apply_pattern(updated, pattern)
                break
        else:
            updated = _apply_fallback(updated, recipe)
        result.append(updated)
    return result


def _apply_pattern(block: Block, pattern: Pattern) -> Block:
    attrs = dict(block.attrs)
    attrs.update(pattern.attrs)
    return block.with_updates(type=pattern.as_type, confidence=pattern.confidence, attrs=attrs)


def _apply_fallback(block: Block, recipe: Recipe) -> Block:
    attrs = dict(block.attrs)
    attrs.update(recipe.fallback_attrs)
    return block.with_updates(
        type=recipe.fallback_type or block.type,
        confidence=recipe.fallback_confidence,
        attrs=attrs,
    )


def _load_recipe_data(name: str) -> Dict[str, object]:
    from importlib import resources

    package = "sr_adapter.recipes"
    if name.endswith(".yaml"):
        target = name
    else:
        target = f"{name}.yaml"
    try:
        with resources.files(package).joinpath(target).open("rb") as fh:  # type: ignore[arg-type]
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Recipe '{name}' was not found")
