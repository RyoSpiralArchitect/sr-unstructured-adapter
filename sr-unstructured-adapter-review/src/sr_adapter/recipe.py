# SPDX-License-Identifier: AGPL-3.0-or-later
"""Utilities for loading and applying YAML based recipes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Union

import yaml

from .schema import Block, clone_model


@dataclass
class RecipePattern:
    regex: re.Pattern[str]
    target_type: str
    attrs: Dict[str, str]
    confidence: Optional[float]


@dataclass
class RecipeConfig:
    name: str
    patterns: List[RecipePattern]
    fallback_type: Optional[str]
    fallback_confidence: Optional[float]
    fallback_attrs: Dict[str, str]
    llm: Dict[str, object]


def _compile_pattern(entry: Dict[str, object]) -> RecipePattern:
    return RecipePattern(
        regex=re.compile(entry["when"], re.MULTILINE),
        target_type=entry["as"],
        attrs=dict(entry.get("attrs", {})),
        confidence=entry.get("confidence"),
    )


def _load_recipe_dict(name: str) -> Dict[str, object]:
    from importlib import resources

    package = "sr_adapter.recipes"
    with resources.files(package).joinpath(f"{name}.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=16)
def load_recipe(name: str) -> RecipeConfig:
    data = _load_recipe_dict(name)
    patterns = [_compile_pattern(entry) for entry in data.get("patterns", [])]
    fallback = data.get("fallback", {})
    llm = data.get("llm", {})
    return RecipeConfig(
        name=data.get("name", name),
        patterns=patterns,
        fallback_type=fallback.get("as"),
        fallback_confidence=fallback.get("confidence"),
        fallback_attrs=dict(fallback.get("attrs", {})),
        llm=dict(llm),
    )


def _apply_recipe_to_block(block: Block, recipe: RecipeConfig) -> Block:
    updated = block
    matched = False
    for pattern in recipe.patterns:
        if pattern.regex.search(block.text):
            data = {"type": pattern.target_type}
            if pattern.attrs:
                attrs = dict(block.attrs)
                attrs.update(pattern.attrs)
                data["attrs"] = attrs
            if pattern.confidence is not None:
                data["confidence"] = pattern.confidence
            updated = clone_model(block, **data)
            matched = True
            break
    if (
        not matched
        and recipe.fallback_type
        and block.type in {"paragraph", "other"}
    ):
        data = {"type": recipe.fallback_type}
        attrs = dict(block.attrs)
        if recipe.fallback_attrs:
            attrs.update(recipe.fallback_attrs)
            data["attrs"] = attrs
        if recipe.fallback_confidence is not None:
            data["confidence"] = recipe.fallback_confidence
        updated = clone_model(block, **data)
    return updated


def apply_recipe_block(block: Block, recipe: Union[RecipeConfig, str]) -> Block:
    if isinstance(recipe, str):
        recipe = load_recipe(recipe)
    return _apply_recipe_to_block(block, recipe)


def apply_recipe(blocks: Iterable[Block], recipe_name: str) -> List[Block]:
    """Apply the recipe named *recipe_name* to *blocks*."""

    recipe = load_recipe(recipe_name)
    return [_apply_recipe_to_block(block, recipe) for block in blocks]

