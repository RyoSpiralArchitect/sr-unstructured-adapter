# SPDX-License-Identifier: AGPL-3.0-or-later
"""Utilities for loading and applying YAML based recipes with validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

try:  # pragma: no cover - maintain compatibility with Pydantic v1
    from pydantic import ConfigDict  # type: ignore
except ImportError:  # pragma: no cover - for Pydantic v1
    ConfigDict = None  # type: ignore[assignment]

from .schema import Block, clone_model


class _RecipePatternModel(BaseModel):
    """Pydantic representation of a single recipe rule."""

    when: str
    target_type: str = Field(alias="as")
    attrs: Dict[str, str] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(populate_by_name=True, extra="forbid")  # type: ignore[assignment]
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            allow_population_by_field_name = True
            extra = "forbid"


class _RecipeFallbackModel(BaseModel):
    """Validation schema for fallback behaviour."""

    target_type: str = Field(alias="as")
    attrs: Dict[str, str] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(populate_by_name=True, extra="forbid")  # type: ignore[assignment]
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            allow_population_by_field_name = True
            extra = "forbid"


class _RecipeDocumentModel(BaseModel):
    """Top-level recipe definition parsed from YAML."""

    name: Optional[str] = None
    patterns: Sequence[_RecipePatternModel] = Field(default_factory=list)
    fallback: Optional[_RecipeFallbackModel] = None
    llm: Dict[str, Any] = Field(default_factory=dict)

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="forbid")  # type: ignore[assignment]
    else:  # pragma: no cover - Pydantic v1 fallback
        class Config:
            extra = "forbid"


@dataclass(frozen=True)
class RecipePattern:
    """Compiled representation of a pattern entry."""

    regex: re.Pattern[str]
    target_type: str
    attrs: Mapping[str, str]
    confidence: Optional[float]


@dataclass(frozen=True)
class RecipeFallback:
    """Fallback target applied when no explicit pattern matches."""

    target_type: str
    attrs: Mapping[str, str]
    confidence: Optional[float]


@dataclass(frozen=True, init=False)
class RecipeConfig:
    """Validated recipe configuration used by the pipeline."""

    name: str
    patterns: Sequence[RecipePattern]
    fallback: Optional[RecipeFallback]
    llm: Mapping[str, Any]

    def __init__(
        self,
        *,
        name: str,
        patterns: Sequence[RecipePattern],
        fallback: Optional[RecipeFallback] = None,
        fallback_type: Optional[str] = None,
        fallback_confidence: Optional[float] = None,
        fallback_attrs: Optional[Mapping[str, str]] = None,
        llm: Mapping[str, Any] = (),
    ) -> None:
        if fallback is None and fallback_type:
            attrs = dict(fallback_attrs or {})
            fallback = RecipeFallback(
                target_type=fallback_type,
                attrs=attrs,
                confidence=fallback_confidence,
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "patterns", tuple(patterns))
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "llm", dict(llm))

    @property
    def fallback_type(self) -> Optional[str]:
        return self.fallback.target_type if self.fallback else None

    @property
    def fallback_confidence(self) -> Optional[float]:
        return self.fallback.confidence if self.fallback else None

    @property
    def fallback_attrs(self) -> Dict[str, str]:
        return dict(self.fallback.attrs) if self.fallback else {}


def _compile_pattern(entry: _RecipePatternModel) -> RecipePattern:
    return RecipePattern(
        regex=re.compile(entry.when, re.MULTILINE),
        target_type=entry.target_type,
        attrs=dict(entry.attrs),
        confidence=entry.confidence,
    )


def _load_recipe_dict(name: str) -> Mapping[str, Any]:
    from importlib import resources

    package = "sr_adapter.recipes"
    with resources.files(package).joinpath(f"{name}.yaml").open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Recipe '{name}' must be a mapping at the top level")
    return dict(data)


def _build_recipe_config(name: str, payload: Mapping[str, Any]) -> RecipeConfig:
    try:
        model = _RecipeDocumentModel(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid recipe '{name}': {exc}") from exc

    patterns = [_compile_pattern(pattern) for pattern in model.patterns]
    fallback = (
        RecipeFallback(
            target_type=model.fallback.target_type,
            attrs=dict(model.fallback.attrs),
            confidence=model.fallback.confidence,
        )
        if model.fallback
        else None
    )
    recipe_name = model.name or name
    llm_payload: Mapping[str, Any] = dict(model.llm)
    return RecipeConfig(name=recipe_name, patterns=patterns, fallback=fallback, llm=llm_payload)


@lru_cache(maxsize=16)
def load_recipe(name: str) -> RecipeConfig:
    data = _load_recipe_dict(name)
    return _build_recipe_config(name, data)


def _apply_recipe_to_block(block: Block, recipe: RecipeConfig) -> Block:
    updated = block
    matched = False
    for pattern in recipe.patterns:
        if pattern.regex.search(block.text):
            data: Dict[str, Any] = {"type": pattern.target_type}
            if pattern.attrs:
                attrs = dict(block.attrs)
                attrs.update(pattern.attrs)
                data["attrs"] = attrs
            if pattern.confidence is not None:
                data["confidence"] = pattern.confidence
            updated = clone_model(block, **data)
            matched = True
            break
    fallback = recipe.fallback
    if (
        not matched
        and fallback is not None
        and block.type in {"paragraph", "other"}
    ):
        data = {"type": fallback.target_type}
        attrs = dict(block.attrs)
        if fallback.attrs:
            attrs.update(fallback.attrs)
        if attrs:
            data["attrs"] = attrs
        if fallback.confidence is not None:
            data["confidence"] = fallback.confidence
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

