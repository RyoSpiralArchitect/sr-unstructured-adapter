# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public interface for :mod:`sr_adapter` with lightweight imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, Tuple

__all__ = [
    "Payload",
    "build_payload",
    "stream_payloads",
    "to_unified_payload",
    "to_llm_messages",
    "Block",
    "Document",
    "Span",
    "convert",
    "batch_convert",
    "apply_recipe",
    "apply_recipe_block",
    "load_recipe",
    "RecipeExample",
    "RecipeSuggestion",
    "RecipeSuggester",
    "BlockEmbedder",
    "EmbeddingIndex",
    "KernelAutoTuner",
    "ProcessingProfile",
    "LLMPolicy",
    "load_processing_profile",
]

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "Payload": (".models", "Payload"),
    "build_payload": (".adapter", "build_payload"),
    "stream_payloads": (".adapter", "stream_payloads"),
    "to_unified_payload": (".adapter", "to_unified_payload"),
    "to_llm_messages": (".messages", "to_llm_messages"),
    "Block": (".schema", "Block"),
    "Document": (".schema", "Document"),
    "Span": (".schema", "Span"),
    "convert": (".pipeline", "convert"),
    "batch_convert": (".pipeline", "batch_convert"),
    "apply_recipe": (".recipe", "apply_recipe"),
    "apply_recipe_block": (".recipe", "apply_recipe_block"),
    "load_recipe": (".recipe", "load_recipe"),
    "RecipeExample": (".recipe_autogen", "RecipeExample"),
    "RecipeSuggestion": (".recipe_autogen", "RecipeSuggestion"),
    "RecipeSuggester": (".recipe_autogen", "RecipeSuggester"),
    "BlockEmbedder": (".embedding", "BlockEmbedder"),
    "EmbeddingIndex": (".embedding", "EmbeddingIndex"),
    "KernelAutoTuner": (".kernel_autotune", "KernelAutoTuner"),
    "ProcessingProfile": (".profiles", "ProcessingProfile"),
    "LLMPolicy": (".profiles", "LLMPolicy"),
    "load_processing_profile": (".profiles", "load_processing_profile"),
}

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from .adapter import build_payload, stream_payloads, to_unified_payload
    from .embedding import BlockEmbedder, EmbeddingIndex
    from .kernel_autotune import KernelAutoTuner
    from .messages import to_llm_messages
    from .models import Payload
    from .pipeline import batch_convert, convert
    from .profiles import LLMPolicy, ProcessingProfile, load_processing_profile
    from .recipe import apply_recipe, apply_recipe_block, load_recipe
    from .recipe_autogen import RecipeExample, RecipeSuggestion, RecipeSuggester
    from .schema import Block, Document, Span


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise AttributeError(name) from exc
    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - simple delegation
    return sorted(__all__)
