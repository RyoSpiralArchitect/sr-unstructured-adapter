# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tools for transforming unstructured files into unified payloads."""

from .adapter import build_payload, stream_payloads, to_unified_payload
from .messages import to_llm_messages
from .models import Payload
from .pipeline import batch_convert, convert
from .profiles import LLMPolicy, ProcessingProfile, load_processing_profile
from .schema import Block, Document, Span
from .recipe import apply_recipe, apply_recipe_block, load_recipe
from .recipe_autogen import RecipeExample, RecipeSuggestion, RecipeSuggester
from .embedding import BlockEmbedder, EmbeddingIndex
from .kernel_autotune import KernelAutoTuner

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
