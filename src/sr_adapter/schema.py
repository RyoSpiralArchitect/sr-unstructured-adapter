"""Core data structures used across the adapter pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field


class Span(BaseModel):
    """Represents an annotated portion of a block of text."""

    start: int
    end: int
    label: Optional[str] = None


class Block(BaseModel):
    """A normalized chunk of content produced by a parser."""

    type: str = Field(default="paragraph", description="Semantic type of the block")
    text: str = Field(default="", description="Normalized textual content")
    spans: List[Span] = Field(default_factory=list, description="Inline annotations")
    attrs: Dict[str, str] = Field(default_factory=dict, description="Structured metadata")
    source: Optional[str] = Field(
        default=None, description="Origin of the block (file path, page index, etc.)"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Document(BaseModel):
    """Top level representation returned by the conversion pipeline."""

    blocks: List[Block]
    meta: Dict[str, str] = Field(default_factory=dict)


ModelType = TypeVar("ModelType", bound=BaseModel)


def clone_model(model: ModelType, **updates: Any) -> ModelType:
    """Return a copy of *model* with ``updates`` applied."""

    if hasattr(model, "model_copy"):
        return model.model_copy(update=updates)  # type: ignore[attr-defined]
    return model.copy(update=updates)

