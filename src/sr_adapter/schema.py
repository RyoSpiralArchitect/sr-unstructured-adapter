"""Core Pydantic models describing the normalized document schema."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Span(BaseModel):
    """A labeled span within a block of text."""

    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    label: Optional[str] = None

    @classmethod
    def empty(cls) -> "Span":
        """Return a zero-width span."""

        return cls(start=0, end=0)


class Block(BaseModel):
    """Normalized unit of content extracted from a document."""

    type: str
    text: str
    spans: List[Span] = Field(default_factory=list)
    attrs: Dict[str, str] = Field(default_factory=dict)
    source: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def with_updates(
        self,
        *,
        type: Optional[str] = None,
        confidence: Optional[float] = None,
        attrs: Optional[Dict[str, str]] = None,
    ) -> "Block":
        """Return a shallow copy with optional updates applied."""

        try:
            data = self.model_dump()
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            data = self.dict()
        if type is not None:
            data["type"] = type
        if confidence is not None:
            data["confidence"] = max(0.0, min(1.0, confidence))
        if attrs:
            merged = dict(data.get("attrs", {}))
            merged.update(attrs)
            data["attrs"] = merged
        return Block(**data)


class Document(BaseModel):
    """Collection of blocks along with document level metadata."""

    blocks: List[Block] = Field(default_factory=list)
    meta: Dict[str, str] = Field(default_factory=dict)

    def add_block(self, block: Block) -> None:
        """Append a block to the document."""

        self.blocks.append(block)

    def to_dict(self) -> Dict[str, object]:  # pragma: no cover - simple passthrough
        """Return a JSON-serialisable representation."""

        try:
            return self.model_dump()
        except AttributeError:  # pragma: no cover - pydantic v1 fallback
            return self.dict()
