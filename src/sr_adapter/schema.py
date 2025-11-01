# SPDX-License-Identifier: AGPL-3.0-or-later
"""Core data structures used across the adapter pipeline (v0.2)."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Literal
from datetime import datetime
import hashlib, uuid

from pydantic import BaseModel, Field, field_validator, model_validator


# -------- Provenance & geometry --------

class BBox(BaseModel):
    """Axis-aligned bounding box in source coordinate space."""
    x0: float; y0: float; x1: float; y1: float

    @model_validator(mode="after")
    def _check_order(self):
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("BBox must satisfy x1>=x0 and y1>=y0")
        return self


class Provenance(BaseModel):
    """Where this block came from."""
    uri: Optional[str] = None           # file path / URL / s3://...
    page: Optional[int] = None          # 0-based page index if paged doc
    bbox: Optional[BBox] = None         # region in page coords
    order: Optional[int] = None         # reading order within page


# -------- Spans / Blocks / Document --------

class Span(BaseModel):
    """Annotated portion inside Block.text [codepoint offsets]."""
    start: int
    end: int
    label: Optional[str] = None

    @field_validator("end")
    @classmethod
    def _end_ge_start(cls, v: int, info):
        start = info.data.get("start", None)
        if start is not None and v < start:
            raise ValueError("Span.end must be >= start")
        return v


BlockType = Literal[
    "paragraph",
    "heading",
    "header",
    "title",
    "list_item",
    "list",
    "table",
    "figure",
    "code",
    "footnote",
    "metadata",
    "meta",
    "kv",
    "log",
    "event",
    "attachment",
    "other",
]


class Block(BaseModel):
    """Normalized chunk from a parser."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: BlockType = "paragraph"
    text: str = ""
    spans: List[Span] = Field(default_factory=list)
    attrs: Dict[str, Any] = Field(default_factory=dict)
    prov: Provenance = Field(default_factory=Provenance)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    lang: Optional[str] = None          # e.g. "en", "ja", "zh"

    @model_validator(mode="after")
    def _spans_within_text(self):
        n = len(self.text)
        for s in self.spans:
            if not (0 <= s.start <= s.end <= n):
                raise ValueError(f"Span({s.start},{s.end}) out of bounds for text length {n}")
        return self


class DocumentMeta(BaseModel):
    """Top-level document metadata."""
    uri: Optional[str] = None
    source_kind: Literal["file","url","s3","gcs","bytes"] = "file"
    title: Optional[str] = None
    mime_type: Optional[str] = None
    type: Optional[str] = None
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None
    page_count: Optional[int] = None
    languages: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metrics_parse_ms: Optional[float] = None
    metrics_normalize_ms: Optional[float] = None
    metrics_recipe_ms: Optional[float] = None
    metrics_total_ms: Optional[float] = None
    llm_escalations: int = 0
    truncated_blocks: int = 0
    block_count: int = 0
    env_no_llm: bool = False
    processing_profile: Optional[str] = None
    llm_policy: Dict[str, Any] = Field(default_factory=dict)
    runtime_text_enabled: Optional[bool] = None
    runtime_layout_enabled: Optional[bool] = None

    def __getitem__(self, key: str) -> Any:
        data = self.model_dump()
        if key not in data:
            raise KeyError(key)
        return data[key]

    def get(self, key: str, default: Any = None) -> Any:
        data = self.model_dump()
        return data.get(key, default)


class Document(BaseModel):
    """Conversion pipeline output."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    blocks: List[Block]
    meta: DocumentMeta = Field(default_factory=DocumentMeta)
    warnings: List[str] = Field(default_factory=list)


# -------- Utilities --------

def compute_checksum(data: bytes, algo: str = "blake2b") -> str:
    if algo == "blake2b":
        return hashlib.blake2b(data, digest_size=32).hexdigest()
    if algo == "sha256":
        return hashlib.sha256(data).hexdigest()
    raise ValueError(f"Unsupported algo: {algo}")

def clone_model(model: BaseModel, **updates: Any):
    """Return a copy of *model* with updates applied (Pydantic v2/v1 compatible)."""
    return model.model_copy(update=updates) if hasattr(model, "model_copy") else model.copy(update=updates)
