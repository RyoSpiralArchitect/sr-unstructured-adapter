# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tools for transforming unstructured files into unified payloads."""

from .adapter import build_payload, stream_payloads, to_unified_payload
from .messages import to_llm_messages
from .models import Payload
from .pipeline import batch_convert, convert
from .profiles import LLMPolicy, ProcessingProfile, load_processing_profile
from .schema import Block, Document, Span

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
    "ProcessingProfile",
    "LLMPolicy",
    "load_processing_profile",
]
